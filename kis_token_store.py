"""
KIS OpenAPI 토큰 캐시 (Phase 4-4)
====================================
한투 토큰은 발급 후 24시간 유효하므로, morning(09:05) 발급분을
closing(15:10)/evening(18:30) 및 다음날 morning까지 재사용한다.

보안:
  - 토큰 파일(state/kis_token.json)은 민감 정보이므로 절대 commit 금지
  - state/.gitignore 에 kis_token.json 반드시 추가
  - GitHub Actions 캐시만으로 job 간 전달 (캐시는 리포지토리 소유자만 접근)
  - 파일 권한 0o600 (소유자만 읽기/쓰기)

사용:
    from kis_token_store import get_or_refresh_token

    token = get_or_refresh_token(
        api_key=KIS_API_KEY,
        api_secret=KIS_API_SECRET,
        base_url=KIS_BASE_URL,
        state_dir="state",
    )
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger(__name__)

TOKEN_FILE = "kis_token.json"
# 만료 직전 여유: 만료 30분 전에는 재발급 (장 시간 중 만료 방지)
SAFETY_MARGIN_MIN = 30


def _token_path(state_dir: str) -> Path:
    return Path(state_dir) / TOKEN_FILE


def load_cached_token(state_dir: str) -> Optional[str]:
    """유효한 캐시 토큰이 있으면 반환, 없거나 만료 임박이면 None."""
    path = _token_path(state_dir)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        expires_at = datetime.fromisoformat(data["expires_at"])
        remaining = expires_at - datetime.now()

        if remaining > timedelta(minutes=SAFETY_MARGIN_MIN):
            log.info(f"[KIS] 캐시 토큰 사용 (잔여 {remaining.total_seconds()/3600:.1f}h)")
            return data["access_token"]

        log.info(f"[KIS] 캐시 토큰 만료 임박 ({remaining}), 재발급 필요")
        return None
    except Exception as e:
        log.warning(f"[KIS] 토큰 캐시 로드 실패: {e}")
        return None


def save_token(state_dir: str, access_token: str, expires_in: int = 86400):
    """토큰을 파일에 저장. 권한 0o600."""
    path = _token_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    expires_at = datetime.now() + timedelta(seconds=expires_in)
    data = {
        "access_token": access_token,
        "expires_at": expires_at.isoformat(),
        "issued_at": datetime.now().isoformat(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass   # Windows 등 chmod 미지원 환경
    log.info(f"[KIS] 토큰 저장: 만료={expires_at.strftime('%Y-%m-%d %H:%M')}")


def get_or_refresh_token(api_key: str, api_secret: str, base_url: str,
                          state_dir: str = "state") -> str:
    """캐시된 토큰이 유효하면 재사용, 아니면 신규 발급 후 저장."""
    cached = load_cached_token(state_dir)
    if cached:
        return cached

    url = f"{base_url}/oauth2/tokenP"
    body = {
        "grant_type": "client_credentials",
        "appkey": api_key,
        "appsecret": api_secret,
    }
    log.info(f"[KIS] 신규 토큰 발급 요청: {base_url}")
    resp = requests.post(url, json=body, timeout=10)
    data = resp.json()

    token = data.get("access_token", "")
    if not token:
        raise RuntimeError(f"[KIS] 토큰 발급 실패: {data}")

    expires_in = int(data.get("expires_in", 86400))
    save_token(state_dir, token, expires_in)
    log.info(f"[KIS] 신규 토큰 발급 완료 (len={len(token)})")
    return token


def invalidate_cache(state_dir: str):
    """API가 401을 반환하는 등 토큰이 거부당한 경우 캐시 삭제."""
    path = _token_path(state_dir)
    if path.exists():
        path.unlink()
        log.warning("[KIS] 토큰 캐시 무효화")


# ═══════════════════════════════════════════════════════════════════
# 토큰 에러 감지
# ═══════════════════════════════════════════════════════════════════

# KIS API 토큰 관련 msg_cd / rt_cd
# 참고: EGW00123 = 기간이 만료된 token, EGW00121 = 유효하지 않은 token
TOKEN_ERROR_CODES = {"EGW00121", "EGW00123", "EGW00124", "EGW00125"}
TOKEN_ERROR_KEYWORDS = ["만료", "유효하지 않은", "token", "TOKEN"]


def is_token_error(resp_status: int, resp_json: dict) -> bool:
    """
    응답이 토큰 만료/무효 에러인지 판단.

    Args:
        resp_status: HTTP 상태 코드
        resp_json: 파싱된 응답 JSON (실패 시 빈 dict 전달)

    Returns:
        토큰 재발급이 필요한 상황이면 True
    """
    if resp_status == 401:
        return True

    msg_cd = resp_json.get("msg_cd", "")
    if msg_cd in TOKEN_ERROR_CODES:
        return True

    # 일부 엔드포인트는 HTTP 200이지만 msg1에 만료 메시지만 담아 돌려줌
    msg1 = resp_json.get("msg1", "")
    if msg1 and any(kw in msg1 for kw in ["기간이 만료", "유효하지 않은 token"]):
        return True

    return False
