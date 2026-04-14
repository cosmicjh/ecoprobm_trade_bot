"""
에코프로비엠 (247540) v4 — Phase 4-5: 실행 로그 수집기
==========================================================
trading_bot.py / data_collector.py 의 매 실행마다 한 줄 JSONL 로 기록.
GitHub Pages 대시보드의 '실행 캘린더'와 '에러 추적'에 사용된다.

JSONL을 사용하는 이유:
  - 매 실행마다 append-only (concurrent write 충돌 최소화)
  - 줄 단위 파싱이라 일부가 깨져도 나머지는 살아있음
  - tail -n으로 최근 실행만 빠르게 확인 가능
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

TICKER = "247540"
LOG_FILE = f"run_log_{TICKER}.jsonl"
MAX_LINES = 5000   # 약 5년치


def log_run(state_dir: str,
            mode: str,
            status: str,
            duration_sec: float = 0,
            signal: Optional[str] = None,
            regime: Optional[str] = None,
            error: Optional[str] = None,
            extra: Optional[dict] = None):
    """
    실행 종료 시 한 줄 기록.

    Args:
        mode: morning / closing / evening / retrain / optimize / collect
        status: ok / error / blocked / no_action
        duration_sec: 실행 소요 시간
        signal: 마지막 시그널 (있으면)
        regime: 마지막 레짐 (있으면)
        error: 에러 메시지 (있으면)
        extra: 추가 메타데이터 dict
    """
    path = Path(state_dir) / LOG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "mode": mode,
        "status": status,
        "duration_sec": round(duration_sec, 2),
    }
    if signal is not None:
        entry["signal"] = signal
    if regime is not None:
        entry["regime"] = regime
    if error is not None:
        entry["error"] = str(error)[:300]
    if extra:
        entry["extra"] = extra

    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # 너무 길어지면 최근 MAX_LINES만 유지 (앞부분 잘라냄)
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) > MAX_LINES * 1.1:
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(lines[-MAX_LINES:])
    except Exception as e:
        log.error(f"[RUN_LOG] 기록 실패: {e}")


def read_recent_runs(state_dir: str, n: int = 100) -> list:
    """최근 n건의 실행 이력 반환."""
    path = Path(state_dir) / LOG_FILE
    if not path.exists():
        return []
    runs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines()[-n:]:
            try:
                runs.append(json.loads(line))
            except Exception:
                continue
    return runs
