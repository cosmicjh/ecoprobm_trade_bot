"""
에코프로비엠 (247540) v4 — Phase 3-2: 주간 AI 재학습 & 결산
==============================================================
매주 일요일 밤(22:00 KST)에 cron-job.org → workflow_dispatch로 호출.

수행 작업:
  1. ai_layer.train_all_models() 호출
       - HMM 레짐 분류기 재학습 (BIC 기반 최적 state 수 자동 선택)
       - Isolation Forest 수급 이상 감지기 재학습
       - 새 모델을 state/hmm_model_247540.pkl, if_model_247540.pkl로 저장
  2. 학습 결과를 텔레그램으로 발송 (state 분포, 현재 레짐 예측 등)
  3. 한 주 매매 결산 (format_weekly_report) 동시 발송
  4. 학습 메타데이터를 state/retrain_history_247540.json에 누적

설계 원칙:
  - 학습 실패해도 봇은 기존 모델로 계속 동작 (graceful fallback)
  - 학습 결과는 GitHub Actions에서 state/ 디렉토리로 commit → 다음 morning 실행 시 자동 반영
  - 학습 데이터는 data_collector가 수집해 둔 ohlcv_247540.json / supply_data_247540.json 사용
"""

import os
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path

import requests

from ai_layer import train_all_models
from reporter import format_weekly_report, load_trade_history

# trading_bot의 BotState를 그대로 재사용 (주간 리포트가 state 객체를 요구)
from trading_bot import load_bot_state, save_bot_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

TICKER = "247540"
TICKER_NAME = "에코프로비엠"


# ═══════════════════════════════════════════════════════════════════
# Telegram
# ═══════════════════════════════════════════════════════════════════

def send_telegram(message: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        log.warning("[TG] 토큰/chat_id 없음, 발송 생략")
        return
    try:
        # Telegram 메시지 길이 제한 4096자 → 안전하게 분할
        chunks = [message[i:i + 3800] for i in range(0, len(message), 3800)]
        for chunk in chunks:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": chunk, "parse_mode": "HTML"},
                timeout=10,
            )
    except Exception as e:
        log.error(f"[TG] 발송 실패: {e}")


# ═══════════════════════════════════════════════════════════════════
# 결과 포매팅
# ═══════════════════════════════════════════════════════════════════

def format_retrain_telegram(result: dict) -> str:
    """train_all_models 결과를 텔레그램용 메시지로 변환."""
    lines = [
        f"🧠 <b>{TICKER_NAME} AI 주간 재학습</b>",
        f"실행: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    # ── HMM 결과 ──
    hmm = result.get("hmm", {})
    if "error" in hmm:
        lines.append(f"❌ HMM: {hmm['error']}")
    else:
        n_states = hmm.get("n_states", "?")
        ll = hmm.get("log_likelihood", 0)
        lines.append(f"<b>HMM</b> ({n_states} states, LL={ll:.1f})")

        dist = hmm.get("state_distribution", {})
        for label, info in dist.items():
            lines.append(
                f"  • {label}: {info['pct']}% "
                f"(ret={info['mean_daily_return']:+.2f}%, "
                f"vol={info['mean_atr_ratio']:.2f})"
            )

    # ── 현재 레짐 ──
    current = result.get("hmm_current", {})
    if current and "regime" in current:
        lines.append("")
        lines.append(
            f"📍 현재 HMM 레짐: <b>{current.get('regime', '?')}</b> "
            f"({current.get('confidence', 0):.0%})"
        )

    # ── Isolation Forest ──
    if_result = result.get("isolation_forest", {})
    lines.append("")
    if "error" in if_result:
        lines.append(f"❌ Isolation Forest: {if_result['error']}")
    else:
        n_samples = if_result.get("n_samples", "?")
        contamination = if_result.get("contamination", "?")
        lines.append(f"<b>Isolation Forest</b>")
        lines.append(f"  • 학습 샘플: {n_samples}")
        lines.append(f"  • contamination: {contamination}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# 재학습 이력 누적
# ═══════════════════════════════════════════════════════════════════

def append_retrain_history(state_dir: str, result: dict, success: bool, error: str = ""):
    path = Path(state_dir) / "retrain_history_247540.json"
    history = []
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    entry = {
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "error": error,
    }

    hmm = result.get("hmm", {}) if result else {}
    if "n_states" in hmm:
        entry["hmm_n_states"] = hmm["n_states"]
        entry["hmm_log_likelihood"] = hmm.get("log_likelihood")

    current = result.get("hmm_current", {}) if result else {}
    if current.get("regime"):
        entry["hmm_current_regime"] = current["regime"]
        entry["hmm_current_confidence"] = current.get("confidence", 0)

    if_res = result.get("isolation_forest", {}) if result else {}
    if "n_samples" in if_res:
        entry["if_n_samples"] = if_res["n_samples"]

    history.append(entry)
    # 최근 52주(1년) 만 보관
    history = history[-52:]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"[RETRAIN] 이력 저장: {path}")


# ═══════════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════════

def run_weekly_retrain(state_dir: str = "state"):
    log.info("=" * 60)
    log.info(f"[RETRAIN] {TICKER_NAME} v4 주간 AI 재학습 시작")
    log.info("=" * 60)

    sd = Path(state_dir)
    sd.mkdir(parents=True, exist_ok=True)

    ohlcv_path = sd / f"ohlcv_{TICKER}.json"
    supply_path = sd / f"supply_data_{TICKER}.json"

    if not ohlcv_path.exists():
        msg = f"❌ {TICKER_NAME} 재학습 실패: OHLCV 파일 없음 ({ohlcv_path})"
        log.error(msg)
        send_telegram(msg)
        append_retrain_history(state_dir, {}, success=False, error="ohlcv missing")
        return

    # ── 1. AI 모델 재학습 ──
    result = {}
    try:
        result = train_all_models(
            ohlcv_path=str(ohlcv_path),
            supply_path=str(supply_path),
            output_dir=str(sd),
        )
        retrain_msg = format_retrain_telegram(result)
        send_telegram(retrain_msg)
        append_retrain_history(state_dir, result, success=True)
        log.info("[RETRAIN] AI 재학습 완료")
    except Exception as e:
        tb = traceback.format_exc()
        log.error(f"[RETRAIN] AI 재학습 실패: {e}\n{tb}")
        send_telegram(
            f"❌ <b>{TICKER_NAME} AI 재학습 실패</b>\n"
            f"<code>{str(e)[:500]}</code>\n\n"
            f"기존 모델로 계속 동작합니다."
        )
        append_retrain_history(state_dir, result, success=False, error=str(e)[:300])

    # ── 2. 주간 결산 리포트 ──
    try:
        state = load_bot_state()
        weekly_msg = format_weekly_report(state, str(sd))
        send_telegram(weekly_msg)

        # 주간 PnL 누적값 리셋 (다음 주 새로 카운트)
        state.weekly_pnl = 0.0
        state.last_week_reset = datetime.now().strftime("%Y-%m-%d")
        save_bot_state(state)
        log.info("[RETRAIN] 주간 리포트 발송 + weekly_pnl 리셋 완료")
    except Exception as e:
        log.error(f"[RETRAIN] 주간 리포트 실패: {e}")
        send_telegram(f"⚠️ {TICKER_NAME} 주간 리포트 생성 실패: {str(e)[:300]}")

    log.info("=" * 60)
    log.info("[RETRAIN] 완료")
    log.info("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=f"{TICKER_NAME} v4 주간 AI 재학습")
    parser.add_argument("--state-dir", type=str, default="state")
    args = parser.parse_args()

    run_weekly_retrain(state_dir=args.state_dir)
