"""
에코프로비엠 (247540) v4 — Phase 4-3: 예측 정확도 트래킹
==========================================================
매일 morning에 봇이 내린 예측(레짐, HMM, 시그널, 수급 이상)을
state/predictions_247540.json 에 누적하고, 다음 영업일에 실제
주가 변화로 채점합니다.

집계 지표:
  - 레짐 분류 정확도 (rule + HMM 앙상블)
  - 매수 시그널 적중률 (BUY_* → 다음날 양수익 비율)
  - 수급 이상 감지 적중률 (bullish → 양수익, bearish → 음수익)

라벨링 규칙 (다음 영업일 종가 수익률 r 기준):
  TREND_UP    correct if r > +0.5%
  TREND_DOWN  correct if r < -0.5%
  RANGE_BOUND correct if |r| < 1.5%
  HIGH_VOL    correct if |r| > 2.0%
  NEUTRAL     평가 제외

  BUY_*       correct if r > 0
  HOLD/NO_ENTRY/BLOCKED  평가 제외 (기회손실 측정 어려움)

  supply bullish  correct if r > 0
  supply bearish  correct if r < 0
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TICKER = "247540"
PRED_FILE = f"predictions_{TICKER}.json"


# ═══════════════════════════════════════════════════════════════════
# 1. 기록 (morning에서 호출)
# ═══════════════════════════════════════════════════════════════════

def record_prediction(state_dir: str, date: str, regime: str,
                       signal: str,
                       hmm_regime: str = "UNKNOWN", hmm_confidence: float = 0,
                       supply_anomaly: bool = False, supply_direction: str = "neutral",
                       price_at_prediction: float = 0):
    """봇이 morning에서 내린 예측을 기록."""
    path = Path(state_dir) / PRED_FILE
    records = []
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = []

    # 같은 날짜 중복 방지
    records = [r for r in records if r.get("date") != date]

    records.append({
        "date": date,
        "regime": regime,
        "signal": signal,
        "hmm_regime": hmm_regime,
        "hmm_confidence": float(hmm_confidence),
        "supply_anomaly": bool(supply_anomaly),
        "supply_direction": supply_direction,
        "price_at_prediction": float(price_at_prediction),
        "evaluated": False,
        "next_return_pct": None,
        "regime_correct": None,
        "hmm_correct": None,
        "signal_correct": None,
        "supply_correct": None,
    })

    # 최근 365일치만 보관
    records = records[-365:]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    log.info(f"[ACC] 예측 기록: {date} regime={regime} signal={signal}")


# ═══════════════════════════════════════════════════════════════════
# 2. 라벨링 (closing/evening에서 호출)
# ═══════════════════════════════════════════════════════════════════

def _label_regime(regime: str, ret_pct: float) -> bool:
    """수익률(%)을 기준으로 레짐 예측 정오 판정."""
    if regime == "TREND_UP":
        return ret_pct > 0.5
    if regime == "TREND_DOWN":
        return ret_pct < -0.5
    if regime == "RANGE_BOUND":
        return abs(ret_pct) < 1.5
    if regime == "HIGH_VOLATILITY":
        return abs(ret_pct) > 2.0
    return None  # NEUTRAL/UNKNOWN 평가 제외


def _label_signal(signal: str, ret_pct: float) -> bool:
    if signal.startswith("BUY"):
        return ret_pct > 0
    return None


def _label_supply(direction: str, ret_pct: float) -> bool:
    if direction == "bullish":
        return ret_pct > 0
    if direction == "bearish":
        return ret_pct < 0
    return None


def evaluate_pending_predictions(state_dir: str) -> int:
    """
    아직 평가되지 않은 예측들에 대해, OHLCV 파일을 참조하여
    예측일 다음 영업일 종가 수익률로 라벨링.

    Returns: 새로 라벨링한 건수
    """
    sd = Path(state_dir)
    pred_path = sd / PRED_FILE
    ohlcv_path = sd / f"ohlcv_{TICKER}.json"

    if not pred_path.exists() or not ohlcv_path.exists():
        return 0

    with open(pred_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    with open(ohlcv_path, "r", encoding="utf-8") as f:
        ohlcv = json.load(f)

    # 일자별 종가 매핑
    closes = {d: v.get("Close", 0) for d, v in ohlcv.items()}
    sorted_dates = sorted(closes.keys())

    n_labeled = 0
    for r in records:
        if r.get("evaluated"):
            continue

        pred_date = r["date"]
        if pred_date not in closes:
            continue

        # 다음 영업일 찾기
        try:
            idx = sorted_dates.index(pred_date)
        except ValueError:
            continue
        if idx + 1 >= len(sorted_dates):
            continue   # 아직 다음 일봉 미수집

        today_close = closes[pred_date]
        next_close = closes[sorted_dates[idx + 1]]
        if today_close <= 0:
            continue

        ret_pct = (next_close - today_close) / today_close * 100

        r["next_return_pct"] = round(ret_pct, 3)
        r["regime_correct"] = _label_regime(r.get("regime", ""), ret_pct)
        r["hmm_correct"] = _label_regime(r.get("hmm_regime", ""), ret_pct)
        r["signal_correct"] = _label_signal(r.get("signal", ""), ret_pct)
        if r.get("supply_anomaly"):
            r["supply_correct"] = _label_supply(r.get("supply_direction", ""), ret_pct)
        r["evaluated"] = True
        n_labeled += 1

    if n_labeled > 0:
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        log.info(f"[ACC] 신규 라벨링: {n_labeled}건")

    return n_labeled


# ═══════════════════════════════════════════════════════════════════
# 3. 집계
# ═══════════════════════════════════════════════════════════════════

def compute_accuracy_stats(state_dir: str, days: int = 30) -> dict:
    """최근 N일 예측에 대한 정확도 통계."""
    path = Path(state_dir) / PRED_FILE
    if not path.exists():
        return {"n": 0}

    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    recent = [r for r in records if r.get("date", "") >= cutoff and r.get("evaluated")]

    if not recent:
        return {"n": 0}

    def acc(field):
        vals = [r[field] for r in recent if r.get(field) is not None]
        if not vals:
            return None
        return sum(1 for v in vals if v) / len(vals), len(vals)

    rule = acc("regime_correct")
    hmm = acc("hmm_correct")
    sig = acc("signal_correct")
    sup = acc("supply_correct")

    # 레짐별 분포
    regime_counts = {}
    for r in recent:
        rg = r.get("regime", "UNKNOWN")
        regime_counts[rg] = regime_counts.get(rg, 0) + 1

    return {
        "n": len(recent),
        "days": days,
        "regime_rule": rule,
        "regime_hmm": hmm,
        "signal": sig,
        "supply": sup,
        "regime_distribution": regime_counts,
    }


def format_accuracy_telegram(stats: dict) -> str:
    """주간/월간 리포트에 첨부할 적중률 메시지."""
    if stats.get("n", 0) == 0:
        return "🎯 <b>예측 정확도</b>\n평가 가능한 데이터 없음"

    lines = [
        f"🎯 <b>예측 정확도 (최근 {stats['days']}일, n={stats['n']})</b>",
    ]

    def fmt(label, t):
        if t is None:
            return f"  • {label}: —"
        rate, n = t
        return f"  • {label}: {rate*100:.1f}% ({n}건)"

    lines.append(fmt("룰 레짐", stats.get("regime_rule")))
    lines.append(fmt("HMM 레짐", stats.get("regime_hmm")))
    lines.append(fmt("매수 시그널", stats.get("signal")))
    lines.append(fmt("수급 이상", stats.get("supply")))

    dist = stats.get("regime_distribution", {})
    if dist:
        lines.append("")
        lines.append("<b>레짐 분포:</b>")
        for rg, c in sorted(dist.items(), key=lambda x: -x[1]):
            lines.append(f"  • {rg}: {c}회")

    return "\n".join(lines)
