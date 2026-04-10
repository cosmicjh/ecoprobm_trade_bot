"""
에코프로비엠 (247540) v4 — Phase 2-2: 리스크 모니터
=====================================================
기존 리스크 관리(일/주/월 한도)에 더해:
  1. 갭다운 사전 경고 — 미국장 2차전지 관련주 모니터링
  2. 동적 포지션 사이징 — 리스크 레벨에 따라 투자비율 조정
  3. 비정상 변동성 감지 — ATR 급등 시 조기 경보

미국장 모니터링 대상:
  - RIVN (Rivian) — EV 대표주
  - QS (QuantumScape) — 전고체 배터리
  - ENVX (Ennovia) — 배터리 소재
  - LAC (Lithium Americas) — 리튬 채굴
  - ALB (Albemarle) — 리튬 화학
  - LIT (Global X Lithium ETF) — 2차전지 섹터 ETF

실행 시점: morning 모드(09:05 KST) 전에 호출
  → 전날 밤 미국장 결과를 보고 오늘 갭다운 리스크를 판단

한투 API:
  해외주식 현재가: /uapi/overseas-price/v1/quotations/price
  tr_id: HHDFS00000300 (실전)
  거래소: NASD (나스닥), NYSE (뉴욕)
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# 1. 모니터링 대상 종목
# ═══════════════════════════════════════════════════════════════════

US_BATTERY_STOCKS = [
    {"symbol": "RIVN", "exchange": "NAS", "name": "Rivian", "weight": 0.20},
    {"symbol": "QS",   "exchange": "NYS", "name": "QuantumScape", "weight": 0.15},
    {"symbol": "ENVX", "exchange": "NAS", "name": "Ennovia", "weight": 0.10},
    {"symbol": "ALB",  "exchange": "NYS", "name": "Albemarle", "weight": 0.25},
    {"symbol": "LIT",  "exchange": "NAS", "name": "Lithium ETF", "weight": 0.30},
]

# 리스크 레벨 임계치
RISK_THRESHOLDS = {
    "green":  {"us_change_min": -2.0, "label": "정상"},
    "yellow": {"us_change_min": -4.0, "label": "주의"},
    "orange": {"us_change_min": -7.0, "label": "경고"},
    "red":    {"us_change_min": float('-inf'), "label": "위험"},
}

# 리스크 레벨별 투자비율 조정 계수
RISK_MULTIPLIER = {
    "green":  1.0,    # 정상 → 100% 투자비율
    "yellow": 0.7,    # 주의 → 70%
    "orange": 0.3,    # 경고 → 30%
    "red":    0.0,    # 위험 → 진입 차단
}


@dataclass
class RiskAssessment:
    """리스크 평가 결과."""
    risk_level: str = "green"          # green/yellow/orange/red
    risk_label: str = "정상"
    invest_multiplier: float = 1.0     # 투자비율 조정 계수

    # 미국장 데이터
    us_weighted_change: float = 0.0    # 가중평균 등락률
    us_stocks: list = None             # 개별 종목 등락률
    us_worst: str = ""                 # 최악 종목

    # 갭다운 경고
    gapdown_probability: str = "LOW"   # LOW / MEDIUM / HIGH
    gapdown_warning: str = ""

    # 국내 지표
    atr_alert: bool = False            # ATR 급등 경보
    consecutive_losses: int = 0        # 연속 손실 횟수

    # 메타
    assessed_at: str = ""
    source: str = ""

    def __post_init__(self):
        if self.us_stocks is None:
            self.us_stocks = []


# ═══════════════════════════════════════════════════════════════════
# 2. 미국장 모니터링 (한투 해외주식 API)
# ═══════════════════════════════════════════════════════════════════

def fetch_us_stock_price(client, symbol: str, exchange: str) -> dict:
    """
    해외주식 현재가 시세 (HHDFS00000300).

    Returns:
        {"symbol": "RIVN", "current": 12.5, "prev_close": 13.0, "change_pct": -3.85}
    """
    path = "/uapi/overseas-price/v1/quotations/price"
    tr_id = "HHDFS00000300"

    params = {
        "AUTH": "",
        "EXCD": exchange,  # NASD, NYSE, AMEX
        "SYMB": symbol,
    }

    try:
        data = client.get(path, tr_id, params)
        output = data.get("output", {})

        current = _sf(output.get("last", 0))        # 현재가(종가)
        prev_close = _sf(output.get("base", 0))      # 전일종가
        change_pct = _sf(output.get("rate", 0))       # 등락률

        # rate가 없으면 직접 계산
        if change_pct == 0 and prev_close > 0 and current > 0:
            change_pct = round((current - prev_close) / prev_close * 100, 2)

        return {
            "symbol": symbol,
            "exchange": exchange,
            "current": current,
            "prev_close": prev_close,
            "change_pct": change_pct,
        }

    except Exception as e:
        log.warning(f"[US] {symbol} 조회 실패: {e}")
        return {"symbol": symbol, "exchange": exchange, "current": 0, "prev_close": 0, "change_pct": 0}


def monitor_us_battery_stocks(client) -> list:
    """
    미국 2차전지 관련주 전체 모니터링.
    Returns: 종목별 등락률 리스트
    """
    results = []

    for stock in US_BATTERY_STOCKS:
        data = fetch_us_stock_price(client, stock["symbol"], stock["exchange"])
        data["name"] = stock["name"]
        data["weight"] = stock["weight"]
        results.append(data)
        log.info(f"[US] {stock['symbol']} ({stock['name']}): {data['change_pct']:+.2f}%")

    return results


def calculate_weighted_change(us_results: list) -> float:
    """가중평균 등락률 계산."""
    total_weight = 0
    weighted_sum = 0

    for r in us_results:
        w = r.get("weight", 0)
        pct = r.get("change_pct", 0)
        if pct != 0:  # 데이터 있는 종목만
            weighted_sum += pct * w
            total_weight += w

    if total_weight > 0:
        return round(weighted_sum / total_weight, 2)
    return 0.0


# ═══════════════════════════════════════════════════════════════════
# 3. 리스크 레벨 판정
# ═══════════════════════════════════════════════════════════════════

def assess_risk(
    client,
    bot_state=None,
    latest_indicators: dict = None,
) -> RiskAssessment:
    """
    종합 리스크 평가.

    실행 순서:
      1. 미국장 2차전지 관련주 등락률 조회
      2. 가중평균 등락률로 리스크 레벨 판정
      3. 갭다운 확률 추정
      4. ATR 급등 체크
      5. 연속 손실 체크
      6. 최종 리스크 레벨 결정
    """
    assessment = RiskAssessment(assessed_at=datetime.now().isoformat())

    # ── 1) 미국장 모니터링 ──
    try:
        us_results = monitor_us_battery_stocks(client)
        assessment.us_stocks = us_results
        assessment.us_weighted_change = calculate_weighted_change(us_results)

        # 최악 종목
        if us_results:
            worst = min(us_results, key=lambda x: x.get("change_pct", 0))
            assessment.us_worst = f"{worst['symbol']} ({worst['change_pct']:+.1f}%)"

        log.info(f"[RISK] 미국 2차전지 가중평균: {assessment.us_weighted_change:+.2f}%")

    except Exception as e:
        log.warning(f"[RISK] 미국장 조회 실패: {e}")
        assessment.source = "us_data_failed"

    # ── 2) 리스크 레벨 판정 (미국장 기준) ──
    wc = assessment.us_weighted_change
    if wc >= RISK_THRESHOLDS["green"]["us_change_min"]:
        base_level = "green"
    elif wc >= RISK_THRESHOLDS["yellow"]["us_change_min"]:
        base_level = "yellow"
    elif wc >= RISK_THRESHOLDS["orange"]["us_change_min"]:
        base_level = "orange"
    else:
        base_level = "red"

    # ── 3) 갭다운 확률 추정 ──
    if wc <= -7:
        assessment.gapdown_probability = "HIGH"
        assessment.gapdown_warning = f"미국 2차전지 {wc:+.1f}% 급락 → 금일 갭다운 가능성 높음"
    elif wc <= -4:
        assessment.gapdown_probability = "MEDIUM"
        assessment.gapdown_warning = f"미국 2차전지 {wc:+.1f}% 하락 → 약세 출발 가능"
    else:
        assessment.gapdown_probability = "LOW"

    # ── 4) ATR 급등 체크 ──
    if latest_indicators:
        atr_ratio = latest_indicators.get("atr_ratio", 1.0)
        if atr_ratio and atr_ratio >= 1.8:
            assessment.atr_alert = True
            # ATR 급등이면 한 단계 상향
            if base_level == "green":
                base_level = "yellow"
            elif base_level == "yellow":
                base_level = "orange"
            log.info(f"[RISK] ATR 급등 감지 (ratio={atr_ratio:.2f}) → 리스크 상향")

    # ── 5) 연속 손실 체크 ──
    if bot_state:
        # bot_state에 최근 매매 이력이 있다면 연속 손절 횟수 확인
        recent_losses = _count_consecutive_losses(bot_state)
        assessment.consecutive_losses = recent_losses
        if recent_losses >= 3:
            # 3연속 손절이면 한 단계 상향
            if base_level == "green":
                base_level = "yellow"
            elif base_level == "yellow":
                base_level = "orange"
            log.info(f"[RISK] 연속 손절 {recent_losses}회 → 리스크 상향")

    # ── 6) 최종 결정 ──
    assessment.risk_level = base_level
    assessment.risk_label = RISK_THRESHOLDS.get(base_level, {}).get("label", "?")
    assessment.invest_multiplier = RISK_MULTIPLIER.get(base_level, 1.0)
    assessment.source = "full_assessment"

    log.info(f"[RISK] 최종: {assessment.risk_level.upper()} ({assessment.risk_label}) "
             f"| 투자배율: {assessment.invest_multiplier:.0%} "
             f"| 갭다운: {assessment.gapdown_probability}")

    return assessment


def _count_consecutive_losses(bot_state) -> int:
    """봇 상태에서 연속 손절 횟수 추정."""
    # bot_state에 trade_history가 없으면 daily_pnl로 근사
    if hasattr(bot_state, 'daily_pnl') and bot_state.daily_pnl < 0:
        return 1  # 보수적 추정
    return 0


# ═══════════════════════════════════════════════════════════════════
# 4. 리스크 조정 적용 함수
# ═══════════════════════════════════════════════════════════════════

def adjust_invest_ratio(
    base_ratio: float,
    risk: RiskAssessment,
) -> float:
    """
    리스크 평가에 따라 투자비율을 조정합니다.

    base_ratio: 전략 기본 투자비율 (예: 0.30)
    Returns: 조정된 투자비율
    """
    adjusted = base_ratio * risk.invest_multiplier
    adjusted = max(0.0, min(adjusted, 0.6))  # 0~60% 범위

    if adjusted != base_ratio:
        log.info(f"[RISK] 투자비율 조정: {base_ratio:.0%} → {adjusted:.0%} "
                 f"(리스크: {risk.risk_level})")

    return round(adjusted, 2)


def should_skip_entry(risk: RiskAssessment) -> tuple:
    """
    진입을 건너뛸지 판단.
    Returns: (skip: bool, reason: str)
    """
    if risk.risk_level == "red":
        return True, f"리스크 RED: {risk.gapdown_warning or '미국장 급락'}"

    if risk.gapdown_probability == "HIGH":
        return True, f"갭다운 HIGH: {risk.gapdown_warning}"

    if risk.consecutive_losses >= 4:
        return True, f"연속 손절 {risk.consecutive_losses}회 — 냉각 필요"

    return False, ""


# ═══════════════════════════════════════════════════════════════════
# 5. 리스크 리포트 (Telegram)
# ═══════════════════════════════════════════════════════════════════

def format_risk_telegram(risk: RiskAssessment) -> str:
    """리스크 평가 결과를 Telegram 메시지로 포맷."""
    emoji_map = {"green": "🟢", "yellow": "🟡", "orange": "🟠", "red": "🔴"}
    gap_emoji = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "🚨"}

    lines = [
        f"{emoji_map.get(risk.risk_level, '❓')} <b>리스크: {risk.risk_label}</b> ({risk.risk_level.upper()})",
        f"투자배율: {risk.invest_multiplier:.0%}",
        "",
        f"🇺🇸 미국 2차전지: {risk.us_weighted_change:+.2f}%",
    ]

    # 개별 종목
    if risk.us_stocks:
        for s in risk.us_stocks:
            pct = s.get("change_pct", 0)
            arrow = "📈" if pct > 0 else "📉" if pct < 0 else "➡️"
            lines.append(f"  {arrow} {s['symbol']}: {pct:+.1f}%")

    # 갭다운
    lines.append("")
    lines.append(f"{gap_emoji.get(risk.gapdown_probability, '❓')} 갭다운: {risk.gapdown_probability}")
    if risk.gapdown_warning:
        lines.append(f"  {risk.gapdown_warning}")

    # ATR
    if risk.atr_alert:
        lines.append("⚡ ATR 급등 감지")

    # 연속 손실
    if risk.consecutive_losses > 0:
        lines.append(f"📉 연속 손절: {risk.consecutive_losses}회")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# 6. 리스크 이력 저장
# ═══════════════════════════════════════════════════════════════════

def save_risk_history(risk: RiskAssessment, state_dir: str = "state"):
    """리스크 평가 이력을 날짜별로 누적."""
    path = Path(state_dir) / "risk_history_247540.json"

    history = {}
    if path.exists():
        try:
            with open(path, "r") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            pass

    today = datetime.now().strftime("%Y-%m-%d")
    history[today] = {
        "risk_level": risk.risk_level,
        "us_weighted_change": risk.us_weighted_change,
        "gapdown_probability": risk.gapdown_probability,
        "invest_multiplier": risk.invest_multiplier,
        "atr_alert": risk.atr_alert,
        "assessed_at": risk.assessed_at,
    }

    # 최근 90일만 보관
    cutoff = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    history = {k: v for k, v in history.items() if k >= cutoff}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════
# 유틸
# ═══════════════════════════════════════════════════════════════════

def _sf(val) -> float:
    if val is None or val == "":
        return 0.0
    try:
        return float(str(val).replace(",", ""))
    except (ValueError, TypeError):
        return 0.0


# ═══════════════════════════════════════════════════════════════════
# CLI (단독 실행 테스트)
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # trading_bot의 KISClient 재사용
    from trading_bot import KISClient

    client = KISClient()
    risk = assess_risk(client)

    print("\n" + format_risk_telegram(risk))
    print(f"\n투자비율 조정 예시: 기본 30% → {adjust_invest_ratio(0.30, risk):.0%}")

    skip, reason = should_skip_entry(risk)
    if skip:
        print(f"\n🚫 진입 차단: {reason}")
    else:
        print(f"\n✅ 진입 허용")

    save_risk_history(risk)
    print(f"\n리스크 이력 저장 완료")
