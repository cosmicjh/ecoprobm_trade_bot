"""
에코프로비엠 (247540) v4 — Phase 2-4: 리포트 & 성과 집계
==========================================================
기능:
  1. 매매 이력 영구 기록 (trade_history_247540.json)
  2. 일일 종합 리포트 (closing 모드)
  3. 주간 성과 리포트 (금요일 closing)
  4. 월간 성과 리포트 (월말 closing)
  5. 백테스트 기대값 vs 실전 괴리 분석

리포트 지표:
  - 총수익률, 승률, Profit Factor
  - 평균 익절/손절, 최대 연속 손실
  - 일/주/월 PnL
  - 매매 횟수, 체결률
  - 전략 시그널 분포 (어떤 레짐에서 진입했는지)
  - 백테스트 대비 실전 괴리도
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

TICKER = "247540"
TICKER_NAME = "에코프로비엠"


# ═══════════════════════════════════════════════════════════════════
# 1. 매매 이력 데이터 구조
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    """단일 매매 이력."""
    date: str = ""              # YYYY-MM-DD
    timestamp: str = ""         # ISO 8601
    side: str = ""              # buy / sell
    reason: str = ""            # ENTRY / SL / TP1 / TRAIL / TP_RB / etc.
    price: int = 0
    qty: int = 0
    pnl: int = 0                # 매도 시 손익 (매수는 0)
    pnl_pct: float = 0.0        # 매도 시 수익률 (%)
    regime: str = ""            # 매매 시점 레짐
    signal: str = ""            # 매매 시점 시그널
    entry_price: float = 0.0    # 매도 시 진입가 (참조용)
    holding_days: int = 0       # 보유 일수


# ═══════════════════════════════════════════════════════════════════
# 2. 매매 이력 저장 / 로드
# ═══════════════════════════════════════════════════════════════════

def log_trade(
    state_dir: str,
    side: str,
    reason: str,
    price: int,
    qty: int,
    pnl: int = 0,
    regime: str = "",
    signal: str = "",
    entry_price: float = 0.0,
    entry_date: str = "",
):
    """
    매매 이력을 trade_history_247540.json에 추가합니다.
    trading_bot._execute_buy/_execute_sell에서 호출.
    """
    path = Path(state_dir) / "trade_history_247540.json"

    history = []
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except (json.JSONDecodeError, IOError):
            history = []

    today = datetime.now().strftime("%Y-%m-%d")
    pnl_pct = 0.0
    holding_days = 0

    if side == "sell" and entry_price > 0:
        pnl_pct = round((price - entry_price) / entry_price * 100, 2)

    if entry_date:
        try:
            ed = datetime.strptime(entry_date, "%Y-%m-%d")
            holding_days = (datetime.now() - ed).days
        except ValueError:
            pass

    record = TradeRecord(
        date=today,
        timestamp=datetime.now().isoformat(),
        side=side,
        reason=reason,
        price=int(price),
        qty=int(qty),
        pnl=int(pnl),
        pnl_pct=pnl_pct,
        regime=regime,
        signal=signal,
        entry_price=float(entry_price),
        holding_days=holding_days,
    )

    history.append(asdict(record))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2, default=str)

    log.info(f"[TRADE_LOG] {side}/{reason} {qty}주 @ {price:,} 기록 (총 {len(history)}건)")


def load_trade_history(state_dir: str) -> list:
    path = Path(state_dir) / "trade_history_247540.json"
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


# ═══════════════════════════════════════════════════════════════════
# 3. 성과 통계 계산
# ═══════════════════════════════════════════════════════════════════

def compute_period_stats(history: list, start_date: str, end_date: str = None) -> dict:
    """
    지정 기간의 매매 통계를 계산합니다.

    Returns:
        {
            "total_trades", "buy_count", "sell_count",
            "win_count", "loss_count", "win_rate",
            "total_pnl", "avg_profit", "avg_loss", "max_profit", "max_loss",
            "profit_factor", "avg_holding_days",
            "regime_distribution", "exit_reason_distribution",
            "consecutive_wins", "consecutive_losses",
        }
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    filtered = [t for t in history if start_date <= t.get("date", "") <= end_date]

    if not filtered:
        return _empty_stats(start_date, end_date)

    buys = [t for t in filtered if t["side"] == "buy"]
    sells = [t for t in filtered if t["side"] == "sell"]

    profit_trades = [t for t in sells if t.get("pnl", 0) > 0]
    loss_trades = [t for t in sells if t.get("pnl", 0) < 0]
    breakeven = [t for t in sells if t.get("pnl", 0) == 0]

    total_pnl = sum(t.get("pnl", 0) for t in sells)

    avg_profit = np.mean([t["pnl"] for t in profit_trades]) if profit_trades else 0
    avg_loss = np.mean([t["pnl"] for t in loss_trades]) if loss_trades else 0
    max_profit = max([t["pnl"] for t in profit_trades]) if profit_trades else 0
    max_loss = min([t["pnl"] for t in loss_trades]) if loss_trades else 0

    gross_profit = sum(t["pnl"] for t in profit_trades) if profit_trades else 0
    gross_loss = abs(sum(t["pnl"] for t in loss_trades)) if loss_trades else 1
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0

    win_rate = round(len(profit_trades) / len(sells) * 100, 1) if sells else 0

    avg_holding = np.mean([t.get("holding_days", 0) for t in sells]) if sells else 0

    # 진입 레짐 분포
    regime_dist = {}
    for t in buys:
        r = t.get("regime", "UNKNOWN")
        regime_dist[r] = regime_dist.get(r, 0) + 1

    # 청산 사유 분포
    exit_dist = {}
    for t in sells:
        r = t.get("reason", "UNKNOWN")
        exit_dist[r] = exit_dist.get(r, 0) + 1

    # 연속 승/패 계산
    cons_win, cons_loss, max_cons_win, max_cons_loss = 0, 0, 0, 0
    cur_win, cur_loss = 0, 0
    for t in sorted(sells, key=lambda x: x.get("timestamp", "")):
        pnl = t.get("pnl", 0)
        if pnl > 0:
            cur_win += 1
            cur_loss = 0
            max_cons_win = max(max_cons_win, cur_win)
        elif pnl < 0:
            cur_loss += 1
            cur_win = 0
            max_cons_loss = max(max_cons_loss, cur_loss)
    cons_win, cons_loss = cur_win, cur_loss

    return {
        "period": f"{start_date} ~ {end_date}",
        "total_trades": len(filtered),
        "buy_count": len(buys),
        "sell_count": len(sells),
        "win_count": len(profit_trades),
        "loss_count": len(loss_trades),
        "breakeven_count": len(breakeven),
        "win_rate": win_rate,
        "total_pnl": int(total_pnl),
        "avg_profit": int(avg_profit),
        "avg_loss": int(avg_loss),
        "max_profit": int(max_profit),
        "max_loss": int(max_loss),
        "profit_factor": profit_factor,
        "avg_holding_days": round(float(avg_holding), 1),
        "regime_distribution": regime_dist,
        "exit_reason_distribution": exit_dist,
        "current_streak_wins": cons_win,
        "current_streak_losses": cons_loss,
        "max_consecutive_wins": max_cons_win,
        "max_consecutive_losses": max_cons_loss,
    }


def _empty_stats(start_date: str, end_date: str) -> dict:
    return {
        "period": f"{start_date} ~ {end_date}",
        "total_trades": 0,
        "buy_count": 0, "sell_count": 0,
        "win_count": 0, "loss_count": 0, "breakeven_count": 0,
        "win_rate": 0, "total_pnl": 0,
        "avg_profit": 0, "avg_loss": 0, "max_profit": 0, "max_loss": 0,
        "profit_factor": 0, "avg_holding_days": 0,
        "regime_distribution": {}, "exit_reason_distribution": {},
        "current_streak_wins": 0, "current_streak_losses": 0,
        "max_consecutive_wins": 0, "max_consecutive_losses": 0,
    }


# ═══════════════════════════════════════════════════════════════════
# 4. 백테스트 vs 실전 괴리 분석
# ═══════════════════════════════════════════════════════════════════

def compare_to_backtest(live_stats: dict, state_dir: str) -> dict:
    """
    백테스트 기대값과 실전 결과를 비교합니다.
    optimized_params_247540.json의 backtest_result 사용.
    """
    params_path = Path(state_dir) / "optimized_params_247540.json"
    if not params_path.exists():
        return {"available": False}

    try:
        with open(params_path, "r") as f:
            opt = json.load(f)
        bt = opt.get("backtest_result", {})
    except (json.JSONDecodeError, IOError):
        return {"available": False}

    if not bt:
        return {"available": False}

    bt_win_rate = bt.get("win_rate", 0)
    bt_pf = bt.get("profit_factor", 0)
    live_win_rate = live_stats.get("win_rate", 0)
    live_pf = live_stats.get("profit_factor", 0)

    # 괴리 계산
    win_rate_diff = round(live_win_rate - bt_win_rate, 1)
    pf_diff = round(live_pf - bt_pf, 2) if bt_pf > 0 else 0

    # 평가
    status = "정상"
    if live_stats.get("sell_count", 0) >= 5:
        if win_rate_diff < -15 or pf_diff < -0.5:
            status = "괴리 ⚠️"
        elif win_rate_diff < -25 or pf_diff < -1.0:
            status = "심각 ❌"

    return {
        "available": True,
        "backtest_win_rate": bt_win_rate,
        "backtest_pf": bt_pf,
        "live_win_rate": live_win_rate,
        "live_pf": live_pf,
        "win_rate_diff": win_rate_diff,
        "pf_diff": pf_diff,
        "status": status,
    }


# ═══════════════════════════════════════════════════════════════════
# 5. Telegram 리포트 포맷
# ═══════════════════════════════════════════════════════════════════

def format_daily_report(state, current_price: int, state_dir: str) -> str:
    """
    일일 종합 리포트 (closing 모드).
    """
    history = load_trade_history(state_dir)
    today = datetime.now().strftime("%Y-%m-%d")
    today_trades = [t for t in history if t.get("date") == today]

    equity = state.cash + state.position_qty * current_price
    pnl_total = (equity - state.initial_capital) / state.initial_capital * 100

    lines = [
        f"📊 <b>{TICKER_NAME} 일일 리포트</b>",
        f"📅 {today}",
        "",
        f"💰 평가금액: {equity:,.0f}원 ({pnl_total:+.2f}%)",
        f"💵 현금: {state.cash:,.0f}원",
        f"📦 포지션: {state.position_qty}주",
    ]

    if state.position_qty > 0 and state.entry_price > 0:
        cur_pnl = (current_price - state.entry_price) * state.position_qty
        cur_pct = (current_price - state.entry_price) / state.entry_price * 100
        lines.append(f"  진입가: {state.entry_price:,.0f}원")
        lines.append(f"  미실현: {cur_pnl:+,.0f}원 ({cur_pct:+.2f}%)")

    lines.append("")
    lines.append(f"📈 일일 PnL: {state.daily_pnl:+,.0f}원")
    lines.append(f"📈 주간 PnL: {state.weekly_pnl:+,.0f}원")
    lines.append(f"📈 월간 PnL: {state.monthly_pnl:+,.0f}원")

    if today_trades:
        lines.append("")
        lines.append(f"🔄 금일 매매: {len(today_trades)}건")
        for t in today_trades:
            side_icon = "🟢" if t["side"] == "buy" else "🔴"
            pnl_str = f" ({t['pnl']:+,}원)" if t["side"] == "sell" else ""
            lines.append(f"  {side_icon} {t['reason']} {t['qty']}주 @ {t['price']:,}{pnl_str}")

    if state.last_regime:
        lines.append("")
        lines.append(f"📊 레짐: {state.last_regime}")

    return "\n".join(lines)


def format_weekly_report(state, state_dir: str,
                          week_start: str = None,
                          week_end: str = None) -> str:
    """
    주간 성과 리포트 (금요일 closing).
    week_start/week_end 미지정 시 이번 주(월~오늘) 사용.
    """
    if week_start is None or week_end is None:
        today = datetime.now()
        week_start = (today - timedelta(days=today.weekday())).strftime("%Y-%m-%d")
        week_end = today.strftime("%Y-%m-%d")

    history = load_trade_history(state_dir)
    stats = compute_period_stats(history, week_start, week_end)

    lines = [
        f"📊 <b>{TICKER_NAME} 주간 성과</b>",
        f"📅 {week_start} ~ {week_end}",
        "",
        f"💰 주간 PnL: <b>{stats['total_pnl']:+,}원</b>",
        f"🔄 매매: 매수 {stats['buy_count']} / 매도 {stats['sell_count']}",
    ]

    if stats["sell_count"] > 0:
        lines.append("")
        lines.append(f"🎯 승률: {stats['win_rate']}% ({stats['win_count']}/{stats['sell_count']})")
        lines.append(f"📊 Profit Factor: {stats['profit_factor']}")
        lines.append(f"📈 평균 익절: {stats['avg_profit']:+,}원")
        lines.append(f"📉 평균 손절: {stats['avg_loss']:+,}원")
        lines.append(f"🏆 최대 익절: {stats['max_profit']:+,}원")
        lines.append(f"💢 최대 손절: {stats['max_loss']:+,}원")
        lines.append(f"⏱ 평균 보유: {stats['avg_holding_days']}일")

        if stats["max_consecutive_losses"] >= 2:
            lines.append(f"⚠️ 최대 연속 손절: {stats['max_consecutive_losses']}회")

    # 진입 레짐 분포
    if stats["regime_distribution"]:
        lines.append("")
        lines.append("📊 진입 레짐:")
        for regime, count in sorted(stats["regime_distribution"].items(),
                                     key=lambda x: -x[1]):
            lines.append(f"  · {regime}: {count}회")

    # 청산 사유 분포
    if stats["exit_reason_distribution"]:
        lines.append("")
        lines.append("🚪 청산 사유:")
        for reason, count in sorted(stats["exit_reason_distribution"].items(),
                                     key=lambda x: -x[1]):
            lines.append(f"  · {reason}: {count}회")

    # 백테스트 비교
    cmp = compare_to_backtest(stats, state_dir)
    if cmp["available"] and stats["sell_count"] >= 3:
        lines.append("")
        lines.append("🎯 백테스트 vs 실전:")
        lines.append(f"  승률: {cmp['backtest_win_rate']}% → {cmp['live_win_rate']}% ({cmp['win_rate_diff']:+}%)")
        lines.append(f"  PF: {cmp['backtest_pf']} → {cmp['live_pf']} ({cmp['pf_diff']:+})")
        lines.append(f"  상태: {cmp['status']}")

    # 자본 변화
    lines.append("")
    lines.append(f"💰 자본: {state.initial_capital:,.0f} → {state.cash + state.position_qty * state.entry_price:,.0f}원")

    return "\n".join(lines)


def format_monthly_report(state, state_dir: str,
                           month_start: str = None,
                           month_end: str = None) -> str:
    """
    월간 성과 리포트 (월말 closing).
    month_start/month_end 미지정 시 이번 달(1일~오늘) 사용.
    """
    if month_start is None or month_end is None:
        today = datetime.now()
        month_start = today.replace(day=1).strftime("%Y-%m-%d")
        month_end = today.strftime("%Y-%m-%d")
        month_str = today.strftime("%Y년 %m월")
    else:
        month_str = month_start[:7]

    history = load_trade_history(state_dir)
    stats = compute_period_stats(history, month_start, month_end)

    lines = [
        f"📊 <b>{TICKER_NAME} 월간 성과 — {month_str}</b>",
        f"📅 {month_start} ~ {month_end}",
        "",
        f"💰 월간 PnL: <b>{stats['total_pnl']:+,}원</b>",
        f"📊 수익률: {stats['total_pnl'] / state.initial_capital * 100:+.2f}%",
        "",
        f"🔄 총 매매: {stats['total_trades']}건 (매수 {stats['buy_count']} / 매도 {stats['sell_count']})",
    ]

    if stats["sell_count"] > 0:
        lines.append("")
        lines.append(f"🎯 승률: {stats['win_rate']}% ({stats['win_count']}/{stats['sell_count']})")
        lines.append(f"📊 Profit Factor: {stats['profit_factor']}")
        lines.append(f"⏱ 평균 보유: {stats['avg_holding_days']}일")
        lines.append(f"⚠️ 최대 연속 손절: {stats['max_consecutive_losses']}회")
        lines.append(f"🏆 최대 연속 익절: {stats['max_consecutive_wins']}회")

    if stats["regime_distribution"]:
        lines.append("")
        lines.append("📊 레짐별 진입:")
        total_buys = sum(stats["regime_distribution"].values())
        for regime, count in sorted(stats["regime_distribution"].items(),
                                     key=lambda x: -x[1]):
            pct = count / total_buys * 100 if total_buys else 0
            lines.append(f"  · {regime}: {count}회 ({pct:.0f}%)")

    if stats["exit_reason_distribution"]:
        lines.append("")
        lines.append("🚪 청산 사유:")
        total_sells = sum(stats["exit_reason_distribution"].values())
        for reason, count in sorted(stats["exit_reason_distribution"].items(),
                                     key=lambda x: -x[1]):
            pct = count / total_sells * 100 if total_sells else 0
            lines.append(f"  · {reason}: {count}회 ({pct:.0f}%)")

    cmp = compare_to_backtest(stats, state_dir)
    if cmp["available"]:
        lines.append("")
        lines.append("🎯 백테스트 vs 실전:")
        lines.append(f"  승률: {cmp['backtest_win_rate']}% → {cmp['live_win_rate']}% ({cmp['win_rate_diff']:+}%)")
        lines.append(f"  PF: {cmp['backtest_pf']} → {cmp['live_pf']} ({cmp['pf_diff']:+})")
        lines.append(f"  상태: {cmp['status']}")

        if cmp["status"] != "정상":
            lines.append("")
            lines.append("💡 권장: Colab에서 파라미터 재최적화 검토")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# 6. 리포트 트리거 판단
# ═══════════════════════════════════════════════════════════════════

def should_send_weekly_report() -> bool:
    """금요일이면 주간 리포트 전송."""
    return datetime.now().weekday() == 4


def should_send_monthly_report() -> bool:
    """월의 마지막 영업일(평일)이면 월간 리포트 전송."""
    today = datetime.now()
    # 이번 달 마지막 날
    next_month = today.replace(day=28) + timedelta(days=4)
    last_day = next_month - timedelta(days=next_month.day)

    # 마지막 평일 찾기
    while last_day.weekday() >= 5:  # 토(5), 일(6)
        last_day -= timedelta(days=1)

    return today.date() == last_day.date()


# ═══════════════════════════════════════════════════════════════════
# 7. CSV 내보내기 (선택)
# ═══════════════════════════════════════════════════════════════════

def export_history_csv(state_dir: str, output_path: str = None) -> str:
    """매매 이력을 CSV로 내보내기. 분석/백업용."""
    history = load_trade_history(state_dir)
    if not history:
        return ""

    if output_path is None:
        output_path = str(Path(state_dir) / "trade_history_247540.csv")

    import csv
    keys = ["date", "timestamp", "side", "reason", "price", "qty",
            "pnl", "pnl_pct", "regime", "signal", "entry_price", "holding_days"]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for record in history:
            writer.writerow({k: record.get(k, "") for k in keys})

    log.info(f"[EXPORT] 매매 이력 CSV 저장: {output_path} ({len(history)}건)")
    return output_path
