"""Phase 2-4 리포트 & 성과 집계 검증."""

import os, sys, json
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from reporter import (
    log_trade, load_trade_history, compute_period_stats,
    compare_to_backtest, format_daily_report, format_weekly_report,
    format_monthly_report, should_send_weekly_report, should_send_monthly_report,
    export_history_csv,
)

TEST_DIR = Path("test_phase24")
TEST_DIR.mkdir(exist_ok=True)


@dataclass
class MockState:
    position_qty: int = 0
    entry_price: float = 0.0
    cash: float = 1_500_000.0
    initial_capital: float = 1_500_000.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    last_regime: str = ""


def _seed_history():
    """테스트용 매매 이력 생성."""
    fp = TEST_DIR / "trade_history_247540.json"
    if fp.exists(): fp.unlink()

    base = datetime(2026, 4, 1)

    # 시나리오: 5번 매매 (3승 2패)
    trades = [
        # Trade 1: 매수 4/1, 익절 4/3 (+5%)
        {"side": "buy", "reason": "ENTRY", "price": 200000, "qty": 5, "regime": "TREND_UP", "days_offset": 0},
        {"side": "sell", "reason": "TP1", "price": 210000, "qty": 5, "regime": "TREND_UP", "days_offset": 2,
         "entry_price": 200000.0, "entry_date": "2026-04-01"},

        # Trade 2: 매수 4/4, 손절 4/5 (-4%)
        {"side": "buy", "reason": "ENTRY", "price": 215000, "qty": 4, "regime": "TREND_UP", "days_offset": 3},
        {"side": "sell", "reason": "SL", "price": 206400, "qty": 4, "regime": "TREND_UP", "days_offset": 4,
         "entry_price": 215000.0, "entry_date": "2026-04-04"},

        # Trade 3: 매수 4/8, 트레일링 4/12 (+8%)
        {"side": "buy", "reason": "ENTRY", "price": 200000, "qty": 5, "regime": "RANGE_BOUND", "days_offset": 7},
        {"side": "sell", "reason": "TRAIL", "price": 216000, "qty": 5, "regime": "TREND_UP", "days_offset": 11,
         "entry_price": 200000.0, "entry_date": "2026-04-08"},

        # Trade 4: 매수 4/15, 손절 4/16 (-3%)
        {"side": "buy", "reason": "ENTRY", "price": 220000, "qty": 4, "regime": "TREND_UP", "days_offset": 14},
        {"side": "sell", "reason": "SL", "price": 213400, "qty": 4, "regime": "TREND_UP", "days_offset": 15,
         "entry_price": 220000.0, "entry_date": "2026-04-15"},

        # Trade 5: 매수 4/18, 익절 4/22 (+11%)
        {"side": "buy", "reason": "ENTRY", "price": 200000, "qty": 5, "regime": "HIGH_VOLATILITY", "days_offset": 17},
        {"side": "sell", "reason": "TP1", "price": 222000, "qty": 5, "regime": "TREND_UP", "days_offset": 21,
         "entry_price": 200000.0, "entry_date": "2026-04-18"},
    ]

    history = []
    for t in trades:
        date = (base + timedelta(days=t["days_offset"])).strftime("%Y-%m-%d")
        record = {
            "date": date,
            "timestamp": (base + timedelta(days=t["days_offset"], hours=9, minutes=5)).isoformat(),
            "side": t["side"],
            "reason": t["reason"],
            "price": t["price"],
            "qty": t["qty"],
            "pnl": (t["price"] - t.get("entry_price", 0)) * t["qty"] if t["side"] == "sell" else 0,
            "pnl_pct": round((t["price"] - t.get("entry_price", 0)) / t.get("entry_price", 1) * 100, 2) if t["side"] == "sell" else 0.0,
            "regime": t["regime"],
            "signal": "",
            "entry_price": t.get("entry_price", 0.0),
            "holding_days": 0,
        }
        history.append(record)

    with open(fp, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return fp


# ═══════════════════════════════════════════════════════════
# 테스트
# ═══════════════════════════════════════════════════════════

def test_log_trade():
    fp = TEST_DIR / "trade_history_247540.json"
    if fp.exists(): fp.unlink()

    log_trade(str(TEST_DIR), side="buy", reason="ENTRY",
              price=200000, qty=10, regime="TREND_UP")
    log_trade(str(TEST_DIR), side="sell", reason="TP1",
              price=215000, qty=10, pnl=150000,
              entry_price=200000, regime="TREND_UP",
              entry_date="2026-04-01")

    history = load_trade_history(str(TEST_DIR))
    assert len(history) == 2
    assert history[0]["side"] == "buy"
    assert history[1]["pnl"] == 150000
    assert history[1]["pnl_pct"] == 7.5  # (215000-200000)/200000*100

    fp.unlink()
    print("✅ 매매 이력 로깅")


def test_compute_period_stats():
    _seed_history()
    history = load_trade_history(str(TEST_DIR))

    stats = compute_period_stats(history, "2026-04-01", "2026-04-30")

    assert stats["total_trades"] == 10
    assert stats["buy_count"] == 5
    assert stats["sell_count"] == 5
    assert stats["win_count"] == 3   # TP1, TRAIL, TP1
    assert stats["loss_count"] == 2  # SL, SL
    assert stats["win_rate"] == 60.0

    # PnL 검증
    # Trade 1: (210000-200000)*5 = 50000
    # Trade 2: (206400-215000)*4 = -34400
    # Trade 3: (216000-200000)*5 = 80000
    # Trade 4: (213400-220000)*4 = -26400
    # Trade 5: (222000-200000)*5 = 110000
    # Total: 50000-34400+80000-26400+110000 = 179200
    assert stats["total_pnl"] == 179200

    assert stats["regime_distribution"].get("TREND_UP", 0) == 3
    assert stats["regime_distribution"].get("RANGE_BOUND", 0) == 1
    assert stats["regime_distribution"].get("HIGH_VOLATILITY", 0) == 1

    assert stats["exit_reason_distribution"].get("TP1", 0) == 2
    assert stats["exit_reason_distribution"].get("SL", 0) == 2
    assert stats["exit_reason_distribution"].get("TRAIL", 0) == 1

    print("✅ 기간 통계 계산")


def test_consecutive_streak():
    _seed_history()
    history = load_trade_history(str(TEST_DIR))
    stats = compute_period_stats(history, "2026-04-01", "2026-04-30")

    # 시퀀스: 익(50K), 손(-34K), 익(80K), 손(-26K), 익(110K)
    # max_consecutive_wins = 1, max_consecutive_losses = 1
    # current_streak: 마지막이 익절이므로 wins=1, losses=0
    assert stats["max_consecutive_wins"] >= 1
    assert stats["max_consecutive_losses"] >= 1
    assert stats["current_streak_wins"] == 1
    print("✅ 연속 승/패 추적")


def test_profit_factor():
    _seed_history()
    history = load_trade_history(str(TEST_DIR))
    stats = compute_period_stats(history, "2026-04-01", "2026-04-30")

    # gross_profit = 50000 + 80000 + 110000 = 240000
    # gross_loss = |-34400 + -26400| = 60800
    # PF = 240000 / 60800 ≈ 3.95
    assert 3.5 < stats["profit_factor"] < 4.5
    print(f"✅ Profit Factor 계산 (PF={stats['profit_factor']})")


def test_compare_to_backtest():
    """백테스트 vs 실전 비교."""
    # 가상의 optimized_params 생성
    opt_path = TEST_DIR / "optimized_params_247540.json"
    with open(opt_path, "w") as f:
        json.dump({
            "params": {},
            "backtest_result": {"win_rate": 64.3, "profit_factor": 2.49}
        }, f)

    _seed_history()
    history = load_trade_history(str(TEST_DIR))
    stats = compute_period_stats(history, "2026-04-01", "2026-04-30")

    cmp = compare_to_backtest(stats, str(TEST_DIR))

    assert cmp["available"] == True
    assert cmp["backtest_win_rate"] == 64.3
    assert cmp["live_win_rate"] == 60.0
    assert cmp["win_rate_diff"] == -4.3  # 60 - 64.3
    print("✅ 백테스트 vs 실전 비교")


def test_compare_unavailable():
    """파라미터 파일 없을 때."""
    empty_dir = TEST_DIR / "empty"
    empty_dir.mkdir(exist_ok=True)

    cmp = compare_to_backtest({}, str(empty_dir))
    assert cmp["available"] == False
    print("✅ 백테스트 파일 없음 처리")


def test_daily_report():
    _seed_history()
    state = MockState(
        position_qty=5, entry_price=200000.0,
        cash=500000, daily_pnl=50000,
        weekly_pnl=80000, monthly_pnl=179200,
        last_regime="TREND_UP",
    )

    report = format_daily_report(state, current_price=210000, state_dir=str(TEST_DIR))

    assert "일일 리포트" in report
    assert "포지션" in report
    assert "5주" in report
    assert "+50,000" in report or "50,000" in report
    print("✅ 일일 리포트 포맷")


def test_weekly_report():
    _seed_history()
    state = MockState(
        cash=1_679_200, initial_capital=1_500_000,
        weekly_pnl=179200,
    )

    # 명시적 기간 지정 (시드 데이터 범위)
    report = format_weekly_report(state, str(TEST_DIR),
                                   week_start="2026-04-01", week_end="2026-04-30")

    assert "주간 성과" in report
    assert "Profit Factor" in report
    assert "승률" in report
    assert "진입 레짐" in report
    print("✅ 주간 리포트 포맷")


def test_monthly_report():
    _seed_history()
    state = MockState(
        cash=1_679_200, initial_capital=1_500_000,
        monthly_pnl=179200,
    )

    report = format_monthly_report(state, str(TEST_DIR),
                                    month_start="2026-04-01", month_end="2026-04-30")

    assert "월간 성과" in report
    assert "수익률" in report
    assert "레짐별" in report
    print("✅ 월간 리포트 포맷")


def test_weekly_trigger():
    """금요일 판정."""
    # 함수가 datetime.now()를 호출하므로, 결과는 환경에 따라 다름
    # 단순히 함수 실행만 검증
    result = should_send_weekly_report()
    assert isinstance(result, bool)
    print(f"✅ 주간 트리거 함수 (오늘: {'금요일' if result else '평일'})")


def test_monthly_trigger():
    result = should_send_monthly_report()
    assert isinstance(result, bool)
    print("✅ 월간 트리거 함수")


def test_csv_export():
    _seed_history()
    csv_path = export_history_csv(str(TEST_DIR))
    assert Path(csv_path).exists()
    with open(csv_path, "r") as f:
        content = f.read()
    assert "date,timestamp,side" in content
    assert "buy" in content
    assert "sell" in content
    print("✅ CSV 내보내기")


def test_empty_period():
    """매매 이력이 없는 기간."""
    fp = TEST_DIR / "trade_history_247540.json"
    if fp.exists(): fp.unlink()

    stats = compute_period_stats([], "2026-01-01", "2026-01-31")
    assert stats["total_trades"] == 0
    assert stats["win_rate"] == 0
    assert stats["profit_factor"] == 0
    print("✅ 빈 기간 처리")


def run_all():
    print(f"\n{'='*55}")
    print("Phase 2-4 리포트 & 성과 집계 검증")
    print(f"{'='*55}\n")

    tests = [
        test_log_trade,
        test_compute_period_stats,
        test_consecutive_streak,
        test_profit_factor,
        test_compare_to_backtest,
        test_compare_unavailable,
        test_daily_report,
        test_weekly_report,
        test_monthly_report,
        test_weekly_trigger,
        test_monthly_trigger,
        test_csv_export,
        test_empty_period,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            import traceback
            print(f"❌ {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*55}")
    print(f"결과: {passed} 통과 / {failed} 실패 (총 {len(tests)}개)")
    print(f"{'='*55}")

    import shutil
    if TEST_DIR.exists(): shutil.rmtree(TEST_DIR)
    return failed == 0


if __name__ == "__main__":
    exit(0 if run_all() else 1)
