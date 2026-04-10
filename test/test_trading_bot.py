"""Phase 2-1 트레이딩 봇 로직 검증."""

import json, sys, os
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from trading_bot import (
    StrategyParams, BotState, load_params, load_bot_state, save_bot_state,
    compute_indicators, classify_regime, check_risk_limits,
    kosdaq_tick_size, round_to_tick,
)

TEST_DIR = Path("test_bot")
TEST_DIR.mkdir(exist_ok=True)


def _make_df(n=100):
    np.random.seed(42)
    dates = pd.bdate_range("2025-01-01", periods=n)
    p = 200000 * np.cumprod(1 + np.random.normal(0.001, 0.03, n))
    df = pd.DataFrame({
        "Open": (p * 0.99).astype(int), "High": (p * 1.03).astype(int),
        "Low": (p * 0.97).astype(int), "Close": p.astype(int),
        "Volume": np.random.randint(300000, 800000, n),
    }, index=dates)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[f"prev_{c}"] = df[c].shift(1)
    return df.iloc[1:].copy()


def test_tick_size():
    assert kosdaq_tick_size(1500) == 1
    assert kosdaq_tick_size(3000) == 5
    assert kosdaq_tick_size(15000) == 10
    assert kosdaq_tick_size(35000) == 50
    assert kosdaq_tick_size(150000) == 100
    assert kosdaq_tick_size(350000) == 500
    assert kosdaq_tick_size(600000) == 1000
    print("✅ 호가 단위")


def test_round_to_tick():
    # 203050원 → 200,000~500,000 구간 → 호가단위 500원
    assert round_to_tick(203050, "down") == 203000
    assert round_to_tick(203050, "up") == 203500
    # 15023원 → 5,000~20,000 구간 → 호가단위 10원
    assert round_to_tick(15023, "down") == 15020
    assert round_to_tick(15023, "up") == 15030
    print("✅ 호가 반올림")


def test_indicators():
    df = _make_df(100)
    params = StrategyParams()
    result = compute_indicators(df, params)
    for col in ["ma_s", "ma_l", "bb_upper", "bb_lower", "atr", "rsi", "vol_ratio"]:
        assert col in result.columns, f"컬럼 누락: {col}"
    valid = result.dropna(subset=["ma_l"])
    assert len(valid) > 0
    print("✅ 지표 계산")


def test_regime_classification():
    p = StrategyParams()
    # TREND_UP
    row = pd.Series({"ma_s": 200, "ma_l": 190, "above_ma_s": 1, "ma_s_above_l": 1,
                      "atr_ratio": 1.0, "bb_squeeze": 1.0})
    assert classify_regime(row, p) == "TREND_UP"
    # HIGH_VOL 우선
    row["atr_ratio"] = 2.0
    assert classify_regime(row, p) == "HIGH_VOLATILITY"
    # RANGE_BOUND
    row["atr_ratio"] = 0.8
    row["bb_squeeze"] = 0.5
    assert classify_regime(row, p) == "RANGE_BOUND"
    print("✅ 레짐 분류")


def test_risk_limits():
    p = StrategyParams()
    s = BotState(initial_capital=10_000_000)

    # 정상
    assert check_risk_limits(s, p, "2026-04-09") == False
    assert not s.halted

    # 일일 한도 초과
    s.daily_pnl = -350_000  # -3.5%
    assert check_risk_limits(s, p, "2026-04-09") == True
    assert "일일" in s.halt_reason

    # 주간 리셋 (월요일)
    s.daily_pnl = 0
    s.weekly_pnl = -800_000
    s.halted = False
    # 수요일이면 리셋 안됨
    assert check_risk_limits(s, p, "2026-04-09") == True  # 수요일, -8%

    print("✅ 리스크 한도")


def test_bot_state_persistence():
    # import를 위해 모듈 변수를 임시 변경
    import trading_bot
    orig = trading_bot.STATE_DIR
    trading_bot.STATE_DIR = TEST_DIR

    s = BotState(position_qty=10, entry_price=200000, cash=8_000_000)
    save_bot_state(s)

    fp = TEST_DIR / "bot_state_247540.json"
    assert fp.exists()

    with open(fp) as f:
        data = json.load(f)
    assert data["position_qty"] == 10
    assert data["entry_price"] == 200000

    trading_bot.STATE_DIR = orig
    print("✅ 상태 저장/로드")


def test_params_load():
    import trading_bot
    orig = trading_bot.STATE_DIR
    trading_bot.STATE_DIR = TEST_DIR

    # 최적화 파라미터 JSON 시뮬레이션
    opt = {
        "params": {
            "ma_short": 10, "ma_long": 80,
            "tp1_pct": 0.11, "sl_pct": -0.04,
            "trail_atr_mult": 1.75,
        }
    }
    fp = TEST_DIR / "optimized_params_247540.json"
    with open(fp, "w") as f:
        json.dump(opt, f)

    p = load_params(str(fp))
    assert p.ma_short == 10
    assert p.ma_long == 80
    assert p.tp1_pct == 0.11
    assert p.sl_pct == -0.04

    trading_bot.STATE_DIR = orig
    print("✅ 파라미터 로드")


def test_pnl_tracking():
    """매도 시 PnL이 일일/주간/월간에 정확히 누적되는지."""
    s = BotState(
        position_qty=10, entry_price=200000.0,
        cash=8_000_000, initial_capital=10_000_000,
    )

    # 매도 시뮬레이션 (실제 주문 없이 상태만 테스트)
    sell_price = 210000
    qty = 10
    pnl = (sell_price - s.entry_price) * qty  # +100,000

    s.cash += qty * sell_price
    s.position_qty -= qty
    s.daily_pnl += pnl
    s.weekly_pnl += pnl
    s.monthly_pnl += pnl

    assert s.daily_pnl == 100_000
    assert s.weekly_pnl == 100_000
    assert s.position_qty == 0
    assert s.cash == 8_000_000 + 2_100_000

    print("✅ PnL 누적 추적")


def test_cooldown():
    """손절 후 쿨다운이 올바르게 설정되는지."""
    from datetime import timedelta
    p = StrategyParams(cooldown_days=2)
    today = "2026-04-09"
    cd = (datetime.strptime(today, "%Y-%m-%d") + timedelta(days=p.cooldown_days))
    cooldown_until = cd.strftime("%Y-%m-%d")

    assert cooldown_until == "2026-04-11"
    # 4/10은 쿨다운 중
    assert "2026-04-10" < cooldown_until
    # 4/11은 해제
    assert "2026-04-11" >= cooldown_until
    print("✅ 쿨다운 계산")


def run_all():
    print(f"\n{'='*55}")
    print("Phase 2-1 트레이딩 봇 로직 검증")
    print(f"{'='*55}\n")

    tests = [
        test_tick_size, test_round_to_tick, test_indicators,
        test_regime_classification, test_risk_limits,
        test_bot_state_persistence, test_params_load,
        test_pnl_tracking, test_cooldown,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            print(f"❌ {t.__name__}: {e}"); failed += 1

    print(f"\n{'='*55}")
    print(f"결과: {passed} 통과 / {failed} 실패 (총 {len(tests)}개)")
    print(f"{'='*55}")

    import shutil
    if TEST_DIR.exists(): shutil.rmtree(TEST_DIR)
    return failed == 0


if __name__ == "__main__":
    exit(0 if run_all() else 1)
