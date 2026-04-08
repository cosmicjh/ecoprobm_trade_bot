"""
Phase 1-2 검증 — 기술적 지표 & 레짐 분류 테스트
"""

import json
import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from indicators import (
    calc_ma, calc_ema, calc_bollinger_bands, calc_atr, calc_rsi,
    calc_volume_indicators, compute_all_indicators, classify_regime,
    add_regime_column, compute_supply_signals, analyze_regime_returns,
)

TEST_DIR = Path("test_phase12")
TEST_DIR.mkdir(exist_ok=True)


def _make_ohlcv(n=100, base_price=200000, seed=42):
    """테스트용 OHLCV DataFrame 생성."""
    np.random.seed(seed)
    dates = pd.bdate_range("2025-01-01", periods=n)
    returns = np.random.normal(0.001, 0.03, n)
    prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        "Open": prices * (1 + np.random.uniform(-0.01, 0.01, n)),
        "High": prices * (1 + np.random.uniform(0.005, 0.04, n)),
        "Low": prices * (1 - np.random.uniform(0.005, 0.04, n)),
        "Close": prices,
        "Volume": np.random.randint(300000, 800000, n),
    }, index=dates).round(0).astype(int)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[f"prev_{col}"] = df[col].shift(1)
    df = df.iloc[1:].copy()
    return df


# ── 개별 지표 테스트 ──

def test_ma():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    ma3 = calc_ma(s, 3)
    assert pd.isna(ma3.iloc[0])
    assert pd.isna(ma3.iloc[1])
    assert ma3.iloc[2] == 2.0  # (1+2+3)/3
    assert ma3.iloc[9] == 9.0  # (8+9+10)/3
    print("✅ MA 계산")


def test_bollinger_bands():
    np.random.seed(42)
    s = pd.Series(np.random.normal(100, 5, 50))
    bb = calc_bollinger_bands(s, period=20, num_std=2.0)

    assert "bb_middle" in bb.columns
    assert "bb_upper" in bb.columns
    assert "bb_lower" in bb.columns
    assert "bb_width" in bb.columns
    assert "bb_pctb" in bb.columns

    # 20일 이후부터 값이 있어야 함
    assert pd.notna(bb["bb_middle"].iloc[19])
    # 상단 > 중심 > 하단
    valid = bb.dropna()
    assert (valid["bb_upper"] >= valid["bb_middle"]).all()
    assert (valid["bb_middle"] >= valid["bb_lower"]).all()
    # %B는 0~1 범위 근처
    assert valid["bb_pctb"].min() >= -0.5  # 약간 벗어날 수 있음
    assert valid["bb_pctb"].max() <= 1.5
    print("✅ 볼린저밴드")


def test_atr():
    df = _make_ohlcv(50)
    atr = calc_atr(df["prev_High"], df["prev_Low"], df["prev_Close"], period=14)
    assert len(atr) == len(df)
    valid = atr.dropna()
    assert len(valid) > 0
    assert (valid > 0).all()  # ATR은 항상 양수
    print("✅ ATR")


def test_rsi():
    # 연속 상승 → RSI > 70
    up = pd.Series([100 + i * 2 for i in range(30)], dtype=float)
    rsi_up = calc_rsi(up, 14)
    assert rsi_up.iloc[-1] > 70, f"상승 RSI={rsi_up.iloc[-1]:.1f}, expected >70"

    # 연속 하락 → RSI < 30
    down = pd.Series([200 - i * 2 for i in range(30)], dtype=float)
    rsi_down = calc_rsi(down, 14)
    assert rsi_down.iloc[-1] < 30, f"하락 RSI={rsi_down.iloc[-1]:.1f}, expected <30"

    # RSI 범위: 0~100
    np.random.seed(42)
    rand = pd.Series(np.random.normal(0, 1, 100).cumsum() + 100)
    rsi_rand = calc_rsi(rand, 14)
    valid = rsi_rand.dropna()
    assert valid.min() >= 0
    assert valid.max() <= 100
    print("✅ RSI")


def test_volume_indicators():
    np.random.seed(42)
    vol = pd.Series(np.random.randint(300000, 600000, 50))
    # 마지막 날 거래량 폭증
    vol.iloc[-1] = 1500000
    vi = calc_volume_indicators(vol, period=20)

    assert "vol_ma20" in vi.columns
    assert "vol_ratio" in vi.columns
    assert "vol_surge" in vi.columns
    assert vi["vol_surge"].iloc[-1] == 1  # 폭증 감지
    print("✅ 거래량 지표")


# ── 통합 지표 테스트 ──

def test_compute_all_indicators():
    df = _make_ohlcv(100)
    result = compute_all_indicators(df)

    expected_cols = [
        "ma20", "ma60", "price_above_ma20", "ma20_above_ma60",
        "bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_pctb",
        "bb_width_ma20", "bb_squeeze",
        "atr14", "atr_ma20", "atr_ratio",
        "rsi14",
        "vol_ma20", "vol_ratio", "vol_surge",
        "daily_return", "range_pct",
    ]
    for col in expected_cols:
        assert col in result.columns, f"컬럼 누락: {col}"

    # NaN이 아닌 행이 있어야 함 (60일 MA 기준, 최소 60일 후)
    valid = result.dropna(subset=["ma60"])
    assert len(valid) > 0, "MA60 이후 데이터 없음"
    print("✅ 통합 지표 계산")


def test_prev_only():
    """★ 핵심 테스트: 지표가 prev_* 컬럼에서만 계산되는지 확인."""
    df = _make_ohlcv(100)

    # 당일 Close를 극단값으로 변경
    df_modified = df.copy()
    df_modified["Close"] = 999999
    df_modified["High"] = 999999
    df_modified["Low"] = 1

    # prev_* 는 그대로이므로, 지표 결과가 동일해야 함
    result_original = compute_all_indicators(df)
    result_modified = compute_all_indicators(df_modified)

    # MA20은 prev_Close에서 계산하므로 동일해야 함
    np.testing.assert_array_equal(
        result_original["ma20"].dropna().values,
        result_modified["ma20"].dropna().values,
    )
    # RSI도 동일
    np.testing.assert_array_equal(
        result_original["rsi14"].dropna().values,
        result_modified["rsi14"].dropna().values,
    )
    print("✅ prev_* 전용 계산 검증 (look-ahead bias 방지)")


# ── 레짐 분류 테스트 ──

def test_regime_trend_up():
    row = pd.Series({
        "ma20": 200000, "ma60": 190000,
        "price_above_ma20": 1, "ma20_above_ma60": 1,
        "atr_ratio": 1.0, "bb_squeeze": 1.0,
    })
    assert classify_regime(row) == "TREND_UP"
    print("✅ 레짐: TREND_UP")


def test_regime_trend_down():
    row = pd.Series({
        "ma20": 180000, "ma60": 190000,
        "price_above_ma20": 0, "ma20_above_ma60": 0,
        "atr_ratio": 1.0, "bb_squeeze": 1.0,
    })
    assert classify_regime(row) == "TREND_DOWN"
    print("✅ 레짐: TREND_DOWN")


def test_regime_high_volatility():
    row = pd.Series({
        "ma20": 200000, "ma60": 190000,
        "price_above_ma20": 1, "ma20_above_ma60": 1,
        "atr_ratio": 2.0,  # > 1.5 임계치
        "bb_squeeze": 1.0,
    })
    # 고변동성이 추세보다 우선
    assert classify_regime(row) == "HIGH_VOLATILITY"
    print("✅ 레짐: HIGH_VOLATILITY (우선순위)")


def test_regime_range_bound():
    row = pd.Series({
        "ma20": 200000, "ma60": 199000,
        "price_above_ma20": 1, "ma20_above_ma60": 1,
        "atr_ratio": 0.8,
        "bb_squeeze": 0.5,  # < 0.7 임계치
    })
    assert classify_regime(row) == "RANGE_BOUND"
    print("✅ 레짐: RANGE_BOUND")


def test_regime_column_and_transitions():
    df = _make_ohlcv(100)
    df = compute_all_indicators(df)
    df = add_regime_column(df)

    assert "regime" in df.columns
    assert "regime_changed" in df.columns
    assert "regime_days" in df.columns

    # 유효한 레짐 값만 포함
    valid_regimes = {"TREND_UP", "TREND_DOWN", "RANGE_BOUND", "HIGH_VOLATILITY", "NEUTRAL", "UNKNOWN"}
    actual = set(df["regime"].unique())
    assert actual.issubset(valid_regimes), f"예상 외 레짐: {actual - valid_regimes}"

    # regime_days는 항상 >= 1
    assert (df["regime_days"] >= 1).all()
    print("✅ 레짐 컬럼 & 전환 감지")


def test_regime_returns_analysis():
    df = _make_ohlcv(200)
    df = compute_all_indicators(df)
    df = add_regime_column(df)
    analysis = analyze_regime_returns(df)

    assert isinstance(analysis, dict)
    # 최소 1개 레짐은 있어야 함
    assert len(analysis) > 0
    for regime, stats in analysis.items():
        assert "count" in stats
        assert "pct" in stats
    print("✅ 레짐별 수익률 분석")


# ── 수급 시그널 테스트 ──

def test_supply_signals():
    supply = {}
    base_date = pd.Timestamp("2026-04-01")
    for i in range(20):
        d = (base_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        supply[d] = {
            "foreign_net_qty": int(np.random.randint(-50000, 50000)),
            "inst_net_qty": int(np.random.randint(-20000, 20000)),
            "short_ratio_vol": round(np.random.uniform(2, 8), 2),
        }

    # 마지막 5일 외국인+기관 동반 매수
    for i in range(15, 20):
        d = (base_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        supply[d]["foreign_net_qty"] = 40000
        supply[d]["inst_net_qty"] = 10000

    df = compute_supply_signals(supply)
    assert "foreign_ma5" in df.columns
    assert "supply_score" in df.columns
    assert "dual_buy" in df.columns

    # 마지막 날은 동반 매수
    assert df["dual_buy"].iloc[-1] == 1
    print("✅ 수급 시그널 계산")


# ── 전체 파이프라인 테스트 ──

def test_full_pipeline():
    """OHLCV JSON → 지표 계산 → 레짐 분류 → 저장 → 재로드 검증."""
    df = _make_ohlcv(150)

    # OHLCV JSON 저장 (Phase 1-1 출력 형식)
    ohlcv_data = {}
    for idx, row in df.iterrows():
        dk = idx.strftime("%Y-%m-%d")
        ohlcv_data[dk] = {
            "Open": int(row["Open"]), "High": int(row["High"]),
            "Low": int(row["Low"]), "Close": int(row["Close"]),
            "Volume": int(row["Volume"]),
            "prev_Open": int(row["prev_Open"]), "prev_High": int(row["prev_High"]),
            "prev_Low": int(row["prev_Low"]), "prev_Close": int(row["prev_Close"]),
            "prev_Volume": int(row["prev_Volume"]),
        }

    ohlcv_path = TEST_DIR / "test_ohlcv.json"
    with open(ohlcv_path, "w") as f:
        json.dump(ohlcv_data, f)

    # 수급 데이터 (간이)
    supply_data = {}
    for dk in list(ohlcv_data.keys())[-20:]:
        supply_data[dk] = {
            "foreign_net_qty": int(np.random.randint(-30000, 30000)),
            "inst_net_qty": int(np.random.randint(-10000, 10000)),
        }
    supply_path = TEST_DIR / "test_supply.json"
    with open(supply_path, "w") as f:
        json.dump(supply_data, f)

    output_path = TEST_DIR / "test_indicators.json"

    from indicators import run_indicator_pipeline
    summary = run_indicator_pipeline(
        ohlcv_path=str(ohlcv_path),
        supply_path=str(supply_path),
        output_path=str(output_path),
    )

    assert summary["total_days"] > 0
    assert output_path.exists()
    assert summary["latest_regime"] in {
        "TREND_UP", "TREND_DOWN", "RANGE_BOUND", "HIGH_VOLATILITY", "NEUTRAL", "UNKNOWN"
    }
    assert summary["latest_rsi"] is None or 0 <= summary["latest_rsi"] <= 100

    # 저장된 파일 검증
    with open(output_path) as f:
        saved = json.load(f)
    assert len(saved) > 0
    sample = list(saved.values())[-1]
    assert "ma20" in sample
    assert "regime" in sample
    assert "rsi14" in sample

    print("✅ 전체 파이프라인 (JSON → 지표 → 레짐 → 저장)")


def run_all():
    print(f"\n{'='*55}")
    print("Phase 1-2 기술적 지표 & 레짐 분류 검증")
    print(f"{'='*55}\n")

    tests = [
        test_ma, test_bollinger_bands, test_atr, test_rsi,
        test_volume_indicators, test_compute_all_indicators, test_prev_only,
        test_regime_trend_up, test_regime_trend_down,
        test_regime_high_volatility, test_regime_range_bound,
        test_regime_column_and_transitions, test_regime_returns_analysis,
        test_supply_signals, test_full_pipeline,
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
