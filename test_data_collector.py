"""
에코프로비엠 v4 Phase 1-1 (v2) — 로컬 검증 스크립트
====================================================
pykrx 전환 후 파이프라인 로직 검증.
실제 pykrx/KRX 호출 없이 로직 정합성만 테스트.

실행: python test_data_collector.py
"""

import json
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data_collector import (
    save_ohlcv_data,
    save_supply_data,
    save_state,
    load_state,
    _get_col_value,
)

TEST_DIR = Path("test_state")
TEST_DIR.mkdir(exist_ok=True)


def test_ohlcv_shift_logic():
    """.shift(1) look-ahead bias 방지 검증."""
    dates = pd.date_range("2026-04-01", periods=5, freq="B")
    data = {
        "Open":   [200000, 202000, 198000, 205000, 210000],
        "High":   [205000, 208000, 203000, 212000, 215000],
        "Low":    [198000, 199000, 195000, 202000, 208000],
        "Close":  [203000, 201000, 200000, 210000, 212000],
        "Volume": [500000, 450000, 600000, 550000, 480000],
    }
    df = pd.DataFrame(data, index=dates)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[f"prev_{col}"] = df[col].shift(1)
    df = df.iloc[1:].copy()

    # 4/2의 prev_Close = 4/1의 Close
    assert df.loc[dates[1], "prev_Close"] == 203000
    # 4/3의 prev_Volume = 4/2의 Volume
    assert df.loc[dates[2], "prev_Volume"] == 450000
    assert len(df) == 4
    print("✅ OHLCV shift(1) 테스트 통과")


def test_ohlcv_save_load():
    """OHLCV 저장/로드 검증."""
    dates = pd.date_range("2026-04-01", periods=3, freq="B")
    data = {
        "Open": [200000, 202000, 198000],
        "High": [205000, 208000, 203000],
        "Low": [198000, 199000, 195000],
        "Close": [203000, 201000, 200000],
        "Volume": [500000, 450000, 600000],
    }
    df = pd.DataFrame(data, index=dates)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[f"prev_{col}"] = df[col].shift(1)
    df = df.iloc[1:].copy()

    filepath = TEST_DIR / "test_ohlcv.json"
    save_ohlcv_data(df, filepath)

    with open(filepath, "r") as f:
        loaded = json.load(f)

    assert len(loaded) == 2
    assert "2026-04-02" in loaded
    assert loaded["2026-04-02"]["Close"] == 201000
    assert loaded["2026-04-02"]["prev_Close"] == 203000
    print("✅ OHLCV 저장/로드 테스트 통과")


def test_supply_data_accumulation():
    """수급 데이터 누적 저장 — 날짜별 병합 검증."""
    filepath = TEST_DIR / "test_supply.json"
    if filepath.exists():
        filepath.unlink()

    batch1 = {
        "2026-04-01": {"foreign_net_qty": 10000, "inst_net_qty": -5000},
        "2026-04-02": {"foreign_net_qty": 15000, "inst_net_qty": 3000},
    }
    save_supply_data(batch1, filepath)

    batch2 = {
        "2026-04-02": {"foreign_net_qty": 16000, "inst_net_qty": 3500},
        "2026-04-03": {"foreign_net_qty": -8000, "inst_net_qty": 12000},
    }
    save_supply_data(batch2, filepath)

    with open(filepath, "r") as f:
        result = json.load(f)

    assert len(result) == 3
    assert result["2026-04-01"]["foreign_net_qty"] == 10000  # 유지
    assert result["2026-04-02"]["foreign_net_qty"] == 16000  # 덮어씀
    assert result["2026-04-03"]["foreign_net_qty"] == -8000   # 신규
    print("✅ 수급 데이터 누적 저장 테스트 통과")


def test_supply_short_merge():
    """수급 + 공매도 데이터 병합 (save_supply_data의 update 동작)."""
    filepath = TEST_DIR / "test_merge.json"
    if filepath.exists():
        filepath.unlink()

    # 수급 먼저 저장
    supply = {"2026-04-07": {"foreign_net_qty": 46800, "inst_net_qty": -3000}}
    save_supply_data(supply, filepath)

    # 공매도 병합 (같은 날짜에 update)
    short = {"2026-04-07": {"short_volume": 25000, "short_ratio": 5.0}}
    save_supply_data(short, filepath)

    with open(filepath, "r") as f:
        result = json.load(f)

    apr7 = result["2026-04-07"]
    assert apr7["foreign_net_qty"] == 46800, "수급 데이터 유실"
    assert apr7["short_volume"] == 25000, "공매도 병합 실패"
    assert apr7["short_ratio"] == 5.0
    print("✅ 수급+공매도 병합 테스트 통과")


def test_short_balance_merge():
    """수급 + 공매도 + 공매도잔고 3종 데이터 병합."""
    filepath = TEST_DIR / "test_3way_merge.json"
    if filepath.exists():
        filepath.unlink()

    # 순서대로 3종 저장
    save_supply_data({"2026-04-07": {"foreign_net_qty": 46800}}, filepath)
    save_supply_data({"2026-04-07": {"short_volume": 25000}}, filepath)
    save_supply_data({"2026-04-07": {"short_balance": 3400000, "short_balance_ratio": 0.06}}, filepath)

    with open(filepath, "r") as f:
        result = json.load(f)

    apr7 = result["2026-04-07"]
    assert apr7["foreign_net_qty"] == 46800
    assert apr7["short_volume"] == 25000
    assert apr7["short_balance"] == 3400000
    assert apr7["short_balance_ratio"] == 0.06
    print("✅ 3종 데이터 병합 테스트 통과")


def test_get_col_value():
    """pykrx 컬럼명 유연 매칭 검증."""
    # 시나리오 1: 정확한 이름
    row = pd.Series({"외국인합계": 46800, "기관합계": -3000, "개인": -43800})
    assert _get_col_value(row, ["외국인합계", "외국인"]) == 46800
    assert _get_col_value(row, ["기관합계", "기관"]) == -3000

    # 시나리오 2: 대체 이름
    row2 = pd.Series({"외국인": 12000, "기관": 5000})
    assert _get_col_value(row2, ["외국인합계", "외국인"]) == 12000

    # 시나리오 3: 없는 컬럼
    assert _get_col_value(row, ["없는컬럼"]) == 0.0

    # 시나리오 4: NaN 값
    row3 = pd.Series({"외국인합계": float("nan"), "외국인": 9999})
    assert _get_col_value(row3, ["외국인합계", "외국인"]) == 9999

    print("✅ 컬럼명 유연 매칭 테스트 통과")


def test_state_management():
    """상태 관리 검증."""
    filepath = TEST_DIR / "test_state.json"
    if filepath.exists():
        filepath.unlink()

    state = load_state(filepath)
    assert state["last_ohlcv_update"] is None
    assert state["data_source"] == "pykrx"

    state["last_ohlcv_update"] = "2026-04-07"
    state["total_ohlcv_rows"] = 1000
    save_state(state, filepath)

    state2 = load_state(filepath)
    assert state2["last_ohlcv_update"] == "2026-04-07"
    assert state2["total_ohlcv_rows"] == 1000
    assert "updated_at" in state2
    print("✅ 상태 관리 테스트 통과")


def test_date_format_normalization():
    """pykrx 날짜 포맷(YYYYMMDD) 처리 검증."""
    # 하이픈 → 제거
    date1 = "2026-04-07"
    assert date1.replace("-", "") == "20260407"

    # 이미 YYYYMMDD → 그대로
    date2 = "20260407"
    assert date2.replace("-", "") == "20260407"

    print("✅ 날짜 포맷 정규화 테스트 통과")


def test_3day_scenario():
    """3일 연속 수집 시나리오 시뮬레이션."""
    filepath = TEST_DIR / "test_scenario.json"
    if filepath.exists():
        filepath.unlink()

    # Day 1: 수급 + 공매도
    save_supply_data({
        "2026-04-07": {
            "foreign_net_qty": 46800, "inst_net_qty": -3000,
            "individual_net_qty": -43800,
        }
    }, filepath)
    save_supply_data({
        "2026-04-07": {"short_volume": 25000, "short_ratio": 5.0}
    }, filepath)

    # Day 2
    save_supply_data({
        "2026-04-07": {"foreign_net_qty": 46800, "inst_net_qty": -3000},
        "2026-04-08": {"foreign_net_qty": -12000, "inst_net_qty": 8000},
    }, filepath)
    save_supply_data({
        "2026-04-08": {"short_volume": 30000, "short_ratio": 6.5}
    }, filepath)

    # Day 3
    save_supply_data({
        "2026-04-09": {"foreign_net_qty": 55000, "inst_net_qty": 15000}
    }, filepath)
    save_supply_data({
        "2026-04-09": {
            "short_volume": 18000, "short_ratio": 3.2,
            "short_balance": 3300000, "short_balance_ratio": 0.055,
        }
    }, filepath)

    with open(filepath, "r") as f:
        result = json.load(f)

    assert len(result) == 3
    # Day 1: 수급 + 공매도 모두 있어야 함
    assert result["2026-04-07"]["foreign_net_qty"] == 46800
    assert result["2026-04-07"]["short_volume"] == 25000
    # Day 3: 잔고 데이터도 포함
    assert result["2026-04-09"]["short_balance"] == 3300000

    # 수급 변화 추이
    dates = sorted(result.keys())
    foreign = [result[d]["foreign_net_qty"] for d in dates]
    assert foreign == [46800, -12000, 55000]
    print("✅ 3일 연속 수집 시나리오 테스트 통과")


def run_all_tests():
    print(f"\n{'='*55}")
    print("에코프로비엠 v4 Phase 1-1 (v2 pykrx) 검증 시작")
    print(f"{'='*55}\n")

    tests = [
        test_ohlcv_shift_logic,
        test_ohlcv_save_load,
        test_supply_data_accumulation,
        test_supply_short_merge,
        test_short_balance_merge,
        test_get_col_value,
        test_state_management,
        test_date_format_normalization,
        test_3day_scenario,
    ]

    passed = failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} 실패: {e}")
            failed += 1

    print(f"\n{'='*55}")
    print(f"결과: {passed} 통과 / {failed} 실패 (총 {len(tests)}개)")
    print(f"{'='*55}\n")

    import shutil
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
        print(f"🧹 테스트 디렉토리 정리 완료")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
