"""
에코프로비엠 v4 Phase 1-1 (v3 한투 API 전용) — 로컬 검증
"""

import json
import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data_collector import (
    save_ohlcv_data, save_supply_data, save_state, load_state, _safe_int,
)

TEST_DIR = Path("test_state")
TEST_DIR.mkdir(exist_ok=True)


def test_safe_int():
    assert _safe_int(None) == 0
    assert _safe_int("") == 0
    assert _safe_int("12,345") == 12345
    assert _safe_int("-5678") == -5678
    assert _safe_int("abc") == 0
    print("✅ _safe_int")


def test_ohlcv_shift():
    dates = pd.date_range("2026-04-01", periods=5, freq="B")
    df = pd.DataFrame({
        "Open": [200000, 202000, 198000, 205000, 210000],
        "High": [205000, 208000, 203000, 212000, 215000],
        "Low": [198000, 199000, 195000, 202000, 208000],
        "Close": [203000, 201000, 200000, 210000, 212000],
        "Volume": [500000, 450000, 600000, 550000, 480000],
    }, index=dates)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[f"prev_{col}"] = df[col].shift(1)
    df = df.iloc[1:].copy()
    assert df.loc[dates[1], "prev_Close"] == 203000
    assert len(df) == 4
    print("✅ OHLCV shift(1)")


def test_ohlcv_save_load():
    dates = pd.date_range("2026-04-01", periods=3, freq="B")
    df = pd.DataFrame({
        "Open": [200000, 202000, 198000], "High": [205000, 208000, 203000],
        "Low": [198000, 199000, 195000], "Close": [203000, 201000, 200000],
        "Volume": [500000, 450000, 600000],
    }, index=dates)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[f"prev_{col}"] = df[col].shift(1)
    df = df.iloc[1:].copy()

    fp = TEST_DIR / "test_ohlcv.json"
    save_ohlcv_data(df, fp)
    with open(fp) as f:
        loaded = json.load(f)
    assert len(loaded) == 2
    assert loaded["2026-04-02"]["prev_Close"] == 203000
    print("✅ OHLCV 저장/로드")


def test_supply_accumulation():
    fp = TEST_DIR / "test_supply.json"
    if fp.exists(): fp.unlink()
    save_supply_data({"2026-04-01": {"foreign_net_qty": 10000}}, fp)
    save_supply_data({"2026-04-01": {"short_volume": 5000}, "2026-04-02": {"foreign_net_qty": -8000}}, fp)
    with open(fp) as f:
        r = json.load(f)
    assert len(r) == 2
    assert r["2026-04-01"]["foreign_net_qty"] == 10000  # 유지
    assert r["2026-04-01"]["short_volume"] == 5000       # 병합
    assert r["2026-04-02"]["foreign_net_qty"] == -8000    # 신규
    print("✅ 수급 누적 저장 + 필드 병합")


def test_state_management():
    fp = TEST_DIR / "test_state.json"
    if fp.exists(): fp.unlink()
    s = load_state(fp)
    assert s["data_source"] == "KIS_OpenAPI"
    s["last_ohlcv_update"] = "2026-04-08"
    save_state(s, fp)
    s2 = load_state(fp)
    assert s2["last_ohlcv_update"] == "2026-04-08"
    print("✅ 상태 관리")


def test_kis_response_parsing():
    """한투 API 투자자별 매매동향 응답 구조 파싱 시뮬레이션."""
    mock_output = [
        {"stck_bsop_date": "20260407", "frgn_ntby_qty": "46800",
         "frgn_ntby_tr_pbmn": "9500000000", "orgn_ntby_qty": "-3000",
         "orgn_ntby_tr_pbmn": "-610000000", "prsn_ntby_qty": "-43800",
         "prsn_ntby_tr_pbmn": "-8890000000"},
        {"stck_bsop_date": "20260404", "frgn_ntby_qty": "63500",
         "frgn_ntby_tr_pbmn": "12700000000", "orgn_ntby_qty": "2000",
         "orgn_ntby_tr_pbmn": "400000000", "prsn_ntby_qty": "-65500",
         "prsn_ntby_tr_pbmn": "-13100000000"},
    ]
    records = []
    for item in mock_output:
        d = item["stck_bsop_date"]
        records.append({
            "date": f"{d[:4]}-{d[4:6]}-{d[6:]}",
            "foreign_net_qty": _safe_int(item["frgn_ntby_qty"]),
            "inst_net_qty": _safe_int(item["orgn_ntby_qty"]),
        })
    assert len(records) == 2
    assert records[0]["foreign_net_qty"] == 46800
    assert records[0]["date"] == "2026-04-07"
    assert records[1]["inst_net_qty"] == 2000
    print("✅ 한투 API 응답 파싱")


def test_ohlcv_pagination_dedup():
    """OHLCV 페이징 시 중복 제거 검증."""
    rows = [
        {"date": "20260401", "Close": 200000},
        {"date": "20260402", "Close": 201000},
        {"date": "20260402", "Close": 201000},  # 중복
        {"date": "20260403", "Close": 202000},
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.drop_duplicates(subset="date").sort_values("date")
    assert len(df) == 3
    print("✅ OHLCV 페이징 중복 제거")


def test_3day_scenario():
    fp = TEST_DIR / "test_scenario.json"
    if fp.exists(): fp.unlink()

    # Day 1
    save_supply_data({"2026-04-07": {"foreign_net_qty": 46800, "inst_net_qty": -3000}}, fp)
    save_supply_data({"2026-04-07": {"short_volume": 25000, "short_ratio": 5.0}}, fp)

    # Day 2
    save_supply_data({"2026-04-08": {"foreign_net_qty": -12000, "inst_net_qty": 8000}}, fp)
    save_supply_data({"2026-04-08": {"short_volume": 30000}}, fp)

    # Day 3
    save_supply_data({"2026-04-09": {"foreign_net_qty": 55000, "inst_net_qty": 15000, "short_volume": 18000}}, fp)

    with open(fp) as f:
        r = json.load(f)
    assert len(r) == 3
    assert r["2026-04-07"]["foreign_net_qty"] == 46800
    assert r["2026-04-07"]["short_volume"] == 25000
    assert r["2026-04-08"]["short_volume"] == 30000
    assert r["2026-04-09"]["foreign_net_qty"] == 55000

    dates = sorted(r.keys())
    foreign = [r[d]["foreign_net_qty"] for d in dates]
    assert foreign == [46800, -12000, 55000]
    print("✅ 3일 연속 시나리오")


def run_all():
    print(f"\n{'='*55}")
    print("에코프로비엠 v4 Phase 1-1 (v3 한투 API) 검증")
    print(f"{'='*55}\n")

    tests = [
        test_safe_int, test_ohlcv_shift, test_ohlcv_save_load,
        test_supply_accumulation, test_state_management,
        test_kis_response_parsing, test_ohlcv_pagination_dedup,
        test_3day_scenario,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            print(f"❌ {t.__name__}: {e}"); failed += 1

    print(f"\n{'='*55}")
    print(f"결과: {passed} 통과 / {failed} 실패 (총 {len(tests)}개)")
    print(f"{'='*55}\n")

    import shutil
    if TEST_DIR.exists(): shutil.rmtree(TEST_DIR)
    return failed == 0


if __name__ == "__main__":
    exit(0 if run_all() else 1)
