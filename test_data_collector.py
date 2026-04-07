"""
에코프로비엠 v4 Phase 1-1 — 로컬 검증 스크립트
================================================
실제 API 호출 없이 파이프라인 로직의 정합성을 검증합니다.

실행: python test_data_collector.py
"""

import json
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# 테스트 대상 모듈 import
sys.path.insert(0, os.path.dirname(__file__))
from data_collector import (
    save_ohlcv_data,
    save_supply_data,
    save_state,
    load_state,
    _safe_int,
    _parse_investor_response,
    _parse_short_response,
    PIPELINE_STATE_FILE,
    OHLCV_FILE,
    SUPPLY_FILE,
)

# 테스트용 임시 디렉토리
TEST_DIR = Path("test_state")
TEST_DIR.mkdir(exist_ok=True)


def test_safe_int():
    """_safe_int 유틸리티 검증."""
    assert _safe_int(None) == 0
    assert _safe_int("") == 0
    assert _safe_int("12,345") == 12345
    assert _safe_int("-5678") == -5678
    assert _safe_int(3.14) == 3
    assert _safe_int("abc") == 0
    print("✅ _safe_int 테스트 통과")


def test_ohlcv_shift_logic():
    """
    .shift(1) look-ahead bias 방지 로직 검증.
    핵심: 전략 시그널은 반드시 prev_* 컬럼만 사용해야 함.
    """
    # 모의 OHLCV 데이터 생성 (5일치)
    dates = pd.date_range("2026-04-01", periods=5, freq="B")
    data = {
        "Open":   [200000, 202000, 198000, 205000, 210000],
        "High":   [205000, 208000, 203000, 212000, 215000],
        "Low":    [198000, 199000, 195000, 202000, 208000],
        "Close":  [203000, 201000, 200000, 210000, 212000],
        "Volume": [500000, 450000, 600000, 550000, 480000],
    }
    df = pd.DataFrame(data, index=dates)

    # shift(1) 적용
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[f"prev_{col}"] = df[col].shift(1)
    df = df.iloc[1:].copy()  # 첫 행 제거 (prev_* = NaN)

    # 검증: 4월 2일의 prev_Close = 4월 1일의 Close
    apr2 = df.loc[dates[1]]
    assert apr2["prev_Close"] == 203000, f"Expected 203000, got {apr2['prev_Close']}"

    # 검증: 4월 3일의 prev_Volume = 4월 2일의 Volume
    apr3 = df.loc[dates[2]]
    assert apr3["prev_Volume"] == 450000, f"Expected 450000, got {apr3['prev_Volume']}"

    # 검증: shift 후 4행 남아야 함 (5일 - 1 = 4일)
    assert len(df) == 4, f"Expected 4 rows, got {len(df)}"

    print("✅ OHLCV shift(1) 로직 테스트 통과")
    return df


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

    # 저장
    filepath = TEST_DIR / "test_ohlcv.json"
    save_ohlcv_data(df, filepath)

    # 로드 & 검증
    with open(filepath, "r") as f:
        loaded = json.load(f)

    assert len(loaded) == 2, f"Expected 2 dates, got {len(loaded)}"
    assert "2026-04-02" in loaded
    assert loaded["2026-04-02"]["Close"] == 201000
    assert loaded["2026-04-02"]["prev_Close"] == 203000

    print("✅ OHLCV 저장/로드 테스트 통과")


def test_supply_data_accumulation():
    """수급 데이터 누적 저장 검증 — 날짜별 병합이 올바르게 동작하는지."""
    filepath = TEST_DIR / "test_supply.json"

    # 기존 파일 삭제
    if filepath.exists():
        filepath.unlink()

    # 1차 저장: 4/1, 4/2 데이터
    batch1 = {
        "2026-04-01": {"foreign_net_qty": 10000, "inst_net_qty": -5000},
        "2026-04-02": {"foreign_net_qty": 15000, "inst_net_qty": 3000},
    }
    save_supply_data(batch1, filepath)

    # 2차 저장: 4/2 (업데이트), 4/3 (신규) 데이터
    batch2 = {
        "2026-04-02": {"foreign_net_qty": 16000, "inst_net_qty": 3500},  # 수정
        "2026-04-03": {"foreign_net_qty": -8000, "inst_net_qty": 12000},
    }
    save_supply_data(batch2, filepath)

    # 검증
    with open(filepath, "r") as f:
        result = json.load(f)

    assert len(result) == 3, f"Expected 3 dates, got {len(result)}"
    # 4/1은 그대로 유지
    assert result["2026-04-01"]["foreign_net_qty"] == 10000
    # 4/2는 최신값으로 덮어씀
    assert result["2026-04-02"]["foreign_net_qty"] == 16000
    assert result["2026-04-02"]["inst_net_qty"] == 3500
    # 4/3은 신규 추가
    assert result["2026-04-03"]["foreign_net_qty"] == -8000

    print("✅ 수급 데이터 누적 저장 테스트 통과")


def test_investor_response_parsing():
    """한투 API 투자자별 매매동향 응답 파싱 검증."""
    mock_response = {
        "output": [
            {
                "stck_bsop_date": "20260407",
                "frgn_ntby_qty": "46800",
                "frgn_ntby_tr_pbmn": "9500000000",
                "orgn_ntby_qty": "-3000",
                "orgn_ntby_tr_pbmn": "-610000000",
                "prsn_ntby_qty": "-43800",
                "prsn_ntby_tr_pbmn": "-8890000000",
            },
            {
                "stck_bsop_date": "20260404",
                "frgn_ntby_qty": "63500",
                "frgn_ntby_tr_pbmn": "12700000000",
                "orgn_ntby_qty": "2000",
                "orgn_ntby_tr_pbmn": "400000000",
                "prsn_ntby_qty": "-65500",
                "prsn_ntby_tr_pbmn": "-13100000000",
            },
        ]
    }

    records = _parse_investor_response(mock_response)
    assert len(records) == 2, f"Expected 2 records, got {len(records)}"
    assert records[0]["foreign_net_qty"] == 46800
    assert records[0]["inst_net_qty"] == -3000
    assert records[1]["date"] == "20260404"

    print("✅ 투자자 매매동향 파싱 테스트 통과")


def test_short_response_parsing():
    """한투 API 공매도 응답 파싱 검증."""
    mock_response = {
        "output1": [
            {
                "stck_bsop_date": "20260407",
                "seld_cntg_qty": "25000",
                "seld_cntg_amt": "5000000000",
                "acml_vol": "500000",
            },
        ]
    }

    records = _parse_short_response(mock_response)
    assert len(records) == 1
    assert records[0]["short_volume"] == 25000
    assert records[0]["total_volume"] == 500000

    # 공매도 비중 수동 계산 검증
    ratio = records[0]["short_volume"] / records[0]["total_volume"] * 100
    assert round(ratio, 2) == 5.0

    print("✅ 공매도 데이터 파싱 테스트 통과")


def test_state_management():
    """파이프라인 상태 관리 검증."""
    filepath = TEST_DIR / "test_pipeline_state.json"

    # 초기 상태 로드 (파일 없음)
    if filepath.exists():
        filepath.unlink()

    state = load_state(filepath)
    assert state["last_ohlcv_update"] is None
    assert state["total_ohlcv_rows"] == 0

    # 상태 업데이트 & 저장
    state["last_ohlcv_update"] = "2026-04-07"
    state["total_ohlcv_rows"] = 1000
    save_state(state, filepath)

    # 다시 로드하여 확인
    state2 = load_state(filepath)
    assert state2["last_ohlcv_update"] == "2026-04-07"
    assert state2["total_ohlcv_rows"] == 1000
    assert "updated_at" in state2

    print("✅ 상태 관리 테스트 통과")


def test_supply_short_merge():
    """수급 데이터에 공매도 필드 병합 검증."""
    filepath = TEST_DIR / "test_merge.json"
    if filepath.exists():
        filepath.unlink()

    # 수급 데이터 먼저 저장
    supply = {
        "2026-04-07": {
            "foreign_net_qty": 46800,
            "inst_net_qty": -3000,
        }
    }
    save_supply_data(supply, filepath)

    # 공매도 데이터 병합
    with open(filepath, "r") as f:
        existing = json.load(f)

    short_data = {
        "2026-04-07": {
            "short_volume": 25000,
            "short_ratio": 5.0,
        }
    }

    for date_key, vals in short_data.items():
        if date_key in existing:
            existing[date_key].update(vals)

    with open(filepath, "w") as f:
        json.dump(existing, f, indent=2)

    # 검증: 수급 + 공매도 필드 모두 존재
    with open(filepath, "r") as f:
        result = json.load(f)

    apr7 = result["2026-04-07"]
    assert apr7["foreign_net_qty"] == 46800, "수급 데이터 유실"
    assert apr7["short_volume"] == 25000, "공매도 데이터 병합 실패"
    assert apr7["short_ratio"] == 5.0, "공매도 비중 병합 실패"

    print("✅ 수급+공매도 병합 테스트 통과")


def test_data_integrity_scenario():
    """
    실전 시나리오 시뮬레이션:
    3일간 연속 데이터 수집 시 누적 데이터의 정합성 검증.
    """
    filepath = TEST_DIR / "test_scenario.json"
    if filepath.exists():
        filepath.unlink()

    # Day 1 수집
    day1 = {
        "2026-04-07": {
            "foreign_net_qty": 46800,
            "inst_net_qty": -3000,
            "short_volume": 25000,
            "short_ratio": 5.0,
        }
    }
    save_supply_data(day1, filepath)

    # Day 2 수집 (Day 1 데이터도 재수집됨 — incremental 모드)
    day2 = {
        "2026-04-07": {  # 동일 날짜 재수집 — 값 동일해야 하지만 덮어써도 OK
            "foreign_net_qty": 46800,
            "inst_net_qty": -3000,
            "short_volume": 25000,
            "short_ratio": 5.0,
        },
        "2026-04-08": {
            "foreign_net_qty": -12000,
            "inst_net_qty": 8000,
            "short_volume": 30000,
            "short_ratio": 6.5,
        },
    }
    save_supply_data(day2, filepath)

    # Day 3 수집
    day3 = {
        "2026-04-08": {  # 재수집
            "foreign_net_qty": -12000,
            "inst_net_qty": 8000,
            "short_volume": 30000,
            "short_ratio": 6.5,
        },
        "2026-04-09": {
            "foreign_net_qty": 55000,
            "inst_net_qty": 15000,
            "short_volume": 18000,
            "short_ratio": 3.2,
        },
    }
    save_supply_data(day3, filepath)

    # 최종 검증
    with open(filepath, "r") as f:
        result = json.load(f)

    assert len(result) == 3, f"Expected 3 dates, got {len(result)}"
    assert result["2026-04-07"]["foreign_net_qty"] == 46800
    assert result["2026-04-08"]["inst_net_qty"] == 8000
    assert result["2026-04-09"]["foreign_net_qty"] == 55000

    # 수급 변화 추이 확인
    dates = sorted(result.keys())
    foreign_trend = [result[d]["foreign_net_qty"] for d in dates]
    assert foreign_trend == [46800, -12000, 55000], f"Unexpected trend: {foreign_trend}"

    print("✅ 3일 연속 수집 시나리오 테스트 통과")


def run_all_tests():
    """전체 테스트 실행."""
    print(f"\n{'='*50}")
    print("에코프로비엠 v4 Phase 1-1 검증 시작")
    print(f"{'='*50}\n")

    tests = [
        test_safe_int,
        test_ohlcv_shift_logic,
        test_ohlcv_save_load,
        test_supply_data_accumulation,
        test_investor_response_parsing,
        test_short_response_parsing,
        test_state_management,
        test_supply_short_merge,
        test_data_integrity_scenario,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} 실패: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"결과: {passed} 통과 / {failed} 실패 (총 {len(tests)}개)")
    print(f"{'='*50}\n")

    # 테스트 디렉토리 정리
    import shutil
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
        print(f"🧹 테스트 디렉토리 정리 완료: {TEST_DIR}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
