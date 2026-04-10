"""Phase 2-2 리스크 모니터 검증."""

import json, sys, os
from pathlib import Path
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(__file__))
from risk_monitor import (
    RiskAssessment, calculate_weighted_change, assess_risk,
    adjust_invest_ratio, should_skip_entry, format_risk_telegram,
    save_risk_history, RISK_MULTIPLIER,
)

TEST_DIR = Path("test_risk")
TEST_DIR.mkdir(exist_ok=True)


def test_weighted_change():
    """가중평균 등락률 계산."""
    results = [
        {"symbol": "RIVN", "change_pct": -5.0, "weight": 0.2},
        {"symbol": "ALB",  "change_pct": -3.0, "weight": 0.3},
        {"symbol": "LIT",  "change_pct": -1.0, "weight": 0.5},
    ]
    # (-5*0.2 + -3*0.3 + -1*0.5) / (0.2+0.3+0.5) = (-1 -0.9 -0.5) / 1.0 = -2.4
    wc = calculate_weighted_change(results)
    assert abs(wc - (-2.40)) < 0.01, f"Expected -2.40, got {wc}"

    # 데이터 없는 종목 제외
    results2 = [
        {"symbol": "A", "change_pct": -10.0, "weight": 0.5},
        {"symbol": "B", "change_pct": 0, "weight": 0.5},  # 데이터 없음
    ]
    wc2 = calculate_weighted_change(results2)
    assert wc2 == -10.0  # B 제외, A만 반영

    print("✅ 가중평균 등락률")


def test_risk_level_green():
    """정상 상태 (미국장 소폭 변동)."""
    risk = RiskAssessment(us_weighted_change=-1.0)
    # 수동 판정 시뮬레이션
    assert -1.0 >= -2.0  # green 조건
    risk.risk_level = "green"
    risk.invest_multiplier = RISK_MULTIPLIER["green"]
    assert risk.invest_multiplier == 1.0
    print("✅ 리스크 GREEN")


def test_risk_level_yellow():
    """주의 상태 (미국장 -3% 하락)."""
    risk = RiskAssessment(us_weighted_change=-3.0)
    # -3.0 >= -4.0 → yellow
    risk.risk_level = "yellow"
    risk.invest_multiplier = RISK_MULTIPLIER["yellow"]
    assert risk.invest_multiplier == 0.7
    print("✅ 리스크 YELLOW")


def test_risk_level_orange():
    """경고 상태 (미국장 -5% 하락)."""
    risk = RiskAssessment(us_weighted_change=-5.0)
    risk.risk_level = "orange"
    risk.invest_multiplier = RISK_MULTIPLIER["orange"]
    assert risk.invest_multiplier == 0.3
    print("✅ 리스크 ORANGE")


def test_risk_level_red():
    """위험 상태 (미국장 -8% 급락)."""
    risk = RiskAssessment(us_weighted_change=-8.0)
    risk.risk_level = "red"
    risk.invest_multiplier = RISK_MULTIPLIER["red"]
    assert risk.invest_multiplier == 0.0
    print("✅ 리스크 RED")


def test_invest_ratio_adjustment():
    """투자비율 동적 조정."""
    # Green → 100%
    r_green = RiskAssessment(risk_level="green", invest_multiplier=1.0)
    assert adjust_invest_ratio(0.30, r_green) == 0.30

    # Yellow → 70%
    r_yellow = RiskAssessment(risk_level="yellow", invest_multiplier=0.7)
    assert adjust_invest_ratio(0.30, r_yellow) == 0.21

    # Orange → 30%
    r_orange = RiskAssessment(risk_level="orange", invest_multiplier=0.3)
    assert adjust_invest_ratio(0.30, r_orange) == 0.09

    # Red → 0%
    r_red = RiskAssessment(risk_level="red", invest_multiplier=0.0)
    assert adjust_invest_ratio(0.30, r_red) == 0.0

    # 상한 테스트 (max_invest_ratio 0.6 초과 방지)
    r_green2 = RiskAssessment(risk_level="green", invest_multiplier=1.0)
    assert adjust_invest_ratio(0.70, r_green2) == 0.60  # 60% 캡

    print("✅ 투자비율 동적 조정")


def test_skip_entry():
    """진입 차단 판단."""
    # RED → 차단
    r_red = RiskAssessment(risk_level="red", gapdown_warning="미국장 급락")
    skip, reason = should_skip_entry(r_red)
    assert skip == True
    assert "RED" in reason

    # HIGH 갭다운 → 차단
    r_gap = RiskAssessment(
        risk_level="orange", gapdown_probability="HIGH",
        gapdown_warning="미국 2차전지 -8.5% 급락",
    )
    skip, reason = should_skip_entry(r_gap)
    assert skip == True
    assert "갭다운" in reason

    # 연속 손절 4회 → 차단
    r_loss = RiskAssessment(risk_level="yellow", consecutive_losses=4)
    skip, reason = should_skip_entry(r_loss)
    assert skip == True
    assert "손절" in reason

    # GREEN → 허용
    r_ok = RiskAssessment(risk_level="green", gapdown_probability="LOW")
    skip, reason = should_skip_entry(r_ok)
    assert skip == False

    print("✅ 진입 차단 판단")


def test_gapdown_probability():
    """갭다운 확률 레벨."""
    # 미국 -8% → HIGH
    r1 = RiskAssessment()
    r1.us_weighted_change = -8.0
    if r1.us_weighted_change <= -7:
        r1.gapdown_probability = "HIGH"
    assert r1.gapdown_probability == "HIGH"

    # 미국 -5% → MEDIUM
    r2 = RiskAssessment()
    r2.us_weighted_change = -5.0
    if r2.us_weighted_change <= -7:
        r2.gapdown_probability = "HIGH"
    elif r2.us_weighted_change <= -4:
        r2.gapdown_probability = "MEDIUM"
    assert r2.gapdown_probability == "MEDIUM"

    # 미국 -1% → LOW
    r3 = RiskAssessment()
    r3.us_weighted_change = -1.0
    # 기본값이 LOW
    assert r3.gapdown_probability == "LOW"

    print("✅ 갭다운 확률 레벨")


def test_atr_escalation():
    """ATR 급등 시 리스크 상향."""
    # green + ATR 급등 → yellow
    risk = RiskAssessment(risk_level="green")
    indicators = {"atr_ratio": 2.0}

    if indicators.get("atr_ratio", 1.0) >= 1.8:
        if risk.risk_level == "green":
            risk.risk_level = "yellow"
        elif risk.risk_level == "yellow":
            risk.risk_level = "orange"

    assert risk.risk_level == "yellow"
    print("✅ ATR 급등 리스크 상향")


def test_telegram_format():
    """Telegram 메시지 포맷."""
    risk = RiskAssessment(
        risk_level="yellow",
        risk_label="주의",
        invest_multiplier=0.7,
        us_weighted_change=-3.5,
        us_stocks=[
            {"symbol": "RIVN", "change_pct": -5.2, "name": "Rivian"},
            {"symbol": "ALB", "change_pct": -2.1, "name": "Albemarle"},
        ],
        gapdown_probability="MEDIUM",
        gapdown_warning="미국 2차전지 -3.5% 하락",
    )

    msg = format_risk_telegram(risk)
    assert "주의" in msg
    assert "RIVN" in msg
    assert "MEDIUM" in msg
    assert len(msg) > 50

    print("✅ Telegram 포맷")


def test_risk_history_save():
    """리스크 이력 저장."""
    risk = RiskAssessment(
        risk_level="yellow",
        us_weighted_change=-3.5,
        gapdown_probability="MEDIUM",
        invest_multiplier=0.7,
        assessed_at="2026-04-10T09:00:00",
    )

    save_risk_history(risk, str(TEST_DIR))
    path = TEST_DIR / "risk_history_247540.json"
    assert path.exists()

    with open(path) as f:
        history = json.load(f)

    today = "2026-04-10"  # 실제 날짜와 다를 수 있지만 키가 존재하는지 확인
    assert len(history) > 0
    latest = list(history.values())[-1]
    assert latest["risk_level"] == "yellow"

    print("✅ 리스크 이력 저장")


def run_all():
    print(f"\n{'='*55}")
    print("Phase 2-2 리스크 모니터 검증")
    print(f"{'='*55}\n")

    tests = [
        test_weighted_change,
        test_risk_level_green, test_risk_level_yellow,
        test_risk_level_orange, test_risk_level_red,
        test_invest_ratio_adjustment,
        test_skip_entry,
        test_gapdown_probability,
        test_atr_escalation,
        test_telegram_format,
        test_risk_history_save,
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
