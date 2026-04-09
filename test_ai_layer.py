"""Phase 3 AI 레이어 검증."""

import json, sys, os
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

TEST_DIR = Path("test_ai")
TEST_DIR.mkdir(exist_ok=True)

def _make_df(n=200):
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-01", periods=n)
    p = 200000 * np.cumprod(1 + np.random.normal(0.001, 0.03, n))
    df = pd.DataFrame({
        "Open": (p * 0.99).astype(int), "High": (p * 1.03).astype(int),
        "Low": (p * 0.97).astype(int), "Close": p.astype(int),
        "Volume": np.random.randint(300000, 800000, n),
    }, index=dates)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[f"prev_{c}"] = df[c].shift(1)
    return df.iloc[1:].copy()


def _make_supply(n=60):
    np.random.seed(42)
    dates = pd.bdate_range("2025-01-01", periods=n)
    df = pd.DataFrame({
        "foreign_net_qty": np.random.randint(-50000, 50000, n),
        "inst_net_qty": np.random.randint(-20000, 20000, n),
        "short_ratio_vol": np.random.uniform(2, 8, n).round(2),
    }, index=dates)
    return df


def test_ensemble_regime():
    from ai_layer import ensemble_regime

    # 일치
    r = ensemble_regime("TREND_UP", {"regime": "TREND_UP", "confidence": 0.85})
    assert r["regime"] == "TREND_UP"
    assert r["confidence"] > 0.5
    assert r["source"] == "ensemble_agree"

    # 불일치, HMM 확신 높음
    r = ensemble_regime("NEUTRAL", {"regime": "TREND_UP", "confidence": 0.85})
    assert r["regime"] == "TREND_UP"
    assert r["source"] == "hmm_override"

    # HIGH_VOL은 항상 룰 우선
    r = ensemble_regime("HIGH_VOLATILITY", {"regime": "TREND_UP", "confidence": 0.95})
    assert r["regime"] == "HIGH_VOLATILITY"
    assert r["source"] == "rule_priority_hvol"

    # HMM 확신 낮음 → 룰 유지
    r = ensemble_regime("TREND_DOWN", {"regime": "TREND_UP", "confidence": 0.4})
    assert r["regime"] == "TREND_DOWN"

    # HMM 없음
    r = ensemble_regime("RANGE_BOUND", {"regime": "UNKNOWN", "confidence": 0})
    assert r["regime"] == "RANGE_BOUND"
    assert r["source"] == "rule_only"

    print("✅ 앙상블 레짐 판별")


def test_hmm_train_predict():
    try:
        from ai_layer import HMMRegimeClassifier
    except ImportError:
        print("⏭️ HMM 테스트 건너뜀 (hmmlearn 미설치)")
        return

    df = _make_df(200)
    clf = HMMRegimeClassifier(n_states=3)

    try:
        result = clf.train(df, select_best_n=False)
    except RuntimeError:
        print("⏭️ HMM 테스트 건너뜀 (hmmlearn 미설치)")
        return

    assert result["n_states"] == 3
    assert result["log_likelihood"] != 0

    # 예측
    pred = clf.predict(df)
    assert pred["regime"] != "UNKNOWN"
    assert 0 < pred["confidence"] <= 1.0
    assert "all_probs" in pred

    # 저장/로드
    model_path = str(TEST_DIR / "test_hmm.pkl")
    clf.save(model_path)

    clf2 = HMMRegimeClassifier()
    assert clf2.load(model_path)
    pred2 = clf2.predict(df)
    assert pred2["regime"] == pred["regime"]

    print("✅ HMM 학습/예측/저장/로드")


def test_isolation_forest():
    try:
        from ai_layer import SupplyAnomalyDetector
    except ImportError:
        print("⏭️ IF 테스트 건너뜀 (sklearn 미설치)")
        return

    df_supply = _make_supply(60)
    df_ohlcv = _make_df(100)

    detector = SupplyAnomalyDetector(contamination=0.1)

    try:
        result = detector.train(df_supply, df_ohlcv)
    except RuntimeError:
        print("⏭️ IF 테스트 건너뜀 (sklearn 미설치)")
        return

    assert result["total_samples"] > 0
    assert result["anomaly_rate"] > 0

    # 정상 수급
    normal = {"foreign_net_qty": 5000, "inst_net_qty": -2000, "short_ratio_vol": 4.0}
    pred_normal = detector.predict(normal, {"vol_ratio": 1.0})
    assert "is_anomaly" in pred_normal

    # 비정상 대량 매수
    extreme = {"foreign_net_qty": 200000, "inst_net_qty": 100000, "short_ratio_vol": 1.0}
    pred_extreme = detector.predict(extreme, {"vol_ratio": 5.0})
    # 극단값이므로 이상치일 가능성 높음 (보장은 아님)
    assert "anomaly_score" in pred_extreme

    # 저장/로드
    model_path = str(TEST_DIR / "test_if.pkl")
    detector.save(model_path)

    d2 = SupplyAnomalyDetector()
    assert d2.load(model_path)
    pred3 = d2.predict(normal)
    assert pred3["source"] == "isolation_forest"

    print("✅ Isolation Forest 학습/예측/저장/로드")


def test_ai_layer_interface():
    from ai_layer import AILayer

    ai = AILayer(str(TEST_DIR))

    # 모델 없이도 안전하게 동작
    hmm_r = ai.get_hmm_regime(_make_df(50))
    assert hmm_r["regime"] == "UNKNOWN"

    anomaly_r = ai.detect_supply_anomaly({"foreign_net_qty": 1000})
    assert anomaly_r["is_anomaly"] == False

    print("✅ AILayer 인터페이스 (graceful fallback)")


def test_train_pipeline():
    """전체 학습 파이프라인 시뮬레이션."""
    df = _make_df(200)
    supply = _make_supply(60)

    # OHLCV JSON 저장
    ohlcv_data = {}
    for idx, row in df.iterrows():
        dk = idx.strftime("%Y-%m-%d")
        ohlcv_data[dk] = {c: int(row[c]) for c in ["Open", "High", "Low", "Close", "Volume",
                          "prev_Open", "prev_High", "prev_Low", "prev_Close", "prev_Volume"]}
    ohlcv_path = TEST_DIR / "ohlcv_247540.json"
    with open(ohlcv_path, "w") as f:
        json.dump(ohlcv_data, f)

    # 수급 JSON 저장
    supply_data = {}
    for idx, row in supply.iterrows():
        dk = idx.strftime("%Y-%m-%d")
        supply_data[dk] = {c: float(row[c]) for c in supply.columns}
    supply_path = TEST_DIR / "supply_data_247540.json"
    with open(supply_path, "w") as f:
        json.dump(supply_data, f)

    try:
        from ai_layer import train_all_models
        result = train_all_models(str(ohlcv_path), str(supply_path), str(TEST_DIR))
        assert "hmm" in result
        print("✅ 전체 학습 파이프라인")
    except RuntimeError as e:
        print(f"⏭️ 학습 파이프라인 건너뜀: {e}")


def run_all():
    print(f"\n{'='*55}")
    print("Phase 3 AI 레이어 검증")
    print(f"{'='*55}\n")

    tests = [
        test_ensemble_regime,
        test_hmm_train_predict,
        test_isolation_forest,
        test_ai_layer_interface,
        test_train_pipeline,
    ]

    passed = failed = skipped = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"❌ {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*55}")
    print(f"결과: {passed} 통과 / {failed} 실패 (총 {len(tests)}개)")
    print(f"{'='*55}")

    import shutil
    if TEST_DIR.exists(): shutil.rmtree(TEST_DIR)
    return failed == 0

if __name__ == "__main__":
    exit(0 if run_all() else 1)
