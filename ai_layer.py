"""
에코프로비엠 (247540) v4 — Phase 3: AI 레이어
===============================================
Phase 2의 룰 기반 레짐 분류기를 ML 모델로 강화합니다.

모듈:
  1. HMM 레짐 분류기 — 관측 불가능한 시장 상태를 확률적으로 추정
  2. Isolation Forest — 외국인·기관 수급의 비정상 패턴 감지

설계 원칙:
  - 기존 룰 기반을 '대체'하지 않고, 앙상블로 '보강'
  - ML이 실패하면 룰 기반으로 fallback (안전장치)
  - 학습은 Colab/로컬에서, 추론만 GitHub Actions에서 실행
  - scikit-learn, hmmlearn만 사용 (경량)
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── 선택적 import (GitHub Actions에서 미설치 시 graceful 처리) ──
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    hmm = None
    HMM_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    IsolationForest = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════
# 1. HMM 레짐 분류기
# ═══════════════════════════════════════════════════════════════════

# 레짐 매핑 (HMM state → 해석)
# 학습 후 각 state의 평균 수익률/변동성으로 자동 매핑
REGIME_LABELS = {
    "bullish": "TREND_UP",
    "bearish": "TREND_DOWN",
    "low_vol": "RANGE_BOUND",
    "high_vol": "HIGH_VOLATILITY",
}

class HMMRegimeClassifier:
    """
    Hidden Markov Model 기반 시장 레짐 분류기.

    관측 벡터 (4차원):
      - daily_return: 일간 수익률
      - volume_change: 거래량 변화율
      - bb_position: 볼린저밴드 내 위치 (%B)
      - atr_ratio: ATR / 20일 평균 ATR

    숨은 상태: 3~5개 (BIC로 최적 수 결정)
    """

    def __init__(self, n_states: int = 4, random_state: int = 42):
        self.n_states = n_states
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.state_labels = {}  # {state_idx: regime_name}
        self.trained = False

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """관측 벡터 생성. prev_* 기반."""
        features = pd.DataFrame(index=df.index)

        pc = df["prev_Close"]
        features["daily_return"] = pc.pct_change()
        features["volume_change"] = df["prev_Volume"].pct_change()

        # BB position (%B)
        bb_mid = pc.rolling(20).mean()
        bb_std = pc.rolling(20).std()
        bb_upper = bb_mid + bb_std * 2
        bb_lower = bb_mid - bb_std * 2
        features["bb_position"] = np.where(
            (bb_upper - bb_lower) > 0,
            (pc - bb_lower) / (bb_upper - bb_lower),
            0.5,
        )

        # ATR ratio
        prev_c = pc.shift(1)
        tr = pd.concat([
            df["prev_High"] - df["prev_Low"],
            (df["prev_High"] - prev_c).abs(),
            (df["prev_Low"] - prev_c).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=14, adjust=False).mean()
        atr_ma = atr.rolling(20).mean()
        features["atr_ratio"] = np.where(atr_ma > 0, atr / atr_ma, 1.0)

        features = features.dropna()
        return features

    def train(self, df: pd.DataFrame, select_best_n: bool = True) -> dict:
        """
        HMM 학습.

        Args:
            df: OHLCV DataFrame (prev_* 컬럼 포함)
            select_best_n: True면 BIC로 최적 n_states 선택 (3~5)

        Returns:
            학습 결과 요약 dict
        """
        if not HMM_AVAILABLE:
            raise RuntimeError("hmmlearn 미설치: pip install hmmlearn")

        features_df = self._prepare_features(df)
        X = features_df.values

        # 스케일링
        X_scaled = self.scaler.fit_transform(X)

        # 최적 상태 수 선택 (BIC 기준)
        if select_best_n:
            best_bic = float('inf')
            best_n = self.n_states

            for n in range(3, 6):
                try:
                    model = hmm.GaussianHMM(
                        n_components=n,
                        covariance_type="diag",
                        n_iter=200,
                        random_state=self.random_state,
                    )
                    model.fit(X_scaled)
                    # BIC = -2*log_likelihood + n_params * log(n_samples)
                    ll = model.score(X_scaled)
                    n_params = n * n + n * X_scaled.shape[1] * 2  # transition + emission
                    bic = -2 * ll + n_params * np.log(len(X_scaled))

                    log.info(f"[HMM] n_states={n}: LL={ll:.1f}, BIC={bic:.1f}")
                    if bic < best_bic:
                        best_bic = bic
                        best_n = n
                except Exception as e:
                    log.warning(f"[HMM] n_states={n} 실패: {e}")

            self.n_states = best_n
            log.info(f"[HMM] 최적 상태 수: {best_n}")

        # 최종 학습
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=300,
            random_state=self.random_state,
        )
        self.model.fit(X_scaled)

        # 상태 라벨링 (각 상태의 평균 수익률/변동성으로 해석)
        states = self.model.predict(X_scaled)
        features_df["state"] = states

        # 다음 5일 수익률로 각 상태의 성격 판단
        features_df["fwd_5d"] = features_df["daily_return"].rolling(5).sum().shift(-5)

        self._label_states(features_df)
        self.trained = True

        # 결과 요약
        result = {
            "n_states": self.n_states,
            "log_likelihood": float(self.model.score(X_scaled)),
            "state_distribution": {},
            "transition_matrix": self.model.transmat_.tolist(),
        }

        for s in range(self.n_states):
            mask = features_df["state"] == s
            label = self.state_labels.get(s, f"STATE_{s}")
            count = int(mask.sum())
            mean_ret = float(features_df.loc[mask, "daily_return"].mean() * 100) if count > 0 else 0
            mean_vol = float(features_df.loc[mask, "atr_ratio"].mean()) if count > 0 else 0

            result["state_distribution"][label] = {
                "count": count,
                "pct": round(count / len(features_df) * 100, 1),
                "mean_daily_return": round(mean_ret, 3),
                "mean_atr_ratio": round(mean_vol, 2),
            }

        log.info(f"[HMM] 학습 완료: {self.n_states} states, {len(X_scaled)} samples")
        return result

    def _label_states(self, features_df: pd.DataFrame):
        """각 HMM state에 레짐 라벨 할당."""
        state_stats = []

        for s in range(self.n_states):
            mask = features_df["state"] == s
            if mask.sum() == 0:
                state_stats.append({"state": s, "ret": 0, "vol": 1, "fwd": 0})
                continue

            mean_ret = features_df.loc[mask, "daily_return"].mean()
            mean_vol = features_df.loc[mask, "atr_ratio"].mean()
            mean_fwd = features_df.loc[mask, "fwd_5d"].mean() if "fwd_5d" in features_df else 0

            state_stats.append({
                "state": s,
                "ret": mean_ret,
                "vol": mean_vol,
                "fwd": mean_fwd if pd.notna(mean_fwd) else 0,
            })

        # 변동성 기준 정렬
        sorted_by_vol = sorted(state_stats, key=lambda x: x["vol"])

        # 라벨링 전략:
        # - 최고 변동성 → HIGH_VOLATILITY
        # - 최저 변동성 → RANGE_BOUND
        # - 나머지 중 수익률 양수 → TREND_UP, 음수 → TREND_DOWN
        for ss in sorted_by_vol:
            s = ss["state"]

        # 간단한 규칙 기반 라벨링
        highest_vol_state = sorted_by_vol[-1]["state"]
        lowest_vol_state = sorted_by_vol[0]["state"]

        self.state_labels[highest_vol_state] = "HIGH_VOLATILITY"
        self.state_labels[lowest_vol_state] = "RANGE_BOUND"

        for ss in sorted_by_vol[1:-1]:
            s = ss["state"]
            if ss["fwd"] > 0:
                self.state_labels[s] = "TREND_UP"
            else:
                self.state_labels[s] = "TREND_DOWN"

        log.info(f"[HMM] 상태 라벨: {self.state_labels}")

    def predict(self, df: pd.DataFrame) -> dict:
        """
        현재 레짐 확률 추정.

        Returns:
            {
                "regime": "TREND_UP",
                "confidence": 0.85,
                "all_probs": {"TREND_UP": 0.85, "RANGE_BOUND": 0.10, ...},
                "transition_hint": "TREND_UP → RANGE_BOUND 전환 가능성 15%"
            }
        """
        if not self.trained or self.model is None:
            return {"regime": "UNKNOWN", "confidence": 0, "source": "hmm_not_trained"}

        try:
            features_df = self._prepare_features(df)
            if features_df.empty:
                return {"regime": "UNKNOWN", "confidence": 0, "source": "no_features"}

            X = self.scaler.transform(features_df.values)

            # 최근 60일 데이터로 상태 확률 계산
            window = X[-60:] if len(X) > 60 else X
            state_probs = self.model.predict_proba(window)
            latest_probs = state_probs[-1]

            best_state = int(np.argmax(latest_probs))
            regime = self.state_labels.get(best_state, f"STATE_{best_state}")
            confidence = float(latest_probs[best_state])

            all_probs = {}
            for s, prob in enumerate(latest_probs):
                label = self.state_labels.get(s, f"STATE_{s}")
                all_probs[label] = round(float(prob), 3)

            # 전환 확률 힌트
            trans_probs = self.model.transmat_[best_state]
            transition_hints = []
            for s, tp in enumerate(trans_probs):
                if s != best_state and tp > 0.1:
                    target = self.state_labels.get(s, f"STATE_{s}")
                    transition_hints.append(f"{target} ({tp*100:.0f}%)")

            hint = f"{regime} → " + ", ".join(transition_hints) if transition_hints else ""

            return {
                "regime": regime,
                "confidence": round(confidence, 3),
                "all_probs": all_probs,
                "transition_hint": hint,
                "source": "hmm",
            }

        except Exception as e:
            log.warning(f"[HMM] 예측 실패: {e}")
            return {"regime": "UNKNOWN", "confidence": 0, "source": "hmm_error"}

    def save(self, path: str):
        """모델 저장 (pickle)."""
        data = {
            "model": self.model,
            "scaler": self.scaler,
            "state_labels": self.state_labels,
            "n_states": self.n_states,
            "trained": self.trained,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        log.info(f"[HMM] 모델 저장: {path}")

    def load(self, path: str) -> bool:
        """모델 로드."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.state_labels = data["state_labels"]
            self.n_states = data["n_states"]
            self.trained = data["trained"]
            log.info(f"[HMM] 모델 로드: {path} ({self.n_states} states)")
            return True
        except Exception as e:
            log.warning(f"[HMM] 모델 로드 실패: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════
# 2. Isolation Forest 수급 이상 감지
# ═══════════════════════════════════════════════════════════════════

class SupplyAnomalyDetector:
    """
    Isolation Forest 기반 비정상 수급 패턴 감지.

    입력 피처 (5차원):
      - foreign_net_qty: 외국인 순매수량
      - inst_net_qty: 기관 순매수량
      - combined_net: 외국인+기관 합계
      - short_ratio_vol: 공매도 거래량 비중
      - vol_ratio: 거래량 / 20일 평균

    출력:
      - anomaly_score: 이상치 점수 (음수일수록 이상)
      - is_anomaly: 이상 여부 (True/False)
      - direction: "bullish" / "bearish" / "neutral"
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_names = [
            "foreign_net_qty", "inst_net_qty", "combined_net",
            "short_ratio_vol", "vol_ratio",
        ]
        self.trained = False

    def _prepare_features(self, supply_df: pd.DataFrame, ohlcv_df: pd.DataFrame = None) -> pd.DataFrame:
        """수급 데이터에서 피처 추출."""
        features = pd.DataFrame(index=supply_df.index)

        if "foreign_net_qty" in supply_df.columns:
            features["foreign_net_qty"] = pd.to_numeric(supply_df["foreign_net_qty"], errors="coerce").fillna(0)
        else:
            features["foreign_net_qty"] = 0

        if "inst_net_qty" in supply_df.columns:
            features["inst_net_qty"] = pd.to_numeric(supply_df["inst_net_qty"], errors="coerce").fillna(0)
        else:
            features["inst_net_qty"] = 0

        features["combined_net"] = features["foreign_net_qty"] + features["inst_net_qty"]

        if "short_ratio_vol" in supply_df.columns:
            features["short_ratio_vol"] = pd.to_numeric(supply_df["short_ratio_vol"], errors="coerce").fillna(0)
        else:
            features["short_ratio_vol"] = 0

        # 거래량 비율 (OHLCV에서)
        if ohlcv_df is not None and "prev_Volume" in ohlcv_df.columns:
            vol = ohlcv_df["prev_Volume"].reindex(features.index)
            vol_ma = vol.rolling(20).mean()
            features["vol_ratio"] = np.where(vol_ma > 0, vol / vol_ma, 1.0)
        else:
            features["vol_ratio"] = 1.0

        features = features.fillna(0)
        return features

    def train(self, supply_df: pd.DataFrame, ohlcv_df: pd.DataFrame = None) -> dict:
        """Isolation Forest 학습."""
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn 미설치")

        features = self._prepare_features(supply_df, ohlcv_df)
        if len(features) < 20:
            raise ValueError(f"학습 데이터 부족: {len(features)}행 (최소 20)")

        X = features.values
        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=200,
        )
        self.model.fit(X_scaled)
        self.trained = True

        # 학습 데이터에서 이상치 비율 확인
        predictions = self.model.predict(X_scaled)
        anomaly_count = int((predictions == -1).sum())

        result = {
            "total_samples": len(X_scaled),
            "anomaly_count": anomaly_count,
            "anomaly_rate": round(anomaly_count / len(X_scaled) * 100, 1),
            "contamination": self.contamination,
        }

        log.info(f"[IF] 학습 완료: {len(X_scaled)}샘플, {anomaly_count}개 이상치 ({result['anomaly_rate']}%)")
        return result

    def predict(self, supply_row: dict, ohlcv_row: dict = None) -> dict:
        """
        단일 날짜의 수급 이상 여부 판단.

        Returns:
            {
                "is_anomaly": True/False,
                "anomaly_score": -0.15,  (음수일수록 이상)
                "direction": "bullish" / "bearish" / "neutral",
                "source": "isolation_forest",
            }
        """
        if not self.trained or self.model is None:
            return {"is_anomaly": False, "anomaly_score": 0, "direction": "neutral", "source": "if_not_trained"}

        try:
            # 단일 행 피처 구성
            foreign = float(supply_row.get("foreign_net_qty", 0) or 0)
            inst = float(supply_row.get("inst_net_qty", 0) or 0)
            combined = foreign + inst
            short_r = float(supply_row.get("short_ratio_vol", 0) or 0)
            vol_r = float(ohlcv_row.get("vol_ratio", 1.0) if ohlcv_row else 1.0)

            X = np.array([[foreign, inst, combined, short_r, vol_r]])
            X_scaled = self.scaler.transform(X)

            score = float(self.model.decision_function(X_scaled)[0])
            prediction = int(self.model.predict(X_scaled)[0])
            is_anomaly = prediction == -1

            # 방향 판단
            direction = "neutral"
            if is_anomaly:
                if combined > 0:
                    direction = "bullish"   # 비정상 대량 매수
                elif combined < 0:
                    direction = "bearish"   # 비정상 대량 매도

            return {
                "is_anomaly": is_anomaly,
                "anomaly_score": round(score, 4),
                "direction": direction,
                "source": "isolation_forest",
                "detail": {
                    "foreign": int(foreign),
                    "inst": int(inst),
                    "combined": int(combined),
                },
            }

        except Exception as e:
            log.warning(f"[IF] 예측 실패: {e}")
            return {"is_anomaly": False, "anomaly_score": 0, "direction": "neutral", "source": "if_error"}

    def save(self, path: str):
        data = {"model": self.model, "scaler": self.scaler, "trained": self.trained}
        with open(path, "wb") as f:
            pickle.dump(data, f)
        log.info(f"[IF] 모델 저장: {path}")

    def load(self, path: str) -> bool:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.trained = data["trained"]
            log.info(f"[IF] 모델 로드: {path}")
            return True
        except Exception as e:
            log.warning(f"[IF] 로드 실패: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════
# 3. 앙상블 레짐 판별 (룰 + HMM)
# ═══════════════════════════════════════════════════════════════════

def ensemble_regime(
    rule_regime: str,
    hmm_result: dict,
    hmm_weight: float = 0.4,
) -> dict:
    """
    룰 기반 + HMM 앙상블.

    전략:
      - HMM confidence > 0.7이고 룰과 일치 → 강한 확신
      - HMM confidence > 0.7이고 룰과 불일치 → HMM 우선 (단, HIGH_VOL은 룰 우선)
      - HMM confidence < 0.5 → 룰 기반 유지
      - HMM unavailable → 룰 기반 100%
    """
    hmm_regime = hmm_result.get("regime", "UNKNOWN")
    hmm_conf = hmm_result.get("confidence", 0)

    if hmm_regime == "UNKNOWN" or hmm_conf == 0:
        return {
            "regime": rule_regime,
            "confidence": 0.5,
            "source": "rule_only",
            "detail": f"Rule={rule_regime}",
        }

    # 일치
    if rule_regime == hmm_regime:
        combined_conf = min(0.5 + hmm_conf * hmm_weight, 0.99)
        return {
            "regime": rule_regime,
            "confidence": round(combined_conf, 2),
            "source": "ensemble_agree",
            "detail": f"Rule={rule_regime}, HMM={hmm_regime}({hmm_conf:.0%})",
        }

    # 불일치
    # HIGH_VOLATILITY는 안전 우선 → 룰이 HIGH_VOL이면 항상 유지
    if rule_regime == "HIGH_VOLATILITY":
        return {
            "regime": "HIGH_VOLATILITY",
            "confidence": 0.7,
            "source": "rule_priority_hvol",
            "detail": f"Rule=HIGH_VOL(우선), HMM={hmm_regime}({hmm_conf:.0%})",
        }

    # HMM 확신도 높으면 HMM 우선
    if hmm_conf > 0.7:
        return {
            "regime": hmm_regime,
            "confidence": round(hmm_conf * hmm_weight + 0.3, 2),
            "source": "hmm_override",
            "detail": f"Rule={rule_regime}, HMM={hmm_regime}({hmm_conf:.0%})",
        }

    # 확신 낮으면 룰 유지
    return {
        "regime": rule_regime,
        "confidence": 0.4,
        "source": "rule_low_hmm_conf",
        "detail": f"Rule={rule_regime}, HMM={hmm_regime}({hmm_conf:.0%})",
    }


# ═══════════════════════════════════════════════════════════════════
# 4. 학습 파이프라인 (Colab/로컬 실행)
# ═══════════════════════════════════════════════════════════════════

def train_all_models(
    ohlcv_path: str = "state/ohlcv_247540.json",
    supply_path: str = "state/supply_data_247540.json",
    output_dir: str = "state",
) -> dict:
    """
    HMM + Isolation Forest 전체 학습.
    Colab 또는 로컬에서 실행. GitHub Actions에서는 실행하지 않음.
    """
    log.info(f"{'='*60}")
    log.info("[PHASE 3] AI 모델 학습 시작")
    log.info(f"{'='*60}")

    output_dir = Path(output_dir)
    results = {}

    # OHLCV 로드
    with open(ohlcv_path, "r") as f:
        ohlcv_data = json.load(f)
    rows = [{"date": d, **v} for d, v in sorted(ohlcv_data.items())]
    df_ohlcv = pd.DataFrame(rows)
    df_ohlcv["date"] = pd.to_datetime(df_ohlcv["date"])
    df_ohlcv = df_ohlcv.sort_values("date").set_index("date")

    # ── HMM 학습 ──
    if HMM_AVAILABLE:
        log.info("[HMM] 학습 시작...")
        hmm_clf = HMMRegimeClassifier(n_states=4)
        hmm_result = hmm_clf.train(df_ohlcv, select_best_n=True)
        hmm_clf.save(str(output_dir / "hmm_model_247540.pkl"))
        results["hmm"] = hmm_result

        # 현재 레짐 예측 테스트
        pred = hmm_clf.predict(df_ohlcv)
        log.info(f"[HMM] 현재 레짐: {pred['regime']} (confidence={pred['confidence']:.0%})")
        results["hmm_current"] = pred
    else:
        log.warning("[HMM] hmmlearn 미설치, 건너뜀")
        results["hmm"] = {"error": "hmmlearn not available"}

    # ── Isolation Forest 학습 ──
    if SKLEARN_AVAILABLE and Path(supply_path).exists():
        log.info("[IF] 학습 시작...")
        with open(supply_path, "r") as f:
            supply_data = json.load(f)

        supply_rows = [{"date": d, **v} for d, v in sorted(supply_data.items())]
        df_supply = pd.DataFrame(supply_rows)
        df_supply["date"] = pd.to_datetime(df_supply["date"])
        df_supply = df_supply.sort_values("date").set_index("date")

        if len(df_supply) >= 20:
            detector = SupplyAnomalyDetector(contamination=0.05)
            if_result = detector.train(df_supply, df_ohlcv)
            detector.save(str(output_dir / "if_model_247540.pkl"))
            results["isolation_forest"] = if_result
        else:
            log.warning(f"[IF] 수급 데이터 부족: {len(df_supply)}일")
            results["isolation_forest"] = {"error": "insufficient data"}
    else:
        log.warning("[IF] scikit-learn 미설치 또는 수급 데이터 없음")
        results["isolation_forest"] = {"error": "sklearn not available or no data"}

    # 결과 저장
    results_path = output_dir / "ai_training_result_247540.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"[PHASE 3] 학습 결과 저장: {results_path}")

    return results


# ═══════════════════════════════════════════════════════════════════
# 5. 추론 인터페이스 (trading_bot.py에서 호출)
# ═══════════════════════════════════════════════════════════════════

class AILayer:
    """
    trading_bot.py에서 사용하는 AI 레이어 인터페이스.

    사용법:
        ai = AILayer("state")
        ai.load_models()

        # 레짐 판별 강화
        hmm_result = ai.get_hmm_regime(df_ohlcv)
        ensemble = ensemble_regime(rule_regime, hmm_result)

        # 수급 이상 감지
        anomaly = ai.detect_supply_anomaly(supply_row, ohlcv_row)
    """

    def __init__(self, state_dir: str = "state"):
        self.state_dir = Path(state_dir)
        self.hmm = HMMRegimeClassifier() if HMM_AVAILABLE else None
        self.detector = SupplyAnomalyDetector() if SKLEARN_AVAILABLE else None
        self.models_loaded = False

    def load_models(self) -> bool:
        """학습된 모델 로드. 실패 시 False (룰 기반 fallback)."""
        loaded = False

        if self.hmm:
            hmm_path = self.state_dir / "hmm_model_247540.pkl"
            if hmm_path.exists():
                loaded = self.hmm.load(str(hmm_path)) or loaded

        if self.detector:
            if_path = self.state_dir / "if_model_247540.pkl"
            if if_path.exists():
                loaded = self.detector.load(str(if_path)) or loaded

        self.models_loaded = loaded
        log.info(f"[AI] 모델 로드: {'성공' if loaded else '실패 (룰 기반 fallback)'}")
        return loaded

    def get_hmm_regime(self, df_ohlcv: pd.DataFrame) -> dict:
        """HMM 레짐 예측. 실패 시 빈 결과."""
        if self.hmm and self.hmm.trained:
            return self.hmm.predict(df_ohlcv)
        return {"regime": "UNKNOWN", "confidence": 0, "source": "hmm_unavailable"}

    def detect_supply_anomaly(self, supply_row: dict, ohlcv_row: dict = None) -> dict:
        """수급 이상 감지. 실패 시 정상으로 간주."""
        if self.detector and self.detector.trained:
            return self.detector.predict(supply_row, ohlcv_row)
        return {"is_anomaly": False, "anomaly_score": 0, "direction": "neutral", "source": "if_unavailable"}


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Phase 3: AI 모델 학습")
    parser.add_argument("--state-dir", default="state")
    args = parser.parse_args()

    sd = Path(args.state_dir)
    result = train_all_models(
        ohlcv_path=str(sd / "ohlcv_247540.json"),
        supply_path=str(sd / "supply_data_247540.json"),
        output_dir=str(sd),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
