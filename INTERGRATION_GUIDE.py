"""
trading_bot.py에 AI 레이어를 통합하는 방법
==========================================

trading_bot.py의 _run_morning() 함수에 아래 코드를 추가하면
HMM 앙상블 레짐 판별 + 수급 이상 감지가 활성화됩니다.

변경이 필요한 부분만 표시합니다.
"""

# ─────────────────────────────────────
# 1. trading_bot.py 상단에 import 추가
# ─────────────────────────────────────
"""
# 기존 import 아래에 추가
from ai_layer import AILayer, ensemble_regime
"""


# ─────────────────────────────────────
# 2. run_bot() 함수에서 AI 초기화 추가
# ─────────────────────────────────────
"""
def run_bot(mode: str = "morning"):
    # ... 기존 코드 ...
    
    # ★ AI 레이어 로드 (없으면 자동 fallback)
    ai = AILayer(str(STATE_DIR))
    ai.load_models()
    
    if mode == "morning":
        _run_morning(client, params, state, today, ai)  # ← ai 인자 추가
    # ...
"""


# ─────────────────────────────────────
# 3. _run_morning() 함수 수정
# ─────────────────────────────────────
"""
def _run_morning(client, params, state, today, ai=None):  # ← ai 인자 추가
    
    # ... 기존 시세 조회, 지표 계산 코드 ...
    
    # ── 레짐 판별 (기존) ──
    regime = classify_regime(latest, params)     # 룰 기반
    
    # ★ AI 앙상블 (추가)
    hmm_result = {"regime": "UNKNOWN", "confidence": 0}
    supply_anomaly = {"is_anomaly": False, "direction": "neutral"}
    
    if ai and ai.models_loaded:
        hmm_result = ai.get_hmm_regime(df)
        ensemble = ensemble_regime(regime, hmm_result)
        regime = ensemble["regime"]    # ← 앙상블 결과로 교체
        log.info(f"[AI] 앙상블: {ensemble['detail']} → {regime} ({ensemble['confidence']:.0%})")
        
        # 수급 이상 감지
        supply_anomaly = ai.detect_supply_anomaly(
            {"foreign_net_qty": inv.get("latest_foreign", 0),
             "inst_net_qty": inv.get("latest_inst", 0),
             "short_ratio_vol": 0},
            {"vol_ratio": latest.get("vol_ratio", 1.0)},
        )
        if supply_anomaly["is_anomaly"]:
            log.info(f"[AI] 수급 이상 감지: {supply_anomaly['direction']} (score={supply_anomaly['anomaly_score']:.3f})")
    
    # ★ 수급 이상을 매수 시그널에 반영 (추가)
    # 미보유 + TREND_UP + bullish 이상 → 최대 비율로 매수
    # 미보유 + bearish 이상 → 진입 보류
    
    # ... 기존 진입/청산 로직에서 아래처럼 활용 ...
    
    # elif regime == "TREND_UP":
    #     invest = state.cash * params.invest_ratio
    #     if dual_buy:
    #         invest = state.cash * params.max_invest_ratio
    #     ★ if supply_anomaly["is_anomaly"] and supply_anomaly["direction"] == "bullish":
    #         invest = state.cash * params.max_invest_ratio  # 이상 매수 감지 → 최대 비율
    #         signal = "BUY_TREND_ANOMALY"
    #     ★ if supply_anomaly["is_anomaly"] and supply_anomaly["direction"] == "bearish":
    #         signal = "NO_ENTRY_BEARISH_ANOMALY"  # 이상 매도 감지 → 진입 차단
    #         continue/return
"""


# ─────────────────────────────────────
# 4. Telegram 알림에 AI 정보 추가
# ─────────────────────────────────────
"""
    lines = [
        f"📊 <b>{TICKER_NAME} Morning</b>",
        f"시그널: <b>{signal}</b>",
        f"레짐: {regime}",
        # ★ AI 정보 추가
        f"HMM: {hmm_result.get('regime', 'N/A')} ({hmm_result.get('confidence', 0):.0%})",
        f"수급이상: {'🚨 ' + supply_anomaly['direction'] if supply_anomaly['is_anomaly'] else '정상'}",
        # ...
    ]
"""


# ─────────────────────────────────────
# 5. GitHub Actions 워크플로우 pip 추가
# ─────────────────────────────────────
"""
    - name: Install dependencies
      run: |
        pip install --upgrade pip
        pip install mojito2 pandas numpy requests
        pip install hmmlearn scikit-learn  # ★ 추가
"""


# ─────────────────────────────────────
# 6. 모델 학습 워크플로우 (선택)
# ─────────────────────────────────────
"""
# 모델은 Colab에서 학습 후 state/에 업로드하는 것을 권장.
# 또는 별도 워크플로우로 주 1회 재학습:
#
# python ai_layer.py --state-dir state
#
# 이 명령이 hmm_model_247540.pkl, if_model_247540.pkl 을 생성합니다.
"""
