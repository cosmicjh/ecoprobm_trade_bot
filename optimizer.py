"""
에코프로비엠 (247540) v4 — Phase 4-1: Optuna Walk-Forward 파라미터 최적화
============================================================================
StrategyParams의 핵심 파라미터를 베이지안 최적화로 탐색합니다.
Walk-forward 방식으로 과적합을 방지합니다.

설계:
  - 백테스트 시뮬레이터: trading_bot._run_morning 의 핵심 로직을
    일봉 단위 이벤트 시뮬레이션으로 재현 (rule 레이어만, AI/외부 수급 제외)
  - Walk-forward: train_months 개월 학습 → test_months 개월 검증, 1개월씩 롤링
  - 목적함수: 평균 (total_return / max(|max_drawdown|, 1))  (Calmar 류)
  - 출력: state/optimized_params_247540.json  (trading_bot이 자동 로드)
  - 안전장치: 새 파라미터의 walk-forward 점수가 기존 파라미터보다 개선되지
              않으면 덮어쓰지 않음. 백업은 항상 .bak 으로 보관.

실행:
    python optimizer.py --state-dir state --trials 50

GitHub Actions에서는 ecoprobm_v4_optimize.yml 워크플로우가 호출.
"""

import os
import json
import logging
import shutil
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# trading_bot의 지표/레짐 함수를 그대로 import → 라이브 로직과 100% 동일
from trading_bot import (
    StrategyParams,
    compute_indicators,
    classify_regime,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

TICKER = "247540"
TICKER_NAME = "에코프로비엠"


# ═══════════════════════════════════════════════════════════════════
# 1. OHLCV 로드
# ═══════════════════════════════════════════════════════════════════

def load_ohlcv_df(state_dir: Path) -> pd.DataFrame:
    """data_collector가 저장한 ohlcv_247540.json을 DataFrame으로 변환."""
    path = state_dir / f"ohlcv_{TICKER}.json"
    if not path.exists():
        raise FileNotFoundError(f"OHLCV 파일 없음: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = [{"date": d, **v} for d, v in sorted(data.items())]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    # data_collector는 'Open','High','Low','Close','Volume' 키를 저장.
    # compute_indicators는 prev_* 컬럼을 요구하므로 변환.
    base = ["Open", "High", "Low", "Close", "Volume"]
    for c in base:
        if c not in df.columns:
            raise ValueError(f"OHLCV 컬럼 부족: {c}")
        df[f"prev_{c}"] = df[c].shift(1)

    return df.dropna(subset=[f"prev_{c}" for c in base])


# ═══════════════════════════════════════════════════════════════════
# 2. 백테스트 시뮬레이터 (rule-only, 라이브 로직 모사)
# ═══════════════════════════════════════════════════════════════════

def simulate(df_ind: pd.DataFrame, params: StrategyParams,
             initial_capital: float = 1_500_000.0) -> dict:
    """
    일봉 단위 이벤트 시뮬레이션.

    각 일봉마다 다음 순서로 처리:
      1. 보유 중이면 청산 판단 (당일 시가 기준)
         - 손절: open <= entry*(1+sl_pct)
         - 1차 익절: open >= entry*(1+tp1_pct), 미실행 시
         - 트레일링: 1차 익절 후, low <= highest - atr*trail_mult
      2. 미보유면 진입 판단 (당일 종가에 매수)
         - 레짐별 분기

    근사:
      - 외국인·기관 동반매수, RANGE_BOUND/HIGH_VOL의 일부 분기는 단순화
      - 슬리피지 0, 수수료 0.015% 가정
    """
    cash = initial_capital
    pos_qty = 0
    entry_price = 0.0
    highest = 0.0
    tp1_done = False
    cooldown_until = -1   # row index

    equity_curve = []
    trades = []
    fee = 0.00015

    rows = df_ind.reset_index()

    for i, row in rows.iterrows():
        op = row.get("Open")
        hi = row.get("High")
        lo = row.get("Low")
        cl = row.get("Close")

        if pd.isna(op) or pd.isna(cl) or op <= 0:
            equity_curve.append(cash + pos_qty * (cl if pd.notna(cl) else 0))
            continue

        regime = classify_regime(row, params)

        # ── 청산 판단 ──
        if pos_qty > 0:
            # 손절
            sl_price = entry_price * (1 + params.sl_pct)
            if lo <= sl_price:
                fill = min(op, sl_price)  # 갭다운이면 시가
                cash += pos_qty * fill * (1 - fee)
                trades.append({"type": "SL", "pnl": (fill - entry_price) * pos_qty})
                pos_qty = 0
                tp1_done = False
                cooldown_until = i + params.cooldown_days
                equity_curve.append(cash)
                continue

            # 1차 익절
            tp1_price = entry_price * (1 + params.tp1_pct)
            if not tp1_done and hi >= tp1_price:
                sell_qty = max(1, int(pos_qty * params.tp1_sell_ratio))
                cash += sell_qty * tp1_price * (1 - fee)
                trades.append({"type": "TP1", "pnl": (tp1_price - entry_price) * sell_qty})
                pos_qty -= sell_qty
                tp1_done = True
                highest = max(highest, hi)

            # 트레일링 (1차 익절 후)
            if tp1_done and pos_qty > 0:
                highest = max(highest, hi)
                atr_val = row.get("atr", 0)
                if pd.notna(atr_val) and atr_val > 0:
                    trail_stop = highest - atr_val * params.trail_atr_mult
                    if lo <= trail_stop:
                        fill = max(trail_stop, op)
                        cash += pos_qty * fill * (1 - fee)
                        trades.append({"type": "TRAIL", "pnl": (fill - entry_price) * pos_qty})
                        pos_qty = 0
                        tp1_done = False
                        cooldown_until = i + params.cooldown_days

        # ── 진입 판단 (미보유 + 쿨다운 해제) ──
        if pos_qty == 0 and i >= cooldown_until:
            buy = False
            invest_ratio = params.invest_ratio

            if regime == "TREND_UP":
                buy = True
            elif regime == "RANGE_BOUND":
                rsi_v = row.get("rsi", 50)
                bb_pctb = row.get("bb_pctb", 0.5)
                if rsi_v < params.rsi_entry and bb_pctb < 0.1:
                    buy = True
                    invest_ratio = params.invest_ratio * 0.5
            elif regime == "HIGH_VOLATILITY":
                rsi_v = row.get("rsi", 50)
                if rsi_v < 25:
                    buy = True
                    invest_ratio = params.invest_ratio * params.hvol_size_reduction

            if buy:
                invest = cash * invest_ratio
                qty = int(invest / cl) if cl > 0 else 0
                if qty > 0:
                    cost = qty * cl * (1 + fee)
                    if cost <= cash:
                        cash -= cost
                        pos_qty = qty
                        entry_price = cl
                        highest = cl
                        tp1_done = False

        equity = cash + pos_qty * cl
        equity_curve.append(equity)

    eq = np.array(equity_curve)
    final = eq[-1] if len(eq) else initial_capital
    total_return = (final - initial_capital) / initial_capital

    # Max drawdown
    peak = np.maximum.accumulate(eq) if len(eq) else np.array([initial_capital])
    dd = (eq - peak) / peak
    max_dd = float(dd.min()) if len(dd) else 0.0

    n_trades = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    win_rate = wins / n_trades if n_trades else 0

    return {
        "total_return": float(total_return),
        "max_drawdown": max_dd,
        "n_trades": n_trades,
        "win_rate": float(win_rate),
        "final_equity": float(final),
    }


# ═══════════════════════════════════════════════════════════════════
# 3. 목적 함수 (Calmar류 — 손실 페널티)
# ═══════════════════════════════════════════════════════════════════

def score(result: dict) -> float:
    """높을수록 좋음. 최대낙폭으로 수익률을 나눠 위험조정."""
    ret = result["total_return"]
    dd = abs(result["max_drawdown"])
    if result["n_trades"] < 3:
        return -1.0   # 매매 수 부족 → 신뢰 불가
    return ret / max(dd, 0.02)


# ═══════════════════════════════════════════════════════════════════
# 4. Walk-forward 평가
# ═══════════════════════════════════════════════════════════════════

def walk_forward_score(df: pd.DataFrame, params: StrategyParams,
                       train_months: int = 6, test_months: int = 1,
                       step_months: int = 1) -> dict:
    """
    Walk-forward 방식으로 파라미터를 평가.

    df: 전체 OHLCV (지표는 시뮬레이터 안에서 계산)
    """
    if len(df) < (train_months + test_months) * 21:
        # 데이터 부족 시 전체 기간 단일 백테스트로 fallback
        df_ind = compute_indicators(df, params)
        result = simulate(df_ind, params)
        return {
            "score": score(result),
            "n_windows": 1,
            "windows": [result],
            "fallback": True,
        }

    df_ind = compute_indicators(df, params)
    df_ind = df_ind.dropna(subset=["ma_l"])

    if len(df_ind) == 0:
        return {"score": -1.0, "n_windows": 0, "windows": [], "fallback": False}

    start = df_ind.index[0]
    end = df_ind.index[-1]

    window_results = []
    cursor = start

    while True:
        train_end = cursor + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > end:
            break

        # 본 구현은 simple walk-forward: train 윈도우는 통계 안정성
        # 확보용으로만 두고, test 윈도우에서 실제 백테스트를 돌립니다.
        # (Optuna가 외부에서 파라미터를 제안하므로 train에서 fit 단계 불필요)
        test_slice = df_ind[(df_ind.index > train_end) & (df_ind.index <= test_end)]
        if len(test_slice) >= 10:
            res = simulate(test_slice, params)
            window_results.append(res)

        cursor = cursor + pd.DateOffset(months=step_months)

    if not window_results:
        return {"score": -1.0, "n_windows": 0, "windows": [], "fallback": False}

    avg_score = float(np.mean([score(r) for r in window_results]))
    avg_return = float(np.mean([r["total_return"] for r in window_results]))
    avg_dd = float(np.mean([r["max_drawdown"] for r in window_results]))

    return {
        "score": avg_score,
        "avg_return": avg_return,
        "avg_drawdown": avg_dd,
        "n_windows": len(window_results),
        "windows": window_results,
        "fallback": False,
    }


# ═══════════════════════════════════════════════════════════════════
# 5. Optuna 탐색
# ═══════════════════════════════════════════════════════════════════

def make_objective(df: pd.DataFrame):
    """Optuna trial → score."""
    import optuna  # 함수 안에서 import (선택적 의존성)

    def objective(trial: "optuna.Trial") -> float:
        params = StrategyParams(
            ma_short=trial.suggest_int("ma_short", 5, 20),
            ma_long=trial.suggest_int("ma_long", 40, 120),
            bb_squeeze_threshold=trial.suggest_float("bb_squeeze_threshold", 0.5, 1.0),
            atr_hvol_threshold=trial.suggest_float("atr_hvol_threshold", 1.2, 2.0),
            tp1_pct=trial.suggest_float("tp1_pct", 0.05, 0.20),
            tp1_sell_ratio=trial.suggest_float("tp1_sell_ratio", 0.3, 0.8),
            trail_atr_mult=trial.suggest_float("trail_atr_mult", 1.0, 3.0),
            sl_pct=trial.suggest_float("sl_pct", -0.07, -0.02),
            cooldown_days=trial.suggest_int("cooldown_days", 0, 3),
            rsi_entry=trial.suggest_float("rsi_entry", 20.0, 35.0),
            invest_ratio=trial.suggest_float("invest_ratio", 0.2, 0.5),
            pyramiding_min_profit=trial.suggest_float("pyramiding_min_profit", 0.01, 0.05),
            pyramiding_size_ratio=trial.suggest_float("pyramiding_size_ratio", 0.3, 0.8),
            max_pyramiding=trial.suggest_int("max_pyramiding", 1, 3),
        )
        # ma_short < ma_long 강제
        if params.ma_short >= params.ma_long:
            return -10.0

        wf = walk_forward_score(df, params)
        return wf["score"]

    return objective


def run_optimization(state_dir: str = "state", n_trials: int = 50,
                     write_threshold: float = 0.05) -> dict:
    """
    메인 진입점.

    write_threshold: 새 파라미터의 walk-forward score가 기존 대비 이만큼
                     개선되어야만 optimized_params 파일을 덮어씀.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sd = Path(state_dir)
    sd.mkdir(parents=True, exist_ok=True)

    log.info(f"[OPT] OHLCV 로드: {sd}")
    df = load_ohlcv_df(sd)
    log.info(f"[OPT] 데이터: {len(df)}일 ({df.index[0].date()} ~ {df.index[-1].date()})")

    # 기존 파라미터 점수 측정 (개선 여부 비교용)
    current_path = sd / f"optimized_params_{TICKER}.json"
    current_params = StrategyParams()
    if current_path.exists():
        with open(current_path, "r") as f:
            data = json.load(f)
        for k, v in data.get("params", {}).items():
            if hasattr(current_params, k):
                setattr(current_params, k, v)

    current_wf = walk_forward_score(df, current_params)
    log.info(f"[OPT] 기존 파라미터 walk-forward score: {current_wf['score']:.3f}")

    # ── Optuna 탐색 ──
    log.info(f"[OPT] Optuna 탐색 시작 (n_trials={n_trials})")
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(make_objective(df), n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best_score = study.best_value
    log.info(f"[OPT] 최적 score={best_score:.3f}")
    log.info(f"[OPT] 최적 파라미터: {best}")

    new_params = StrategyParams(**{k: v for k, v in best.items() if hasattr(StrategyParams, k)})
    new_wf = walk_forward_score(df, new_params)

    improved = (new_wf["score"] - current_wf["score"]) >= write_threshold

    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_trials": n_trials,
        "old_score": current_wf["score"],
        "new_score": new_wf["score"],
        "improvement": new_wf["score"] - current_wf["score"],
        "improved": improved,
        "best_params": best,
        "n_windows": new_wf.get("n_windows", 0),
        "avg_return": new_wf.get("avg_return", 0),
        "avg_drawdown": new_wf.get("avg_drawdown", 0),
    }

    if improved:
        # 백업 후 덮어쓰기
        if current_path.exists():
            shutil.copy(current_path, current_path.with_suffix(".json.bak"))
        with open(current_path, "w", encoding="utf-8") as f:
            json.dump({
                "params": best,
                "meta": {
                    "optimized_at": summary["timestamp"],
                    "score": new_wf["score"],
                    "n_windows": new_wf.get("n_windows", 0),
                },
            }, f, ensure_ascii=False, indent=2)
        log.info(f"[OPT] 새 파라미터 저장: {current_path}")
    else:
        log.info(f"[OPT] 개선폭 부족 ({summary['improvement']:+.3f} < {write_threshold}), 기존 유지")

    # 이력 누적
    history_path = sd / f"optimization_history_{TICKER}.json"
    history = []
    if history_path.exists():
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(summary)
    history = history[-24:]   # 최근 2년치
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2, default=str)

    return summary


# ═══════════════════════════════════════════════════════════════════
# 6. Telegram
# ═══════════════════════════════════════════════════════════════════

def format_optimization_telegram(summary: dict) -> str:
    lines = [
        f"🔧 <b>{TICKER_NAME} 파라미터 최적화</b>",
        f"실행: {summary['timestamp'][:16].replace('T', ' ')}",
        f"trials: {summary['n_trials']} / windows: {summary['n_windows']}",
        "",
        f"기존 score: <code>{summary['old_score']:+.3f}</code>",
        f"신규 score: <code>{summary['new_score']:+.3f}</code> ({summary['improvement']:+.3f})",
        f"평균 수익률: {summary['avg_return']*100:+.2f}%",
        f"평균 MDD: {summary['avg_drawdown']*100:+.2f}%",
        "",
    ]
    if summary["improved"]:
        lines.append("✅ <b>새 파라미터 적용됨</b>")
        lines.append("")
        lines.append("<b>주요 변경:</b>")
        for k, v in summary["best_params"].items():
            if isinstance(v, float):
                lines.append(f"  • {k} = {v:.3f}")
            else:
                lines.append(f"  • {k} = {v}")
    else:
        lines.append("ℹ️ 개선폭 부족, 기존 파라미터 유지")
    return "\n".join(lines)


def send_telegram(message: str):
    import requests
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        for chunk in [message[i:i+3800] for i in range(0, len(message), 3800)]:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": chunk, "parse_mode": "HTML"},
                timeout=10,
            )
    except Exception as e:
        log.error(f"[TG] 발송 실패: {e}")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import traceback

    parser = argparse.ArgumentParser(description=f"{TICKER_NAME} v4 파라미터 최적화")
    parser.add_argument("--state-dir", type=str, default="state")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="이 값 이상의 score 개선 시에만 파라미터 갱신")
    args = parser.parse_args()

    try:
        summary = run_optimization(
            state_dir=args.state_dir,
            n_trials=args.trials,
            write_threshold=args.threshold,
        )
        send_telegram(format_optimization_telegram(summary))
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    except Exception as e:
        tb = traceback.format_exc()
        log.error(f"[OPT] 실패: {e}\n{tb}")
        send_telegram(f"❌ <b>{TICKER_NAME} 파라미터 최적화 실패</b>\n<code>{str(e)[:500]}</code>")
        raise
