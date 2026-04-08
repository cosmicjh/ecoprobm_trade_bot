"""
에코프로비엠 (247540) v4 — Phase 1-2: 기술적 지표 계산 모듈
==============================================================
Phase 1-1에서 수집한 OHLCV 데이터를 기반으로 기술적 지표를 계산합니다.

핵심 원칙:
  ★ 모든 지표는 prev_* 컬럼(전일 확정 데이터)에서만 계산
  ★ 당일 데이터(Open, High, Low, Close)는 시그널 계산에 절대 사용하지 않음
  ★ 이것이 .shift(1)을 적용한 이유 — look-ahead bias 완전 방지

계산 지표:
  Layer 1 (레짐 판별):
    - MA(20), MA(60): 이동평균선
    - BB(20,2): 볼린저밴드 (상단/중심/하단/밴드폭)
    - ATR(14): Average True Range
    - KOSDAQ 추세: (별도 지수 데이터 필요, 여기서는 종목 자체 추세로 대체)

  Layer 2 (수급 시그널 보조):
    - RSI(14): 과매수/과매도
    - 거래량 MA(20) & 비율: 거래량 이상 감지

  레짐 분류 (룰 기반 v1):
    - TREND_UP / TREND_DOWN / RANGE_BOUND / HIGH_VOLATILITY
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# 1. 개별 기술적 지표 계산 함수
# ═══════════════════════════════════════════════════════════════════

def calc_ma(series: pd.Series, period: int) -> pd.Series:
    """단순 이동평균 (SMA)."""
    return series.rolling(window=period, min_periods=period).mean()


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    """지수 이동평균 (EMA)."""
    return series.ewm(span=period, adjust=False).mean()


def calc_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """
    볼린저밴드 계산.

    Returns:
        DataFrame with columns:
            bb_middle: 중심선 (SMA)
            bb_upper:  상단밴드
            bb_lower:  하단밴드
            bb_width:  밴드폭 (상단-하단)/중심 × 100
            bb_pctb:   %B (현재가의 밴드 내 위치, 0=하단, 1=상단)
    """
    middle = calc_ma(series, period)
    std = series.rolling(window=period, min_periods=period).std()

    upper = middle + (std * num_std)
    lower = middle - (std * num_std)

    width = np.where(middle > 0, (upper - lower) / middle * 100, 0.0)
    pctb = np.where(
        (upper - lower) > 0,
        (series - lower) / (upper - lower),
        0.5,
    )

    return pd.DataFrame({
        "bb_middle": middle,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_width": width,
        "bb_pctb": pctb,
    }, index=series.index)


def calc_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average True Range (ATR).
    True Range = max(H-L, |H-prev_C|, |L-prev_C|)
    ATR = TR의 EMA(period)
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=period, adjust=False).mean()

    return atr


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI (Relative Strength Index).
    Wilder의 smoothing 방식 사용.
    """
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return pd.Series(rsi, index=series.index)


def calc_volume_indicators(volume: pd.Series, period: int = 20) -> pd.DataFrame:
    """
    거래량 지표.
    - vol_ma: 거래량 이동평균
    - vol_ratio: 당일 거래량 / MA (1.0 = 평균 수준)
    - vol_surge: 거래량이 평균의 2배 이상이면 True
    """
    vol_ma = calc_ma(volume, period)
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    vol_surge = vol_ratio >= 2.0

    return pd.DataFrame({
        "vol_ma20": vol_ma,
        "vol_ratio": np.round(vol_ratio, 3),
        "vol_surge": vol_surge.astype(int),
    }, index=volume.index)


# ═══════════════════════════════════════════════════════════════════
# 2. 통합 지표 계산 (prev_* 컬럼 전용)
# ═══════════════════════════════════════════════════════════════════

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    OHLCV DataFrame에 모든 기술적 지표를 추가합니다.

    ★ 핵심: prev_Close, prev_High, prev_Low, prev_Volume에서만 계산
      → 전략 시그널이 '어제까지 확정된 정보'만으로 판단하도록 보장

    Args:
        df: Phase 1-1에서 생성한 OHLCV DataFrame.
            필수 컬럼: prev_Open, prev_High, prev_Low, prev_Close, prev_Volume

    Returns:
        지표가 추가된 DataFrame (원본 컬럼 + 지표 컬럼).
        NaN 행은 제거하지 않음 (호출자가 판단).
    """
    required = ["prev_Open", "prev_High", "prev_Low", "prev_Close", "prev_Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    result = df.copy()

    # ── 이동평균선 ──
    result["ma20"] = calc_ma(result["prev_Close"], 20)
    result["ma60"] = calc_ma(result["prev_Close"], 60)

    # MA 위치 관계
    result["price_above_ma20"] = (result["prev_Close"] > result["ma20"]).astype(int)
    result["ma20_above_ma60"] = (result["ma20"] > result["ma60"]).astype(int)

    # ── 볼린저밴드 ──
    bb = calc_bollinger_bands(result["prev_Close"], period=20, num_std=2.0)
    result = pd.concat([result, bb], axis=1)

    # BB 폭의 이동평균 (스퀴즈 감지용)
    result["bb_width_ma20"] = calc_ma(result["bb_width"], 20)
    result["bb_squeeze"] = np.where(
        result["bb_width_ma20"] > 0,
        (result["bb_width"] / result["bb_width_ma20"]).round(3),
        1.0,
    )

    # ── ATR ──
    result["atr14"] = calc_atr(
        result["prev_High"], result["prev_Low"], result["prev_Close"], period=14
    )

    # ATR 비율 (현재 ATR / 20일 평균 ATR — 변동성 급등 감지)
    result["atr_ma20"] = calc_ma(result["atr14"], 20)
    result["atr_ratio"] = np.where(
        result["atr_ma20"] > 0,
        (result["atr14"] / result["atr_ma20"]).round(3),
        1.0,
    )

    # ── RSI ──
    result["rsi14"] = calc_rsi(result["prev_Close"], period=14)

    # ── 거래량 지표 ──
    vol_ind = calc_volume_indicators(result["prev_Volume"], period=20)
    result = pd.concat([result, vol_ind], axis=1)

    # ── 일간 수익률 ──
    result["daily_return"] = result["prev_Close"].pct_change()

    # ── 변동폭 비율 (당일 range / close) ──
    result["range_pct"] = np.where(
        result["prev_Close"] > 0,
        ((result["prev_High"] - result["prev_Low"]) / result["prev_Close"] * 100).round(2),
        0.0,
    )

    log.info(f"[INDICATORS] {len(result)}행, {len(result.columns)}컬럼 지표 계산 완료")
    return result


# ═══════════════════════════════════════════════════════════════════
# 3. 레짐 분류기 (룰 기반 v1)
# ═══════════════════════════════════════════════════════════════════

# 레짐 파라미터
REGIME_PARAMS = {
    "BB_SQUEEZE_THRESHOLD": 0.7,   # bb_width < 평균의 70% → 횡보
    "ATR_HVOL_THRESHOLD": 1.5,     # atr > 평균의 1.5배 → 고변동성
}


def classify_regime(row: pd.Series) -> str:
    """
    단일 행의 지표 값으로 시장 레짐을 분류합니다.

    레짐 우선순위:
      1. HIGH_VOLATILITY: ATR이 비정상적으로 높음 → 최우선 감지
      2. RANGE_BOUND: 볼린저밴드 스퀴즈 → 방향성 없음
      3. TREND_UP: 가격 > MA20 > MA60 → 상승 추세
      4. TREND_DOWN: 가격 < MA20 < MA60 → 하락 추세
      5. NEUTRAL: 위 조건에 해당 없음 (MA 혼조)

    Args:
        row: compute_all_indicators()의 결과 DataFrame 한 행

    Returns:
        "TREND_UP" | "TREND_DOWN" | "RANGE_BOUND" | "HIGH_VOLATILITY" | "NEUTRAL"
    """
    # 필요한 값이 NaN이면 판별 불가
    if pd.isna(row.get("ma20")) or pd.isna(row.get("ma60")):
        return "UNKNOWN"

    atr_ratio = row.get("atr_ratio", 1.0)
    bb_squeeze = row.get("bb_squeeze", 1.0)
    price_above_ma20 = row.get("price_above_ma20", 0)
    ma20_above_ma60 = row.get("ma20_above_ma60", 0)

    # 1) 고변동성 체크
    if atr_ratio >= REGIME_PARAMS["ATR_HVOL_THRESHOLD"]:
        return "HIGH_VOLATILITY"

    # 2) 횡보 (볼린저밴드 스퀴즈) 체크
    if bb_squeeze <= REGIME_PARAMS["BB_SQUEEZE_THRESHOLD"]:
        return "RANGE_BOUND"

    # 3) 추세 판별
    if price_above_ma20 == 1 and ma20_above_ma60 == 1:
        return "TREND_UP"
    elif price_above_ma20 == 0 and ma20_above_ma60 == 0:
        return "TREND_DOWN"

    # 4) 혼조 (MA 크로스 진행 중 등)
    return "NEUTRAL"


def add_regime_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame에 레짐 컬럼을 추가합니다.
    또한 레짐 전환 시점을 표시합니다.
    """
    result = df.copy()
    result["regime"] = result.apply(classify_regime, axis=1)

    # 레짐 전환 감지
    result["regime_changed"] = (result["regime"] != result["regime"].shift(1)).astype(int)

    # 연속 레짐 일수
    regime_groups = (result["regime"] != result["regime"].shift(1)).cumsum()
    result["regime_days"] = result.groupby(regime_groups).cumcount() + 1

    log.info(f"[REGIME] 레짐 분포:")
    if "regime" in result.columns:
        counts = result["regime"].value_counts()
        for regime, count in counts.items():
            pct = count / len(result) * 100
            log.info(f"  {regime}: {count}일 ({pct:.1f}%)")

    return result


# ═══════════════════════════════════════════════════════════════════
# 4. 레짐별 후속 수익률 분석 (백테스트 검증용)
# ═══════════════════════════════════════════════════════════════════

def analyze_regime_returns(df: pd.DataFrame, forward_days: list = [1, 3, 5, 10]) -> dict:
    """
    각 레짐 진입 후 N일간의 평균 수익률을 계산합니다.
    레짐 분류의 유효성을 검증하는 용도.

    좋은 레짐 분류기라면:
      - TREND_UP 진입 후 수익률 > 0
      - TREND_DOWN 진입 후 수익률 < 0
      - HIGH_VOLATILITY는 수익률 분산이 큼
    """
    results = {}

    for days in forward_days:
        col = f"fwd_return_{days}d"
        df[col] = df["prev_Close"].pct_change(days).shift(-days) * 100

    for regime in df["regime"].unique():
        if regime == "UNKNOWN":
            continue

        mask = df["regime"] == regime
        regime_data = df[mask]

        regime_result = {
            "count": int(mask.sum()),
            "pct": round(mask.sum() / len(df) * 100, 1),
        }

        for days in forward_days:
            col = f"fwd_return_{days}d"
            returns = regime_data[col].dropna()
            if len(returns) > 0:
                regime_result[f"{days}d_mean"] = round(returns.mean(), 2)
                regime_result[f"{days}d_std"] = round(returns.std(), 2)
                regime_result[f"{days}d_win_rate"] = round((returns > 0).mean() * 100, 1)

        results[regime] = regime_result

    # 임시 컬럼 제거
    for days in forward_days:
        col = f"fwd_return_{days}d"
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return results


# ═══════════════════════════════════════════════════════════════════
# 5. 수급 시그널 계산 (supply_data 기반)
# ═══════════════════════════════════════════════════════════════════

def compute_supply_signals(supply_data: dict, lookback: int = 5) -> pd.DataFrame:
    """
    supply_data_247540.json에서 수급 시그널을 계산합니다.

    시그널:
      - foreign_ma5: 외국인 순매수 5일 이동평균
      - inst_ma5: 기관 순매수 5일 이동평균
      - supply_score: 외국인+기관 동반 매수 강도 (5일 평균 대비 배수)
      - short_trend: 공매도 비중 5일 추세 (감소=-1, 횡보=0, 증가=1)
    """
    if not supply_data:
        return pd.DataFrame()

    records = []
    for date_str, data in sorted(supply_data.items()):
        record = {"date": date_str}
        record.update(data)
        records.append(record)

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    # 외국인/기관 순매수 이동평균
    if "foreign_net_qty" in df.columns:
        df["foreign_ma5"] = calc_ma(df["foreign_net_qty"].astype(float), lookback)
    if "inst_net_qty" in df.columns:
        df["inst_ma5"] = calc_ma(df["inst_net_qty"].astype(float), lookback)

    # 동반 매수 강도 (외국인+기관 순매수 합계 / 5일 평균의 절대값)
    if "foreign_net_qty" in df.columns and "inst_net_qty" in df.columns:
        combined = df["foreign_net_qty"].astype(float) + df["inst_net_qty"].astype(float)
        combined_ma = calc_ma(combined.abs(), lookback)
        df["supply_score"] = np.where(
            combined_ma > 0,
            (combined / combined_ma).round(2),
            0.0,
        )

        # 외국인·기관 동반 매수 여부
        df["dual_buy"] = (
            (df["foreign_net_qty"].astype(float) > 0) &
            (df["inst_net_qty"].astype(float) > 0)
        ).astype(int)

    # 공매도 비중 추세
    if "short_ratio_vol" in df.columns:
        sr = df["short_ratio_vol"].astype(float)
        sr_ma5 = calc_ma(sr, lookback)
        df["short_trend"] = np.where(
            sr < sr_ma5 * 0.9, -1,       # 감소
            np.where(sr > sr_ma5 * 1.1, 1, 0)  # 증가 / 횡보
        )
        # 공매도 5일 연속 감소 (숏커버링 시그널)
        df["short_declining"] = (
            sr.diff().rolling(lookback).apply(lambda x: (x < 0).all(), raw=True)
        ).fillna(0).astype(int)

    log.info(f"[SUPPLY_SIGNAL] {len(df)}일치 수급 시그널 계산 완료")
    return df


# ═══════════════════════════════════════════════════════════════════
# 6. 메인: OHLCV 파일에서 지표 계산 → 저장
# ═══════════════════════════════════════════════════════════════════

def run_indicator_pipeline(
    ohlcv_path: str = "ohlcv_247540.json",
    supply_path: str = "supply_data_247540.json",
    output_path: str = "indicators_247540.json",
) -> dict:
    """
    Phase 1-2 메인 파이프라인.
    1. OHLCV 로드 → 지표 계산 → 레짐 분류
    2. 수급 데이터 로드 → 수급 시그널 계산
    3. 결과 저장 (JSON)
    """
    log.info(f"{'='*60}")
    log.info("[PHASE 1-2] 기술적 지표 계산 시작")
    log.info(f"{'='*60}")

    # ── 1) OHLCV 로드 ──
    ohlcv_file = Path(ohlcv_path)
    if not ohlcv_file.exists():
        raise FileNotFoundError(f"OHLCV 파일 없음: {ohlcv_file}")

    with open(ohlcv_file, "r") as f:
        ohlcv_data = json.load(f)

    rows = []
    for date_str, vals in sorted(ohlcv_data.items()):
        row = {"date": date_str}
        row.update(vals)
        rows.append(row)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    log.info(f"[OHLCV] {len(df)}일 로드 ({df.index[0]} ~ {df.index[-1]})")

    # ── 2) 지표 계산 ──
    df = compute_all_indicators(df)

    # ── 3) 레짐 분류 ──
    df = add_regime_column(df)

    # ── 4) 레짐 유효성 분석 ──
    regime_analysis = analyze_regime_returns(df)
    log.info("[REGIME] 레짐별 후속 수익률:")
    for regime, stats in regime_analysis.items():
        mean_1d = stats.get("1d_mean", "N/A")
        mean_5d = stats.get("5d_mean", "N/A")
        win_5d = stats.get("5d_win_rate", "N/A")
        log.info(f"  {regime}: 1d={mean_1d}%, 5d={mean_5d}%, 5d승률={win_5d}%")

    # ── 5) 수급 시그널 ──
    supply_file = Path(supply_path)
    supply_signals = pd.DataFrame()
    if supply_file.exists():
        with open(supply_file, "r") as f:
            supply_data = json.load(f)
        supply_signals = compute_supply_signals(supply_data)

    # ── 6) 결과 저장 ──
    # 지표 DataFrame → JSON
    indicator_data = {}
    for idx, row in df.iterrows():
        date_key = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
        indicator_data[date_key] = {}
        for col in row.index:
            val = row[col]
            if pd.isna(val):
                indicator_data[date_key][col] = None
            elif isinstance(val, (np.integer, int)):
                indicator_data[date_key][col] = int(val)
            elif isinstance(val, (np.floating, float)):
                indicator_data[date_key][col] = round(float(val), 4)
            else:
                indicator_data[date_key][col] = str(val)

    output_file = Path(output_path)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(indicator_data, f, ensure_ascii=False, indent=2)
    log.info(f"[OUTPUT] {len(indicator_data)}일치 지표 저장: {output_file}")

    # ── 7) 요약 리포트 ──
    summary = {
        "total_days": len(df),
        "date_range": f"{df.index[0]} ~ {df.index[-1]}",
        "indicator_count": len([c for c in df.columns if c not in [
            "Open", "High", "Low", "Close", "Volume",
            "prev_Open", "prev_High", "prev_Low", "prev_Close", "prev_Volume",
        ]]),
        "regime_distribution": {
            regime: int(count) for regime, count in df["regime"].value_counts().items()
        },
        "regime_analysis": regime_analysis,
        "supply_signal_days": len(supply_signals) if not supply_signals.empty else 0,
        "latest_regime": str(df["regime"].iloc[-1]) if len(df) > 0 else "N/A",
        "latest_rsi": round(float(df["rsi14"].iloc[-1]), 1) if len(df) > 0 and pd.notna(df["rsi14"].iloc[-1]) else None,
        "latest_atr_ratio": round(float(df["atr_ratio"].iloc[-1]), 2) if len(df) > 0 and pd.notna(df["atr_ratio"].iloc[-1]) else None,
        "latest_bb_squeeze": round(float(df["bb_squeeze"].iloc[-1]), 2) if len(df) > 0 and pd.notna(df["bb_squeeze"].iloc[-1]) else None,
    }

    log.info(f"[SUMMARY] 최신 레짐: {summary['latest_regime']}")
    log.info(f"[SUMMARY] RSI: {summary['latest_rsi']}, ATR비율: {summary['latest_atr_ratio']}, BB스퀴즈: {summary['latest_bb_squeeze']}")

    return summary


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Phase 1-2: 기술적 지표 계산")
    parser.add_argument("--state-dir", type=str, default=".")
    args = parser.parse_args()

    sd = Path(args.state_dir)
    summary = run_indicator_pipeline(
        ohlcv_path=str(sd / "ohlcv_247540.json"),
        supply_path=str(sd / "supply_data_247540.json"),
        output_path=str(sd / "indicators_247540.json"),
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
