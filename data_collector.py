"""
에코프로비엠 (247540) v4 자동트레이딩 — Phase 1-1: 데이터 수집 파이프라인 (v2)
================================================================================
변경 이력:
  v1 → v2: 시장 데이터 수집을 한투 API → pykrx (KRX 공식 데이터)로 전면 전환
           한투 API(mojito2)는 주문 실행 전용으로 분리

수집 소스:
  ┌─────────────────────┬──────────────┬─────────────────────────────────┐
  │ 데이터               │ 소스          │ pykrx 함수                       │
  ├─────────────────────┼──────────────┼─────────────────────────────────┤
  │ 일봉 OHLCV          │ pykrx (KRX)  │ get_market_ohlcv_by_date        │
  │ 투자자별 매매동향     │ pykrx (KRX)  │ get_market_trading_value_by_date │
  │ 공매도 거래량/잔고    │ pykrx (KRX)  │ get_shorting_volume_by_date     │
  │ KOSDAQ 지수          │ pykrx (KRX)  │ get_index_ohlcv                 │
  │ ETF 프록시           │ pykrx (KRX)  │ get_market_ohlcv_by_date        │
  └─────────────────────┴──────────────┴─────────────────────────────────┘

왜 pykrx로 전환하는가:
  1. KRX 공식 데이터 — 가장 정확하고 권위 있는 원천
  2. 한투 API로는 공매도 데이터 수집 불가 확인
  3. 투자자별 매매동향도 pykrx가 더 안정적 (API 키/인증 불필요)
  4. 단일 라이브러리로 OHLCV + 수급 + 공매도 모두 커버
  5. 한투 API는 주문 실행(Phase 2)에만 집중 → 역할 분리 명확

주의사항:
  - pykrx 투자자별 매매내역은 당일 오후 6시 이후에 확정됨
  - pykrx는 KRX 웹을 스크래핑하므로, KRX 점검 시 일시 불가
  - FDR 데이터와 동일하게 .shift(1) 적용 필수 (look-ahead bias 방지)
  - pykrx 날짜 포맷: "YYYYMMDD" (하이픈 없음)
"""

import os
import json
import logging
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from time import sleep

import pandas as pd
import numpy as np

# ── pykrx (KRX 공식 데이터) ──────────────────────────────────
try:
    from pykrx import stock as krx
    PYKRX_AVAILABLE = True
except ImportError:
    krx = None
    PYKRX_AVAILABLE = False
    logging.warning("pykrx not installed. Run: pip install pykrx")

# ── FDR (pykrx 실패 시 fallback 전용) ────────────────────────
try:
    import FinanceDataReader as fdr
    FDR_AVAILABLE = True
except ImportError:
    fdr = None
    FDR_AVAILABLE = False

# ── 설정 ─────────────────────────────────────────────────────────
TICKER = "247540"
TICKER_NAME = "에코프로비엠"
KOSDAQ_INDEX_TICKER = "2001"  # pykrx KOSDAQ 지수 티커 (KRX 코드)
ETF_TICKER = "305720"         # KODEX 2차전지산업

# 데이터 수집 기간 (백테스트용)
HIST_START = "20220101"  # pykrx 포맷: YYYYMMDD

# pykrx 요청 간 딜레이 (KRX 서버 부하 방지)
PYKRX_DELAY = 1.0  # 초

# 파일 경로
STATE_DIR = Path(os.getenv("STATE_DIR", "."))
OHLCV_FILE = STATE_DIR / "ohlcv_247540.json"
SUPPLY_FILE = STATE_DIR / "supply_data_247540.json"
PIPELINE_STATE_FILE = STATE_DIR / "pipeline_state_247540.json"

# 로깅
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 1. OHLCV 데이터 수집 (pykrx → FDR fallback)
# ═══════════════════════════════════════════════════════════════════

def fetch_ohlcv(
    ticker: str = TICKER,
    start: str = HIST_START,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    일봉 OHLCV 수집. pykrx 우선, 실패 시 FDR fallback.

    pykrx 함수: stock.get_market_ohlcv_by_date(start, end, ticker)
    반환 컬럼: 시가, 고가, 저가, 종가, 거래량

    .shift(1) 적용하여 prev_* 컬럼 생성 (전일 확정 데이터).
    """
    if end is None:
        end = datetime.now().strftime("%Y%m%d")

    # start도 YYYYMMDD 포맷으로 통일
    start = start.replace("-", "")
    end = end.replace("-", "")

    log.info(f"[OHLCV] {ticker} 데이터 수집: {start} ~ {end}")

    df = pd.DataFrame()

    # ── pykrx 시도 ──
    if PYKRX_AVAILABLE:
        try:
            df = krx.get_market_ohlcv_by_date(start, end, ticker)
            if not df.empty:
                # pykrx 컬럼명 통일 (한글 → 영문)
                df = df.rename(columns={
                    "시가": "Open", "고가": "High", "저가": "Low",
                    "종가": "Close", "거래량": "Volume",
                })
                # 불필요 컬럼 제거 (등락률, 거래대금 등이 있을 수 있음)
                keep_cols = ["Open", "High", "Low", "Close", "Volume"]
                df = df[[c for c in keep_cols if c in df.columns]]
                log.info(f"[OHLCV] pykrx 수집 성공: {len(df)}일")
        except Exception as e:
            log.warning(f"[OHLCV] pykrx 실패: {e}")
            df = pd.DataFrame()

    # ── FDR fallback ──
    if df.empty and FDR_AVAILABLE:
        try:
            fdr_start = f"{start[:4]}-{start[4:6]}-{start[6:]}"
            fdr_end = f"{end[:4]}-{end[4:6]}-{end[6:]}"
            df = fdr.DataReader(ticker, fdr_start, fdr_end)
            if not df.empty:
                log.info(f"[OHLCV] FDR fallback 성공: {len(df)}일")
        except Exception as e:
            log.warning(f"[OHLCV] FDR fallback 실패: {e}")

    if df.empty:
        raise ValueError(f"[OHLCV] {ticker}: pykrx, FDR 모두 실패")

    # ── .shift(1): look-ahead bias 방지 ──
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[f"prev_{col}"] = df[col].shift(1)

    df = df.iloc[1:].copy()
    log.info(f"[OHLCV] 최종 {len(df)}일치 (shift 적용 후)")
    return df


# ═══════════════════════════════════════════════════════════════════
# 2. KOSDAQ 지수 수집 (pykrx)
# ═══════════════════════════════════════════════════════════════════

def fetch_kosdaq_index(
    start: str = HIST_START,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """KOSDAQ 지수 OHLCV (레짐 판별에 사용)."""
    if not PYKRX_AVAILABLE:
        raise RuntimeError("pykrx가 설치되지 않았습니다.")

    if end is None:
        end = datetime.now().strftime("%Y%m%d")
    start = start.replace("-", "")
    end = end.replace("-", "")

    log.info(f"[INDEX] KOSDAQ 지수 수집: {start} ~ {end}")

    df = krx.get_index_ohlcv(start, end, KOSDAQ_INDEX_TICKER)
    if df.empty:
        raise ValueError("[INDEX] KOSDAQ 지수 데이터 없음")

    df = df.rename(columns={
        "시가": "Open", "고가": "High", "저가": "Low",
        "종가": "Close", "거래량": "Volume",
    })

    df["prev_Close"] = df["Close"].shift(1)
    df = df.iloc[1:].copy()

    log.info(f"[INDEX] {len(df)}일치 수집 완료")
    return df


# ═══════════════════════════════════════════════════════════════════
# 3. 투자자별 매매동향 (pykrx — KRX 공식 데이터)
# ═══════════════════════════════════════════════════════════════════

def fetch_investor_trading(
    ticker: str = TICKER,
    start: Optional[str] = None,
    end: Optional[str] = None,
    days: int = 30,
) -> pd.DataFrame:
    """
    개별 종목의 일별 투자자별 순매수량/순매수금액 수집.

    pykrx 함수:
      - get_market_trading_volume_by_date(start, end, ticker)
        → 일별 투자자 유형별 순매수 "수량"
      - get_market_trading_value_by_date(start, end, ticker)
        → 일별 투자자 유형별 순매수 "금액"

    ※ 당일 확정 데이터는 오후 6시 이후에 제공

    Returns:
        DataFrame with columns:
            date,
            foreign_net_qty, foreign_net_amt,   (외국인)
            inst_net_qty, inst_net_amt,         (기관합계)
            individual_net_qty, individual_net_amt (개인)
    """
    if not PYKRX_AVAILABLE:
        raise RuntimeError("pykrx가 설치되지 않았습니다.")

    if end is None:
        end = datetime.now().strftime("%Y%m%d")
    if start is None:
        start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    start = start.replace("-", "")
    end = end.replace("-", "")

    log.info(f"[INVESTOR] {ticker} 투자자별 매매동향: {start} ~ {end}")

    # ── 거래량 (순매수 수량) ──
    try:
        df_vol = krx.get_market_trading_volume_by_date(start, end, ticker)
        sleep(PYKRX_DELAY)
    except Exception as e:
        log.error(f"[INVESTOR] 거래량 수집 실패: {e}")
        return pd.DataFrame()

    # ── 거래대금 (순매수 금액) ──
    try:
        df_val = krx.get_market_trading_value_by_date(start, end, ticker)
        sleep(PYKRX_DELAY)
    except Exception as e:
        log.warning(f"[INVESTOR] 거래대금 수집 실패: {e}")
        df_val = pd.DataFrame()

    if df_vol.empty:
        log.warning("[INVESTOR] 거래량 데이터 없음")
        return pd.DataFrame()

    # ── 컬럼 매핑 ──
    # pykrx 반환 컬럼: 기관합계, 기타법인, 개인, 외국인합계, 전체
    # (또는: 금융투자, 보험, 투신, 사모, 은행, 기타금융, 연기금등, 기타법인, 개인, 외국인합계, 전체)
    records = []

    for date_idx in df_vol.index:
        date_str = date_idx.strftime("%Y-%m-%d") if hasattr(date_idx, "strftime") else str(date_idx)

        vol_row = df_vol.loc[date_idx]
        val_row = df_val.loc[date_idx] if not df_val.empty and date_idx in df_val.index else None

        # 외국인 순매수
        foreign_qty = _get_col_value(vol_row, ["외국인합계", "외국인"])
        # 기관 순매수 (기관합계 또는 개별 합산)
        inst_qty = _get_col_value(vol_row, ["기관합계", "기관"])
        if inst_qty == 0:
            # 개별 항목 합산 시도
            inst_cols = ["금융투자", "보험", "투신", "사모", "은행", "기타금융", "연기금등"]
            inst_qty = sum(_get_col_value(vol_row, [c]) for c in inst_cols)
        # 개인 순매수
        indiv_qty = _get_col_value(vol_row, ["개인"])

        record = {
            "date": date_str,
            "foreign_net_qty": int(foreign_qty),
            "inst_net_qty": int(inst_qty),
            "individual_net_qty": int(indiv_qty),
        }

        # 거래대금 추가
        if val_row is not None:
            record["foreign_net_amt"] = int(_get_col_value(val_row, ["외국인합계", "외국인"]))
            record["inst_net_amt"] = int(_get_col_value(val_row, ["기관합계", "기관"]))
            if record["inst_net_amt"] == 0:
                inst_cols = ["금융투자", "보험", "투신", "사모", "은행", "기타금융", "연기금등"]
                record["inst_net_amt"] = int(sum(_get_col_value(val_row, [c]) for c in inst_cols))
            record["individual_net_amt"] = int(_get_col_value(val_row, ["개인"]))

        records.append(record)

    df = pd.DataFrame(records)
    log.info(f"[INVESTOR] {len(df)}일치 수급 데이터 수집 완료")
    return df


def _get_col_value(row, col_names: list) -> float:
    """여러 가능한 컬럼명 중 존재하는 것의 값을 반환."""
    for name in col_names:
        if name in row.index:
            val = row[name]
            if pd.notna(val):
                return float(val)
    return 0.0


# ═══════════════════════════════════════════════════════════════════
# 4. 공매도 데이터 (pykrx — KRX 공식 데이터)
# ═══════════════════════════════════════════════════════════════════

def fetch_short_selling(
    ticker: str = TICKER,
    start: Optional[str] = None,
    end: Optional[str] = None,
    days: int = 30,
) -> pd.DataFrame:
    """
    개별 종목의 일별 공매도 거래량/거래대금 수집.

    pykrx 함수:
      - get_shorting_volume_by_date(start, end, ticker)
        → 일별 공매도 수량, 매수 수량, 공매도 비중

    Returns:
        DataFrame with columns:
            date, short_volume, total_volume, short_ratio
    """
    if not PYKRX_AVAILABLE:
        raise RuntimeError("pykrx가 설치되지 않았습니다.")

    if end is None:
        end = datetime.now().strftime("%Y%m%d")
    if start is None:
        start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    start = start.replace("-", "")
    end = end.replace("-", "")

    log.info(f"[SHORT] {ticker} 공매도 데이터: {start} ~ {end}")

    try:
        df = krx.get_shorting_volume_by_date(start, end, ticker)
        sleep(PYKRX_DELAY)
    except Exception as e:
        log.error(f"[SHORT] 수집 실패: {e}")
        return pd.DataFrame()

    if df.empty:
        log.warning("[SHORT] 공매도 데이터 없음")
        return pd.DataFrame()

    # pykrx 반환 컬럼: 공매도, 매수, 비중
    records = []
    for date_idx in df.index:
        date_str = date_idx.strftime("%Y-%m-%d") if hasattr(date_idx, "strftime") else str(date_idx)

        row = df.loc[date_idx]
        short_vol = _get_col_value(row, ["공매도", "공매도거래량"])
        total_vol = _get_col_value(row, ["매수", "총거래량"])
        ratio = _get_col_value(row, ["비중", "공매도비중"])

        # 비중이 없으면 직접 계산
        if ratio == 0.0 and total_vol > 0:
            ratio = round(short_vol / total_vol * 100, 2)

        records.append({
            "date": date_str,
            "short_volume": int(short_vol),
            "total_volume": int(total_vol),
            "short_ratio": float(ratio),
        })

    result = pd.DataFrame(records)
    log.info(f"[SHORT] {len(result)}일치 공매도 데이터 수집 완료")
    return result


def fetch_short_balance(
    ticker: str = TICKER,
    start: Optional[str] = None,
    end: Optional[str] = None,
    days: int = 30,
) -> pd.DataFrame:
    """
    공매도 잔고 추이 수집 (숏커버링 시그널 감지용).

    pykrx 함수:
      - get_shorting_status_by_date(start, end, ticker)
        → 공매도잔고, 상장주식수, 공매도금액, 시가총액, 비중
    """
    if not PYKRX_AVAILABLE:
        raise RuntimeError("pykrx가 설치되지 않았습니다.")

    if end is None:
        end = datetime.now().strftime("%Y%m%d")
    if start is None:
        start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    start = start.replace("-", "")
    end = end.replace("-", "")

    log.info(f"[SHORT_BAL] {ticker} 공매도 잔고: {start} ~ {end}")

    try:
        df = krx.get_shorting_status_by_date(start, end, ticker)
        sleep(PYKRX_DELAY)
    except Exception as e:
        log.error(f"[SHORT_BAL] 수집 실패: {e}")
        return pd.DataFrame()

    if df.empty:
        log.warning("[SHORT_BAL] 공매도 잔고 데이터 없음")
        return pd.DataFrame()

    # pykrx 반환 컬럼: 공매도잔고, 상장주식수, 공매도금액, 시가총액, 비중
    records = []
    for date_idx in df.index:
        date_str = date_idx.strftime("%Y-%m-%d") if hasattr(date_idx, "strftime") else str(date_idx)
        row = df.loc[date_idx]

        records.append({
            "date": date_str,
            "short_balance": int(_get_col_value(row, ["공매도잔고"])),
            "listed_shares": int(_get_col_value(row, ["상장주식수"])),
            "short_balance_ratio": float(_get_col_value(row, ["비중"])),
        })

    result = pd.DataFrame(records)
    log.info(f"[SHORT_BAL] {len(result)}일치 수집 완료")
    return result


# ═══════════════════════════════════════════════════════════════════
# 5. ETF 프록시 데이터 (KODEX 2차전지산업)
# ═══════════════════════════════════════════════════════════════════

def fetch_etf_proxy(
    start: str = HIST_START,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """KODEX 2차전지산업 ETF — 섹터 자금 유입 프록시."""
    if end is None:
        end = datetime.now().strftime("%Y%m%d")
    start = start.replace("-", "")
    end = end.replace("-", "")

    log.info(f"[ETF] {ETF_TICKER} (KODEX 2차전지) 수집: {start} ~ {end}")

    df = pd.DataFrame()

    if PYKRX_AVAILABLE:
        try:
            df = krx.get_market_ohlcv_by_date(start, end, ETF_TICKER)
            if not df.empty:
                df = df.rename(columns={
                    "시가": "Open", "고가": "High", "저가": "Low",
                    "종가": "Close", "거래량": "Volume",
                })
                sleep(PYKRX_DELAY)
        except Exception as e:
            log.warning(f"[ETF] pykrx 실패: {e}")

    if df.empty:
        log.warning(f"[ETF] {ETF_TICKER}: 데이터 없음")
        return pd.DataFrame()

    # 거래량 이동평균 및 프록시 지표
    df["vol_ma20"] = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = np.where(
        df["vol_ma20"] > 0,
        (df["Volume"] / df["vol_ma20"]).round(3),
        1.0,
    )
    df["is_bullish"] = (df["Close"] > df["Open"]).astype(int)

    for col in ["Close", "Volume", "vol_ma20", "vol_ratio", "is_bullish"]:
        if col in df.columns:
            df[f"prev_{col}"] = df[col].shift(1)

    df = df.iloc[1:].copy()
    log.info(f"[ETF] {len(df)}일치 수집 완료")
    return df


# ═══════════════════════════════════════════════════════════════════
# 6. 상태 관리 (State File)
# ═══════════════════════════════════════════════════════════════════

def load_state(filepath: Path = PIPELINE_STATE_FILE) -> dict:
    """파이프라인 상태 파일 로드."""
    default_state = {
        "last_ohlcv_update": None,
        "last_supply_update": None,
        "last_short_update": None,
        "total_ohlcv_rows": 0,
        "total_supply_rows": 0,
        "total_short_rows": 0,
        "data_source": "pykrx",
        "errors": [],
        "created_at": datetime.now().isoformat(),
    }

    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                state = json.load(f)
            log.info(f"[STATE] 기존 상태 로드: {filepath}")
            return state
        except json.JSONDecodeError:
            log.warning(f"[STATE] 상태 파일 손상, 초기화: {filepath}")

    return default_state


def save_state(state: dict, filepath: Path = PIPELINE_STATE_FILE) -> None:
    """파이프라인 상태 파일 저장."""
    state["updated_at"] = datetime.now().isoformat()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"[STATE] 상태 저장: {filepath}")


def save_supply_data(new_data: dict, filepath: Path = SUPPLY_FILE) -> None:
    """수급 데이터를 날짜별로 누적 저장."""
    existing = {}
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            log.warning(f"[SUPPLY] 파일 손상, 새로 생성: {filepath}")

    for date_key, values in new_data.items():
        if date_key in existing:
            existing[date_key].update(values)
        else:
            existing[date_key] = values

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2, default=str)

    log.info(f"[SUPPLY] {len(new_data)}일치 추가/갱신 → 총 {len(existing)}일치")


def save_ohlcv_data(df: pd.DataFrame, filepath: Path = OHLCV_FILE) -> None:
    """OHLCV 데이터를 JSON으로 저장."""
    data = {}
    for idx, row in df.iterrows():
        date_key = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
        data[date_key] = {
            "Open": float(row.get("Open", 0)),
            "High": float(row.get("High", 0)),
            "Low": float(row.get("Low", 0)),
            "Close": float(row.get("Close", 0)),
            "Volume": int(row.get("Volume", 0)),
            "prev_Open": float(row.get("prev_Open", 0)),
            "prev_High": float(row.get("prev_High", 0)),
            "prev_Low": float(row.get("prev_Low", 0)),
            "prev_Close": float(row.get("prev_Close", 0)),
            "prev_Volume": int(row.get("prev_Volume", 0)),
        }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    log.info(f"[OHLCV] {len(data)}일치 저장: {filepath}")


# ═══════════════════════════════════════════════════════════════════
# 7. Telegram 알림
# ═══════════════════════════════════════════════════════════════════

def send_telegram(message: str) -> None:
    """Telegram 알림 발송."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        log.warning("[TELEGRAM] 토큰/채팅ID 미설정, 알림 생략")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}

    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            log.info("[TELEGRAM] 알림 발송 완료")
        else:
            log.warning(f"[TELEGRAM] 발송 실패: {resp.status_code}")
    except Exception as e:
        log.warning(f"[TELEGRAM] 발송 오류: {e}")


# ═══════════════════════════════════════════════════════════════════
# 8. 메인 파이프라인
# ═══════════════════════════════════════════════════════════════════

def run_daily_collection(mode: str = "incremental") -> dict:
    """
    일일 데이터 수집 파이프라인.

    Args:
        mode:
            "full"        — 전체 기간 재수집 (초기 구축/복구)
            "incremental" — 최근 7일치만 수집하여 기존 데이터에 병합
    """
    log.info(f"{'='*60}")
    log.info(f"[PIPELINE] {TICKER_NAME} 데이터 수집 시작")
    log.info(f"  mode={mode}, source=pykrx (KRX)")
    log.info(f"{'='*60}")

    state = load_state()
    result = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "source": "pykrx",
        "ohlcv": {"status": "skip", "rows": 0},
        "supply": {"status": "skip", "rows": 0},
        "short": {"status": "skip", "rows": 0},
        "short_balance": {"status": "skip", "rows": 0},
        "etf": {"status": "skip", "rows": 0},
        "index": {"status": "skip", "rows": 0},
        "errors": [],
    }

    if mode == "full":
        start = HIST_START
    else:
        start = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")

    end = datetime.now().strftime("%Y%m%d")

    # ── 1) OHLCV ──
    try:
        df_ohlcv = fetch_ohlcv(TICKER, start=start, end=end)
        save_ohlcv_data(df_ohlcv)
        result["ohlcv"] = {"status": "ok", "rows": len(df_ohlcv)}
        state["last_ohlcv_update"] = datetime.now().strftime("%Y-%m-%d")
        state["total_ohlcv_rows"] = len(df_ohlcv)
    except Exception as e:
        log.error(f"[OHLCV] 수집 실패: {e}")
        result["ohlcv"] = {"status": "error", "error": str(e)}
        result["errors"].append(f"OHLCV: {e}")

    # ── 2) KOSDAQ 지수 ──
    try:
        df_index = fetch_kosdaq_index(start=start, end=end)
        result["index"] = {"status": "ok", "rows": len(df_index)}
    except Exception as e:
        log.error(f"[INDEX] 수집 실패: {e}")
        result["index"] = {"status": "error", "error": str(e)}
        result["errors"].append(f"INDEX: {e}")

    sleep(PYKRX_DELAY)

    # ── 3) 투자자별 매매동향 ──
    try:
        df_investor = fetch_investor_trading(TICKER, start=start, end=end)

        if not df_investor.empty:
            supply_dict = {}
            for _, row in df_investor.iterrows():
                supply_dict[row["date"]] = {
                    k: v for k, v in row.items() if k != "date"
                }
            save_supply_data(supply_dict)

            result["supply"] = {"status": "ok", "rows": len(df_investor)}
            state["last_supply_update"] = datetime.now().strftime("%Y-%m-%d")
            state["total_supply_rows"] += len(df_investor)
    except Exception as e:
        log.error(f"[INVESTOR] 수집 실패: {e}")
        result["supply"] = {"status": "error", "error": str(e)}
        result["errors"].append(f"INVESTOR: {e}")

    # ── 4) 공매도 거래량 ──
    try:
        df_short = fetch_short_selling(TICKER, start=start, end=end)

        if not df_short.empty:
            short_dict = {}
            for _, row in df_short.iterrows():
                short_dict[row["date"]] = {
                    k: v for k, v in row.items() if k != "date"
                }
            save_supply_data(short_dict)  # 기존 수급 데이터에 병합

            result["short"] = {"status": "ok", "rows": len(df_short)}
            state["last_short_update"] = datetime.now().strftime("%Y-%m-%d")
    except Exception as e:
        log.error(f"[SHORT] 수집 실패: {e}")
        result["short"] = {"status": "error", "error": str(e)}
        result["errors"].append(f"SHORT: {e}")

    # ── 5) 공매도 잔고 ──
    try:
        df_bal = fetch_short_balance(TICKER, start=start, end=end)

        if not df_bal.empty:
            bal_dict = {}
            for _, row in df_bal.iterrows():
                bal_dict[row["date"]] = {
                    k: v for k, v in row.items() if k != "date"
                }
            save_supply_data(bal_dict)

            result["short_balance"] = {"status": "ok", "rows": len(df_bal)}
    except Exception as e:
        log.error(f"[SHORT_BAL] 수집 실패: {e}")
        result["short_balance"] = {"status": "error", "error": str(e)}
        result["errors"].append(f"SHORT_BAL: {e}")

    # ── 6) ETF 프록시 ──
    try:
        df_etf = fetch_etf_proxy(start=start, end=end)
        result["etf"] = {"status": "ok", "rows": len(df_etf)}
    except Exception as e:
        log.error(f"[ETF] 수집 실패: {e}")
        result["etf"] = {"status": "error", "error": str(e)}
        result["errors"].append(f"ETF: {e}")

    # ── 상태 저장 & 알림 ──
    state["data_source"] = "pykrx"
    state["errors"] = result["errors"][-10:]
    save_state(state)

    _send_collection_report(result)
    return result


def _send_collection_report(result: dict) -> None:
    """수집 결과 Telegram 알림."""
    emoji = {"ok": "✅", "error": "❌", "skip": "⏭️"}

    lines = [
        f"📊 <b>{TICKER_NAME} 데이터 수집 (pykrx)</b>",
        f"시각: {result['timestamp'][:19]}",
        f"모드: {result['mode']}",
        "",
    ]

    labels = {
        "ohlcv": "OHLCV", "index": "KOSDAQ지수", "supply": "수급",
        "short": "공매도", "short_balance": "공매도잔고", "etf": "ETF",
    }

    for key, label in labels.items():
        data = result.get(key, {"status": "skip"})
        e = emoji.get(data["status"], "❓")
        rows = data.get("rows", 0)
        lines.append(f"{e} {label}: {rows}건")

    if result["errors"]:
        lines.append("")
        lines.append("⚠️ 오류:")
        for err in result["errors"]:
            lines.append(f"  · {err[:80]}")

    send_telegram("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════
# 9. CLI 엔트리포인트
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=f"{TICKER_NAME} ({TICKER}) 데이터 수집 (pykrx)"
    )
    parser.add_argument(
        "--mode", choices=["full", "incremental"],
        default="incremental",
        help="full: 전체 재수집 / incremental: 최근 업데이트",
    )
    parser.add_argument(
        "--state-dir", type=str, default=".",
        help="상태 파일 저장 디렉토리",
    )
    args = parser.parse_args()

    STATE_DIR = Path(args.state_dir)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    OHLCV_FILE = STATE_DIR / "ohlcv_247540.json"
    SUPPLY_FILE = STATE_DIR / "supply_data_247540.json"
    PIPELINE_STATE_FILE = STATE_DIR / "pipeline_state_247540.json"

    result = run_daily_collection(mode=args.mode)
    exit(1 if result["errors"] else 0)
