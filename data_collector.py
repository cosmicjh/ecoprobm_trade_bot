"""
에코프로비엠 (247540) v4 자동트레이딩 — Phase 1-1: 데이터 수집 파이프라인
==========================================================================
수집 대상:
  1. 일봉 OHLCV (FinanceDataReader)
  2. 투자자별 매매동향 — 외국인/기관 순매수 (한투 API)
  3. 공매도 거래량 (한투 API)
  4. 프로그램 매매 동향 (한투 API)

인프라: GitHub Actions + cron-job.org (09:05, 11:00, 13:30, 15:10 KST)
상태관리: JSON state file (supply_data.json)

주의사항:
  - FDR 데이터는 반드시 .shift(1) 적용 (look-ahead bias 방지)
  - 한투 API 투자자별 매매동향은 장중 20분 지연 데이터
  - ACC_NO는 반드시 하이픈 포함 (예: 12345678-01)
  - KOSDAQ 지수 = KQ11
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import requests

# ── 외부 라이브러리 (GitHub Actions 환경에서 설치 필요) ─────────
try:
    import FinanceDataReader as fdr
except ImportError:
    fdr = None
    logging.warning("FinanceDataReader not installed. OHLCV collection disabled.")

try:
    import mojito
except ImportError:
    mojito = None
    logging.warning("mojito2 not installed. KIS API collection disabled.")

# ── 설정 ─────────────────────────────────────────────────────────
TICKER = "247540"
TICKER_NAME = "에코프로비엠"
MARKET_INDEX = "KQ11"  # KOSDAQ

# 데이터 수집 기간 (백테스트용)
HIST_START = "2022-01-01"

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
# 1. OHLCV 데이터 수집 (FinanceDataReader)
# ═══════════════════════════════════════════════════════════════════

def fetch_ohlcv(
    ticker: str = TICKER,
    start: str = HIST_START,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    FDR에서 일봉 OHLCV를 가져온 뒤 .shift(1)을 적용하여
    '전일까지 확정된 데이터'만 사용할 수 있도록 처리합니다.

    Returns:
        DataFrame with columns:
            Open, High, Low, Close, Volume (원본 — 당일 데이터)
            prev_Open, prev_High, prev_Low, prev_Close, prev_Volume
                (.shift(1) — 전일 확정 데이터, 전략 시그널에 사용)
    """
    if fdr is None:
        raise RuntimeError("FinanceDataReader가 설치되지 않았습니다.")

    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    log.info(f"[OHLCV] {ticker} 데이터 수집: {start} ~ {end}")

    # FDR 호출 (yfinance 대신 사용 — GitHub Actions IP 차단 회피)
    df = fdr.DataReader(ticker, start, end)

    if df.empty:
        log.warning(f"[OHLCV] {ticker}: 데이터 없음. FDR fallback 시도...")
        # fallback: 네이버 소스 시도
        df = fdr.DataReader(ticker, start, end, exchange="KRX")

    if df.empty:
        raise ValueError(f"[OHLCV] {ticker}: 데이터를 가져올 수 없습니다.")

    log.info(f"[OHLCV] {len(df)}일치 데이터 수집 완료 ({df.index[0]} ~ {df.index[-1]})")

    # ── .shift(1): look-ahead bias 방지 ──
    # 전략 시그널 계산에는 반드시 prev_* 컬럼만 사용해야 함
    shifted_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in shifted_cols:
        df[f"prev_{col}"] = df[col].shift(1)

    # 첫 행은 prev_* 가 NaN이므로 제거
    df = df.iloc[1:].copy()

    return df


def fetch_market_index(
    start: str = HIST_START,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """KOSDAQ 지수 데이터 수집 (레짐 판별에 사용)."""
    if fdr is None:
        raise RuntimeError("FinanceDataReader가 설치되지 않았습니다.")

    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    log.info(f"[INDEX] {MARKET_INDEX} 데이터 수집: {start} ~ {end}")
    df = fdr.DataReader(MARKET_INDEX, start, end)

    if df.empty:
        raise ValueError(f"[INDEX] {MARKET_INDEX}: 데이터를 가져올 수 없습니다.")

    # 지수도 동일하게 shift(1) 적용
    df["prev_Close"] = df["Close"].shift(1)
    df = df.iloc[1:].copy()

    log.info(f"[INDEX] {len(df)}일치 수집 완료")
    return df


# ═══════════════════════════════════════════════════════════════════
# 2. 투자자별 매매동향 (한투 API — mojito2)
# ═══════════════════════════════════════════════════════════════════

def _get_kis_broker() -> object:
    """한투 API 브로커 객체 생성. 환경변수에서 인증정보 로드."""
    if mojito is None:
        raise RuntimeError("mojito2가 설치되지 않았습니다.")

    api_key = os.getenv("KIS_API_KEY")
    api_secret = os.getenv("KIS_API_SECRET")
    acc_no = os.getenv("KIS_ACC_NO")  # 예: "12345678-01" (하이픈 필수)
    is_mock = os.getenv("KIS_MOCK", "Y") == "Y"

    if not all([api_key, api_secret, acc_no]):
        raise ValueError(
            "KIS_API_KEY, KIS_API_SECRET, KIS_ACC_NO 환경변수가 필요합니다."
        )

    broker = mojito.KoreaInvestment(
        api_key=api_key,
        api_secret=api_secret,
        acc_no=acc_no,
        mock=is_mock,
    )
    log.info(f"[KIS] 브로커 연결 완료 (mock={is_mock})")
    return broker


def fetch_investor_trading(
    ticker: str = TICKER,
    days: int = 30,
) -> pd.DataFrame:
    """
    투자자별 매매동향 수집 (외국인, 기관, 개인 순매수량).

    한투 API: /uapi/domestic-stock/v1/quotations/investor
    - 일별 투자자 유형별 순매수량/금액 제공
    - 장중에는 20분 지연 데이터

    Returns:
        DataFrame with columns:
            date, foreign_net_qty, foreign_net_amt,
            inst_net_qty, inst_net_amt,
            individual_net_qty, individual_net_amt
    """
    broker = _get_kis_broker()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    log.info(f"[INVESTOR] {ticker} 투자자별 매매동향: 최근 {days}일")

    # ── 한투 API 호출 ──
    # API: 주식현재가 투자자 (FHKST01010900)
    # 참고: mojito2의 정확한 메서드명은 버전에 따라 다를 수 있음
    # 아래는 한투 OpenAPI 직접 호출 방식
    path = "/uapi/domestic-stock/v1/quotations/investor"
    headers = {
        "tr_id": "FHKST01010900",  # 주식현재가 투자자
    }
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",  # 주식
        "FID_INPUT_ISCD": ticker,
    }

    try:
        # 한투 실전/모의 서버 URL 설정
        base_url = "https://openapi.koreainvestment.com:9443" if not broker.mock else "https://openapivts.koreainvestment.com:29443"
        
        # 직접 API 요청을 위한 헤더 세팅
        req_headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {broker.access_token}",
            "appkey": broker.api_key,
            "appsecret": broker.api_secret,
            "tr_id": headers["tr_id"]
        }
        
        # requests로 직접 호출
        res = requests.get(base_url + path, headers=req_headers, params=params)
        resp = res.json()
        
        records = _parse_investor_response(resp)
    except Exception as e:
        log.warning(f"[INVESTOR] API 호출 실패: {e}")
        log.info("[INVESTOR] 대안: KRX 데이터마켓 크롤링 또는 캐시 사용")
        records = []

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    log.info(f"[INVESTOR] {len(df)}일치 수급 데이터 수집 완료")
    return df


def _parse_investor_response(resp: dict) -> list:
    """
    한투 API 투자자별 매매동향 응답을 파싱합니다.

    실제 응답 구조는 API 문서 참조:
    https://apiportal.koreainvestment.com/apiservice/apiservice-domestic-stock-quotations

    output 배열의 각 항목:
    - stck_bsop_date: 영업일자
    - frgn_ntby_qty: 외국인 순매수 수량
    - frgn_ntby_tr_pbmn: 외국인 순매수 금액
    - orgn_ntby_qty: 기관 순매수 수량
    - orgn_ntby_tr_pbmn: 기관 순매수 금액
    - prsn_ntby_qty: 개인 순매수 수량
    - prsn_ntby_tr_pbmn: 개인 순매수 금액
    """
    records = []

    output = resp.get("output", [])
    if not output:
        output = resp.get("output1", [])

    for item in output:
        try:
            record = {
                "date": item.get("stck_bsop_date", ""),
                "foreign_net_qty": _safe_int(item.get("frgn_ntby_qty", 0)),
                "foreign_net_amt": _safe_int(item.get("frgn_ntby_tr_pbmn", 0)),
                "inst_net_qty": _safe_int(item.get("orgn_ntby_qty", 0)),
                "inst_net_amt": _safe_int(item.get("orgn_ntby_tr_pbmn", 0)),
                "individual_net_qty": _safe_int(item.get("prsn_ntby_qty", 0)),
                "individual_net_amt": _safe_int(item.get("prsn_ntby_tr_pbmn", 0)),
            }
            if record["date"]:
                records.append(record)
        except (ValueError, TypeError) as e:
            log.debug(f"[INVESTOR] 파싱 스킵: {e}")
            continue

    return records


# ═══════════════════════════════════════════════════════════════════
# 3. 공매도 데이터 (한투 API)
# ═══════════════════════════════════════════════════════════════════

def fetch_short_selling(
    ticker: str = TICKER,
    days: int = 30,
) -> pd.DataFrame:
    """
    종목별 공매도 거래량/거래대금 수집.

    한투 API: /uapi/domestic-stock/v1/quotations/inquire-daily-short-selling
    tr_id: FHKST03060100

    Returns:
        DataFrame with columns:
            date, short_volume, short_amount, total_volume,
            short_ratio (공매도 비중 %)
    """
    broker = _get_kis_broker()

    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

    log.info(f"[SHORT] {ticker} 공매도 데이터: 최근 {days}일")

    path = "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
    headers = {
        "tr_id": "FHKST03060100",  # 공매도 일별 거래
    }
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": ticker,
        "FID_INPUT_DATE_1": start_date,
        "FID_INPUT_DATE_2": end_date,
    }

    try:
        base_url = "https://openapi.koreainvestment.com:9443" if not broker.mock else "https://openapivts.koreainvestment.com:29443"
        req_headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {broker.access_token}",
            "appkey": broker.api_key,
            "appsecret": broker.api_secret,
            "tr_id": headers["tr_id"]
        }
        res = requests.get(base_url + path, headers=req_headers, params=params)
        resp = res.json()
        
        records = _parse_short_response(resp)
    except Exception as e:
        log.warning(f"[SHORT] API 호출 실패: {e}")
        records = []

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 공매도 비중 계산
    df["short_ratio"] = np.where(
        df["total_volume"] > 0,
        (df["short_volume"] / df["total_volume"] * 100).round(2),
        0.0,
    )

    log.info(f"[SHORT] {len(df)}일치 공매도 데이터 수집 완료")
    return df


def _parse_short_response(resp: dict) -> list:
    """공매도 API 응답 파싱."""
    records = []
    output = resp.get("output", resp.get("output1", []))

    for item in output:
        try:
            record = {
                "date": item.get("stck_bsop_date", ""),
                "short_volume": _safe_int(item.get("seld_cntg_qty", 0)),
                "short_amount": _safe_int(item.get("seld_cntg_amt", 0)),
                "total_volume": _safe_int(item.get("acml_vol", 0)),
            }
            if record["date"]:
                records.append(record)
        except (ValueError, TypeError) as e:
            log.debug(f"[SHORT] 파싱 스킵: {e}")
            continue

    return records


# ═══════════════════════════════════════════════════════════════════
# 4. ETF 프록시 데이터 (KODEX 2차전지산업)
# ═══════════════════════════════════════════════════════════════════

ETF_TICKER = "305720"  # KODEX 2차전지산업

def fetch_etf_proxy(
    start: str = HIST_START,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    KODEX 2차전지산업 ETF 데이터 수집.
    섹터 자금 유입/유출의 프록시 시그널로 활용.
    """
    if fdr is None:
        raise RuntimeError("FinanceDataReader가 설치되지 않았습니다.")

    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    log.info(f"[ETF] {ETF_TICKER} (KODEX 2차전지) 수집: {start} ~ {end}")

    df = fdr.DataReader(ETF_TICKER, start, end)
    if df.empty:
        log.warning(f"[ETF] {ETF_TICKER}: 데이터 없음")
        return pd.DataFrame()

    # 거래량 이동평균 (수급 프록시)
    df["vol_ma20"] = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = np.where(
        df["vol_ma20"] > 0,
        (df["Volume"] / df["vol_ma20"]).round(3),
        1.0,
    )

    # 양봉 여부
    df["is_bullish"] = (df["Close"] > df["Open"]).astype(int)

    # shift(1) 적용
    for col in ["Close", "Volume", "vol_ma20", "vol_ratio", "is_bullish"]:
        df[f"prev_{col}"] = df[col].shift(1)

    df = df.iloc[1:].copy()
    log.info(f"[ETF] {len(df)}일치 수집 완료")
    return df


# ═══════════════════════════════════════════════════════════════════
# 5. 상태 관리 (State File)
# ═══════════════════════════════════════════════════════════════════

def load_state(filepath: Path = PIPELINE_STATE_FILE) -> dict:
    """파이프라인 상태 파일 로드. 없으면 초기 상태 반환."""
    default_state = {
        "last_ohlcv_update": None,
        "last_supply_update": None,
        "last_short_update": None,
        "total_ohlcv_rows": 0,
        "total_supply_rows": 0,
        "total_short_rows": 0,
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
    """
    수급 데이터를 날짜별로 누적 저장합니다.

    구조:
    {
        "2026-04-07": {
            "foreign_net_qty": 12345,
            "inst_net_qty": -5678,
            "short_volume": 9012,
            "short_ratio": 5.2,
            ...
        },
        ...
    }
    """
    existing = {}
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            log.warning(f"[SUPPLY] 파일 손상, 새로 생성: {filepath}")

    # 날짜 키로 병합 (최신 데이터가 기존 데이터를 덮어씀)
    for date_key, values in new_data.items():
        existing[date_key] = values

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2, default=str)

    log.info(f"[SUPPLY] {len(new_data)}일치 추가 → 총 {len(existing)}일치 저장")


def save_ohlcv_data(df: pd.DataFrame, filepath: Path = OHLCV_FILE) -> None:
    """OHLCV 데이터를 JSON으로 저장. 날짜를 키로 사용."""
    data = {}
    for idx, row in df.iterrows():
        date_key = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
        data[date_key] = {
            "Open": float(row["Open"]),
            "High": float(row["High"]),
            "Low": float(row["Low"]),
            "Close": float(row["Close"]),
            "Volume": int(row["Volume"]),
            "prev_Open": float(row["prev_Open"]),
            "prev_High": float(row["prev_High"]),
            "prev_Low": float(row["prev_Low"]),
            "prev_Close": float(row["prev_Close"]),
            "prev_Volume": int(row["prev_Volume"]),
        }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    log.info(f"[OHLCV] {len(data)}일치 저장: {filepath}")


# ═══════════════════════════════════════════════════════════════════
# 6. 유틸리티
# ═══════════════════════════════════════════════════════════════════

def _safe_int(val) -> int:
    """안전한 정수 변환. 빈 문자열, None 등 처리."""
    if val is None or val == "":
        return 0
    try:
        return int(float(str(val).replace(",", "")))
    except (ValueError, TypeError):
        return 0


def _today_str() -> str:
    """오늘 날짜 문자열 (YYYY-MM-DD)."""
    return datetime.now().strftime("%Y-%m-%d")


def send_telegram(message: str) -> None:
    """
    Telegram 알림 발송.
    환경변수: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    """
    import requests

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
# 7. 메인 파이프라인
# ═══════════════════════════════════════════════════════════════════

def run_daily_collection(mode: str = "incremental") -> dict:
    """
    일일 데이터 수집 파이프라인 실행.

    Args:
        mode:
            "full"        — 전체 기간 데이터 재수집 (초기 구축 또는 복구)
            "incremental" — 최근 5일치만 수집하여 기존 데이터에 병합

    Returns:
        수집 결과 요약 dict
    """
    log.info(f"{'='*60}")
    log.info(f"[PIPELINE] 에코프로비엠 데이터 수집 시작 (mode={mode})")
    log.info(f"{'='*60}")

    state = load_state()
    result = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "ohlcv": {"status": "skip", "rows": 0},
        "supply": {"status": "skip", "rows": 0},
        "short": {"status": "skip", "rows": 0},
        "etf": {"status": "skip", "rows": 0},
        "errors": [],
    }

    # ── 1) OHLCV 수집 ──
    try:
        if mode == "full":
            start = HIST_START
        else:
            # 최근 5일치만 (주말/공휴일 고려)
            start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        df_ohlcv = fetch_ohlcv(TICKER, start=start)
        save_ohlcv_data(df_ohlcv)

        # KOSDAQ 지수도 함께 수집
        df_index = fetch_market_index(start=start)

        result["ohlcv"] = {"status": "ok", "rows": len(df_ohlcv)}
        state["last_ohlcv_update"] = _today_str()
        state["total_ohlcv_rows"] = len(df_ohlcv)

    except Exception as e:
        log.error(f"[OHLCV] 수집 실패: {e}")
        result["ohlcv"] = {"status": "error", "error": str(e)}
        result["errors"].append(f"OHLCV: {e}")

    # ── 2) 투자자별 매매동향 수집 ──
    try:
        df_investor = fetch_investor_trading(TICKER, days=30 if mode == "full" else 7)

        if not df_investor.empty:
            # 날짜별 dict 변환 후 누적 저장
            supply_dict = {}
            for _, row in df_investor.iterrows():
                date_key = row["date"].strftime("%Y-%m-%d")
                supply_dict[date_key] = {
                    "foreign_net_qty": int(row["foreign_net_qty"]),
                    "foreign_net_amt": int(row["foreign_net_amt"]),
                    "inst_net_qty": int(row["inst_net_qty"]),
                    "inst_net_amt": int(row["inst_net_amt"]),
                    "individual_net_qty": int(row["individual_net_qty"]),
                    "individual_net_amt": int(row["individual_net_amt"]),
                }
            save_supply_data(supply_dict)

            result["supply"] = {"status": "ok", "rows": len(df_investor)}
            state["last_supply_update"] = _today_str()
            state["total_supply_rows"] += len(df_investor)

    except Exception as e:
        log.error(f"[INVESTOR] 수집 실패: {e}")
        result["supply"] = {"status": "error", "error": str(e)}
        result["errors"].append(f"INVESTOR: {e}")

    # ── 3) 공매도 데이터 수집 ──
    try:
        df_short = fetch_short_selling(TICKER, days=30 if mode == "full" else 7)

        if not df_short.empty:
            # 기존 supply_data에 공매도 필드 병합
            short_dict = {}
            for _, row in df_short.iterrows():
                date_key = row["date"].strftime("%Y-%m-%d")
                short_dict[date_key] = {
                    "short_volume": int(row["short_volume"]),
                    "short_amount": int(row["short_amount"]),
                    "total_volume": int(row["total_volume"]),
                    "short_ratio": float(row["short_ratio"]),
                }

            # 기존 수급 데이터에 공매도 필드 병합
            if SUPPLY_FILE.exists():
                with open(SUPPLY_FILE, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                for date_key, vals in short_dict.items():
                    if date_key in existing:
                        existing[date_key].update(vals)
                    else:
                        existing[date_key] = vals
                with open(SUPPLY_FILE, "w", encoding="utf-8") as f:
                    json.dump(existing, f, ensure_ascii=False, indent=2)

            result["short"] = {"status": "ok", "rows": len(df_short)}
            state["last_short_update"] = _today_str()
            state["total_short_rows"] += len(df_short)

    except Exception as e:
        log.error(f"[SHORT] 수집 실패: {e}")
        result["short"] = {"status": "error", "error": str(e)}
        result["errors"].append(f"SHORT: {e}")

    # ── 4) ETF 프록시 수집 ──
    try:
        df_etf = fetch_etf_proxy(
            start=HIST_START if mode == "full" else (
                datetime.now() - timedelta(days=7)
            ).strftime("%Y-%m-%d")
        )
        result["etf"] = {"status": "ok", "rows": len(df_etf)}

    except Exception as e:
        log.error(f"[ETF] 수집 실패: {e}")
        result["etf"] = {"status": "error", "error": str(e)}
        result["errors"].append(f"ETF: {e}")

    # ── 상태 저장 ──
    state["errors"] = result["errors"][-10:]  # 최근 10개 오류만 보관
    save_state(state)

    # ── 결과 로그 ──
    log.info(f"{'='*60}")
    log.info(f"[PIPELINE] 수집 완료:")
    log.info(f"  OHLCV:    {result['ohlcv']['status']} ({result['ohlcv'].get('rows', 0)}rows)")
    log.info(f"  SUPPLY:   {result['supply']['status']} ({result['supply'].get('rows', 0)}rows)")
    log.info(f"  SHORT:    {result['short']['status']} ({result['short'].get('rows', 0)}rows)")
    log.info(f"  ETF:      {result['etf']['status']} ({result['etf'].get('rows', 0)}rows)")
    if result["errors"]:
        log.warning(f"  ERRORS:   {result['errors']}")
    log.info(f"{'='*60}")

    # ── Telegram 알림 ──
    _send_collection_report(result)

    return result


def _send_collection_report(result: dict) -> None:
    """수집 결과를 Telegram으로 발송합니다."""
    status_emoji = {
        "ok": "✅",
        "error": "❌",
        "skip": "⏭️",
    }

    lines = [
        f"📊 <b>{TICKER_NAME} 데이터 수집 리포트</b>",
        f"시각: {result['timestamp'][:19]}",
        f"모드: {result['mode']}",
        "",
    ]

    for key in ["ohlcv", "supply", "short", "etf"]:
        data = result[key]
        emoji = status_emoji.get(data["status"], "❓")
        rows = data.get("rows", 0)
        label = {"ohlcv": "OHLCV", "supply": "수급", "short": "공매도", "etf": "ETF"}[key]
        lines.append(f"{emoji} {label}: {rows}건")

    if result["errors"]:
        lines.append("")
        lines.append("⚠️ 오류:")
        for err in result["errors"]:
            lines.append(f"  · {err[:80]}")

    send_telegram("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════
# 8. CLI 엔트리포인트
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=f"{TICKER_NAME} ({TICKER}) 데이터 수집 파이프라인"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "incremental"],
        default="incremental",
        help="수집 모드: full (전체 재수집) / incremental (최근 업데이트)",
    )
    parser.add_argument(
        "--state-dir",
        type=str,
        default=".",
        help="상태 파일 저장 디렉토리",
    )
    args = parser.parse_args()

    # 상태 디렉토리 설정
    STATE_DIR = Path(args.state_dir)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    # 파일 경로 갱신
    OHLCV_FILE = STATE_DIR / "ohlcv_247540.json"
    SUPPLY_FILE = STATE_DIR / "supply_data_247540.json"
    PIPELINE_STATE_FILE = STATE_DIR / "pipeline_state_247540.json"

    result = run_daily_collection(mode=args.mode)

    # 종료 코드: 오류가 있으면 1
    exit(1 if result["errors"] else 0)
