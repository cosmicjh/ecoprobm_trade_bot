"""
에코프로비엠 (247540) v4 자동트레이딩 — Phase 1-1: 데이터 수집 파이프라인 (v3)
================================================================================
v2 → v3 변경: pykrx/FDR 완전 제거, 한투 OpenAPI 단일 소스

왜 한투 API 단일 소스인가:
  - pykrx, FDR, 네이버 금융 = 웹 스크래핑 → GitHub IP 차단
  - 한투 OpenAPI = 인증 REST API → IP 무관, GitHub Actions에서 안정 동작
  - 기존 피엔티/포스코퓨처엠 봇이 동일 환경에서 정상 운영 중

수집 데이터 & 한투 엔드포인트:
  ┌───────────────────┬──────────────────────────────────────────────────┬──────────────┐
  │ 데이터             │ 엔드포인트                                        │ tr_id         │
  ├───────────────────┼──────────────────────────────────────────────────┼──────────────┤
  │ 일봉 OHLCV        │ /quotations/inquire-daily-itemchartprice         │ FHKST03010100│
  │ 투자자별 매매동향   │ /quotations/inquire-investor                     │ FHKST01010900│
  │ 공매도 일별추이     │ /quotations/inquire-daily-shortselling-by-stock  │ FHKST03060100│
  │ 업종(KOSDAQ) 지수  │ /quotations/inquire-daily-indexchartprice        │ FHKUP03500100│
  └───────────────────┴──────────────────────────────────────────────────┴──────────────┘

인프라: GitHub Actions + cron-job.org (workflow_dispatch)
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

# ── mojito2 (토큰 관리용) ────────────────────────────────────
try:
    import mojito
    MOJITO_AVAILABLE = True
except ImportError:
    mojito = None
    MOJITO_AVAILABLE = False
    logging.warning("mojito2 not installed.")

# ── 설정 ─────────────────────────────────────────────────────────
TICKER = "247540"
TICKER_NAME = "에코프로비엠"
ETF_TICKER = "305720"  # KODEX 2차전지산업

# 한투 API 기본 설정
KIS_BASE_URL_PROD = "https://openapi.koreainvestment.com:9443"
KIS_BASE_URL_MOCK = "https://openapivts.koreainvestment.com:29443"
API_CALL_DELAY = 0.5  # API 호출 간 딜레이 (초당 제한 방지)

# 데이터 수집 기간
HIST_START = "20220101"

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
# 1. 한투 API 공통 인프라
# ═══════════════════════════════════════════════════════════════════

class KISClient:
    """
    한투 OpenAPI 클라이언트.
    mojito2에서 access_token, api_key, api_secret을 가져와서
    requests.get()으로 직접 호출하는 패턴.
    (준홍님이 검증한 방식)
    """

    def __init__(self):
        self.api_key = os.getenv("KIS_API_KEY", "")
        self.api_secret = os.getenv("KIS_API_SECRET", "")
        self.acc_no = os.getenv("KIS_ACC_NO", "")
        self.is_mock = os.getenv("KIS_MOCK", "N") == "Y"  # 기본값: 실전

        self.base_url = KIS_BASE_URL_MOCK if self.is_mock else KIS_BASE_URL_PROD
        self.access_token = None

        self._authenticate()

    def _authenticate(self):
        """mojito2로 access_token 발급."""
        if MOJITO_AVAILABLE and self.api_key and self.api_secret and self.acc_no:
            try:
                broker = mojito.KoreaInvestment(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    acc_no=self.acc_no,
                    mock=self.is_mock,
                )
                self.access_token = broker.access_token
                log.info(f"[KIS] 인증 완료 (mock={self.is_mock})")
            except Exception as e:
                log.error(f"[KIS] mojito2 인증 실패: {e}")
                self._authenticate_direct()
        else:
            self._authenticate_direct()

    def _authenticate_direct(self):
        """mojito2 없이 직접 토큰 발급."""
        if not self.api_key or not self.api_secret:
            log.error("[KIS] API_KEY/SECRET 미설정")
            return

        url = f"{self.base_url}/oauth2/tokenP"
        body = {
            "grant_type": "client_credentials",
            "appkey": self.api_key,
            "appsecret": self.api_secret,
        }
        try:
            resp = requests.post(url, json=body, timeout=10)
            data = resp.json()
            self.access_token = data.get("access_token", "")
            if self.access_token:
                log.info("[KIS] 직접 토큰 발급 완료")
            else:
                log.error(f"[KIS] 토큰 발급 실패: {data}")
        except Exception as e:
            log.error(f"[KIS] 토큰 발급 오류: {e}")

    def get(self, path: str, tr_id: str, params: dict) -> dict:
        """
        한투 API GET 요청.
        준홍님이 검증한 패턴: requests.get + 직접 헤더 구성.
        """
        if not self.access_token:
            raise RuntimeError("[KIS] access_token 없음. 인증을 먼저 수행하세요.")

        url = self.base_url + path
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.api_key,
            "appsecret": self.api_secret,
            "tr_id": tr_id,
        }

        resp = requests.get(url, headers=headers, params=params, timeout=15)
        data = resp.json()

        # API 오류 체크
        rt_cd = data.get("rt_cd", "")
        if rt_cd != "0":
            msg = data.get("msg1", "Unknown error")
            log.warning(f"[KIS] API 오류 (tr_id={tr_id}): {msg}")

        sleep(API_CALL_DELAY)
        return data


# ═══════════════════════════════════════════════════════════════════
# 2. OHLCV 데이터 수집
# ═══════════════════════════════════════════════════════════════════

def fetch_ohlcv(
    client: KISClient,
    ticker: str = TICKER,
    start: str = HIST_START,
    end: Optional[str] = None,
    period: str = "D",  # D:일, W:주, M:월
) -> pd.DataFrame:
    """
    국내주식기간별시세 (FHKST03010100)
    - 최대 100건씩 조회 → 페이징 필요

    .shift(1) 적용하여 prev_* 컬럼 생성.
    """
    if end is None:
        end = datetime.now().strftime("%Y%m%d")
    start = start.replace("-", "")
    end = end.replace("-", "")

    log.info(f"[OHLCV] {ticker} 수집: {start} ~ {end}")

    path = "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
    tr_id = "FHKST03010100"

    all_rows = []
    current_end = end

    # 페이징: 100건씩 역순으로 수집
    for page in range(50):  # 최대 5000일 (안전장치)
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": ticker,
            "FID_INPUT_DATE_1": start,
            "FID_INPUT_DATE_2": current_end,
            "FID_PERIOD_DIV_CODE": period,
            "FID_ORG_ADJ_PRC": "0",  # 수정주가 원주가 (0: 수정주가)
        }

        data = client.get(path, tr_id, params)
        output = data.get("output2", [])

        if not output:
            break

        for item in output:
            date_str = item.get("stck_bsop_date", "")
            if not date_str or date_str < start:
                continue

            all_rows.append({
                "date": date_str,
                "Open": _safe_int(item.get("stck_oprc", 0)),
                "High": _safe_int(item.get("stck_hgpr", 0)),
                "Low": _safe_int(item.get("stck_lwpr", 0)),
                "Close": _safe_int(item.get("stck_clpr", 0)),
                "Volume": _safe_int(item.get("acml_vol", 0)),
            })

        # 다음 페이지: 마지막 데이터의 전일로 이동
        last_date = output[-1].get("stck_bsop_date", "")
        if not last_date or last_date <= start:
            break

        # 하루 전으로 설정
        try:
            last_dt = datetime.strptime(last_date, "%Y%m%d")
            current_end = (last_dt - timedelta(days=1)).strftime("%Y%m%d")
        except ValueError:
            break

        if current_end < start:
            break

    if not all_rows:
        raise ValueError(f"[OHLCV] {ticker}: 데이터 없음")

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.drop_duplicates(subset="date").sort_values("date").set_index("date")

    # .shift(1): look-ahead bias 방지
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[f"prev_{col}"] = df[col].shift(1)
    df = df.iloc[1:].copy()

    log.info(f"[OHLCV] {len(df)}일치 수집 완료 ({df.index[0]} ~ {df.index[-1]})")
    return df


# ═══════════════════════════════════════════════════════════════════
# 3. 투자자별 매매동향
# ═══════════════════════════════════════════════════════════════════

def fetch_investor_trading(
    client: KISClient,
    ticker: str = TICKER,
) -> pd.DataFrame:
    """
    주식현재가 투자자 (FHKST01010900)
    - 최근 30일간 일별 투자자 유형별 순매수 수량 제공
    - 당일 데이터는 장 종료 후 확정

    output 필드:
      stck_bsop_date: 영업일자
      prsn_ntby_qty:  개인 순매수
      frgn_ntby_qty:  외국인 순매수
      orgn_ntby_qty:  기관 순매수
    """
    log.info(f"[INVESTOR] {ticker} 투자자별 매매동향 수집")

    path = "/uapi/domestic-stock/v1/quotations/inquire-investor"
    tr_id = "FHKST01010900"

    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": ticker,
    }

    data = client.get(path, tr_id, params)
    output = data.get("output", data.get("output1", []))

    if not output:
        log.warning("[INVESTOR] 데이터 없음")
        return pd.DataFrame()

    records = []
    for item in output:
        date_str = item.get("stck_bsop_date", "")
        if not date_str:
            continue

        records.append({
            "date": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}",
            "foreign_net_qty": _safe_int(item.get("frgn_ntby_qty", 0)),
            "foreign_net_amt": _safe_int(item.get("frgn_ntby_tr_pbmn", 0)),
            "inst_net_qty": _safe_int(item.get("orgn_ntby_qty", 0)),
            "inst_net_amt": _safe_int(item.get("orgn_ntby_tr_pbmn", 0)),
            "individual_net_qty": _safe_int(item.get("prsn_ntby_qty", 0)),
            "individual_net_amt": _safe_int(item.get("prsn_ntby_tr_pbmn", 0)),
        })

    df = pd.DataFrame(records)
    log.info(f"[INVESTOR] {len(df)}일치 수급 데이터 수집 완료")
    return df


# ═══════════════════════════════════════════════════════════════════
# 4. 공매도 일별 추이
# ═══════════════════════════════════════════════════════════════════

def fetch_short_selling(
    client: KISClient,
    ticker: str = TICKER,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    국내주식 공매도 일별추이 (FHKST03060100)

    ※ 한투 API 포털에서 정확한 경로/파라미터를 확인 필요.
       아래는 문서 기반 추정. 실제 호출 시 응답 구조 확인 후 조정.
    """
    if end is None:
        end = datetime.now().strftime("%Y%m%d")
    if start is None:
        start = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
    start = start.replace("-", "")
    end = end.replace("-", "")

    log.info(f"[SHORT] {ticker} 공매도 일별추이: {start} ~ {end}")

    # 경로 후보 (한투 포털에서 확인 필요 — 아래 2개 중 하나)
    path = "/uapi/domestic-stock/v1/quotations/inquire-daily-shortselling-by-stock"
    tr_id = "FHKST03060100"

    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": ticker,
        "FID_INPUT_DATE_1": start,
        "FID_INPUT_DATE_2": end,
    }

    try:
        data = client.get(path, tr_id, params)
    except Exception as e:
        log.warning(f"[SHORT] 1차 경로 실패: {e}")
        # 대체 경로 시도
        path = "/uapi/domestic-stock/v1/quotations/inquire-daily-short-selling"
        try:
            data = client.get(path, tr_id, params)
        except Exception as e2:
            log.error(f"[SHORT] 2차 경로도 실패: {e2}")
            return pd.DataFrame()

    output = data.get("output", data.get("output1", []))

    if not output:
        rt_cd = data.get("rt_cd", "")
        msg = data.get("msg1", "")
        log.warning(f"[SHORT] 데이터 없음 (rt_cd={rt_cd}, msg={msg})")
        log.info("[SHORT] → 한투 포털에서 정확한 엔드포인트 확인 필요")
        return pd.DataFrame()

    records = []
    for item in output:
        date_str = item.get("stck_bsop_date", "")
        if not date_str:
            continue

        short_vol = _safe_int(item.get("seld_cntg_qty",
                    item.get("shrt_sell_qty",
                    item.get("cntg_vol", 0))))
        total_vol = _safe_int(item.get("acml_vol",
                    item.get("total_vol", 0)))

        ratio = 0.0
        if total_vol > 0:
            ratio = round(short_vol / total_vol * 100, 2)

        records.append({
            "date": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}",
            "short_volume": short_vol,
            "total_volume": total_vol,
            "short_ratio": ratio,
        })

    df = pd.DataFrame(records)
    log.info(f"[SHORT] {len(df)}일치 수집 완료")
    return df


# ═══════════════════════════════════════════════════════════════════
# 5. ETF 프록시 (OHLCV와 동일 방식)
# ═══════════════════════════════════════════════════════════════════

def fetch_etf_proxy(
    client: KISClient,
    start: str = HIST_START,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """KODEX 2차전지산업 ETF OHLCV (섹터 자금 유입 프록시)."""
    log.info(f"[ETF] {ETF_TICKER} (KODEX 2차전지) 수집")
    try:
        df = fetch_ohlcv(client, ticker=ETF_TICKER, start=start, end=end)
    except Exception as e:
        log.warning(f"[ETF] 수집 실패: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    # 거래량 이동평균 및 프록시 지표
    df["vol_ma20"] = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = np.where(
        df["vol_ma20"] > 0,
        (df["Volume"] / df["vol_ma20"]).round(3),
        1.0,
    )
    df["is_bullish"] = (df["Close"] > df["Open"]).astype(int)

    log.info(f"[ETF] {len(df)}일치 수집 완료")
    return df


# ═══════════════════════════════════════════════════════════════════
# 6. 상태 관리
# ═══════════════════════════════════════════════════════════════════

def load_state(filepath: Path = PIPELINE_STATE_FILE) -> dict:
    default_state = {
        "last_ohlcv_update": None,
        "last_supply_update": None,
        "last_short_update": None,
        "total_ohlcv_rows": 0,
        "total_supply_rows": 0,
        "data_source": "KIS_OpenAPI",
        "errors": [],
        "created_at": datetime.now().isoformat(),
    }
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            log.warning(f"[STATE] 파일 손상, 초기화")
    return default_state


def save_state(state: dict, filepath: Path = PIPELINE_STATE_FILE) -> None:
    state["updated_at"] = datetime.now().isoformat()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, default=str)


def save_supply_data(new_data: dict, filepath: Path = SUPPLY_FILE) -> None:
    """수급 데이터를 날짜별로 누적 저장 (update 방식으로 필드 병합)."""
    existing = {}
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            pass

    for date_key, values in new_data.items():
        if date_key in existing:
            existing[date_key].update(values)
        else:
            existing[date_key] = values

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"[SUPPLY] {len(new_data)}일치 추가/갱신 → 총 {len(existing)}일치")


def save_ohlcv_data(df: pd.DataFrame, filepath: Path = OHLCV_FILE) -> None:
    data = {}
    for idx, row in df.iterrows():
        date_key = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
        data[date_key] = {
            "Open": int(row.get("Open", 0)),
            "High": int(row.get("High", 0)),
            "Low": int(row.get("Low", 0)),
            "Close": int(row.get("Close", 0)),
            "Volume": int(row.get("Volume", 0)),
            "prev_Open": int(row.get("prev_Open", 0)),
            "prev_High": int(row.get("prev_High", 0)),
            "prev_Low": int(row.get("prev_Low", 0)),
            "prev_Close": int(row.get("prev_Close", 0)),
            "prev_Volume": int(row.get("prev_Volume", 0)),
        }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info(f"[OHLCV] {len(data)}일치 저장: {filepath}")


# ═══════════════════════════════════════════════════════════════════
# 7. 유틸리티
# ═══════════════════════════════════════════════════════════════════

def _safe_int(val) -> int:
    if val is None or val == "":
        return 0
    try:
        return int(float(str(val).replace(",", "")))
    except (ValueError, TypeError):
        return 0


def send_telegram(message: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        log.warning(f"[TELEGRAM] 오류: {e}")


# ═══════════════════════════════════════════════════════════════════
# 8. 메인 파이프라인
# ═══════════════════════════════════════════════════════════════════

def run_daily_collection(mode: str = "incremental") -> dict:
    """
    일일 데이터 수집 파이프라인 (한투 OpenAPI 전용).

    mode:
      "full"        — 전체 기간 OHLCV 재수집 (2022~현재)
      "incremental" — OHLCV 최근 30일 + 수급/공매도 최근 데이터
    """
    log.info(f"{'='*60}")
    log.info(f"[PIPELINE] {TICKER_NAME} 데이터 수집 (한투 OpenAPI)")
    log.info(f"  mode={mode}")
    log.info(f"{'='*60}")

    state = load_state()
    result = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "source": "KIS_OpenAPI",
        "ohlcv": {"status": "skip", "rows": 0},
        "supply": {"status": "skip", "rows": 0},
        "short": {"status": "skip", "rows": 0},
        "etf": {"status": "skip", "rows": 0},
        "errors": [],
    }

    # ── 한투 클라이언트 초기화 ──
    try:
        client = KISClient()
        if not client.access_token:
            raise RuntimeError("access_token 발급 실패")
    except Exception as e:
        log.error(f"[AUTH] 인증 실패: {e}")
        result["errors"].append(f"AUTH: {e}")
        _send_report(result)
        return result

    if mode == "full":
        ohlcv_start = HIST_START
    else:
        ohlcv_start = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")

    # ── 1) OHLCV ──
    try:
        df_ohlcv = fetch_ohlcv(client, TICKER, start=ohlcv_start)
        save_ohlcv_data(df_ohlcv)
        result["ohlcv"] = {"status": "ok", "rows": len(df_ohlcv)}
        state["last_ohlcv_update"] = datetime.now().strftime("%Y-%m-%d")
        state["total_ohlcv_rows"] = len(df_ohlcv)
    except Exception as e:
        log.error(f"[OHLCV] 실패: {e}")
        result["ohlcv"] = {"status": "error", "error": str(e)}
        result["errors"].append(f"OHLCV: {e}")

    # ── 2) 투자자별 매매동향 ──
    try:
        df_inv = fetch_investor_trading(client, TICKER)
        if not df_inv.empty:
            supply_dict = {}
            for _, row in df_inv.iterrows():
                supply_dict[row["date"]] = {
                    k: v for k, v in row.items() if k != "date"
                }
            save_supply_data(supply_dict)
            result["supply"] = {"status": "ok", "rows": len(df_inv)}
            state["last_supply_update"] = datetime.now().strftime("%Y-%m-%d")
            state["total_supply_rows"] = len(df_inv)
    except Exception as e:
        log.error(f"[INVESTOR] 실패: {e}")
        result["supply"] = {"status": "error", "error": str(e)}
        result["errors"].append(f"INVESTOR: {e}")

    # ── 3) 공매도 ──
    try:
        short_start = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
        df_short = fetch_short_selling(client, TICKER, start=short_start)
        if not df_short.empty:
            short_dict = {}
            for _, row in df_short.iterrows():
                short_dict[row["date"]] = {
                    k: v for k, v in row.items() if k != "date"
                }
            save_supply_data(short_dict)
            result["short"] = {"status": "ok", "rows": len(df_short)}
            state["last_short_update"] = datetime.now().strftime("%Y-%m-%d")
        else:
            log.info("[SHORT] 데이터 없음 — 한투 포털에서 경로 확인 필요")
            result["short"] = {"status": "no_data", "rows": 0}
    except Exception as e:
        log.error(f"[SHORT] 실패: {e}")
        result["short"] = {"status": "error", "error": str(e)}
        result["errors"].append(f"SHORT: {e}")

    # ── 4) ETF 프록시 ──
    try:
        df_etf = fetch_etf_proxy(client, start=ohlcv_start)
        result["etf"] = {"status": "ok", "rows": len(df_etf)}
    except Exception as e:
        log.error(f"[ETF] 실패: {e}")
        result["etf"] = {"status": "error", "error": str(e)}
        result["errors"].append(f"ETF: {e}")

    # ── 저장 & 알림 ──
    state["data_source"] = "KIS_OpenAPI"
    state["errors"] = result["errors"][-10:]
    save_state(state)
    _send_report(result)

    return result


def _send_report(result: dict) -> None:
    emoji = {"ok": "✅", "error": "❌", "skip": "⏭️", "no_data": "⚠️"}
    labels = {"ohlcv": "OHLCV", "supply": "수급", "short": "공매도", "etf": "ETF"}

    lines = [
        f"📊 <b>{TICKER_NAME} 데이터 수집</b>",
        f"소스: 한투 OpenAPI",
        f"모드: {result['mode']}",
        "",
    ]
    for key, label in labels.items():
        d = result.get(key, {"status": "skip"})
        e = emoji.get(d["status"], "❓")
        lines.append(f"{e} {label}: {d.get('rows', 0)}건")

    if result["errors"]:
        lines.append("\n⚠️ 오류:")
        for err in result["errors"]:
            lines.append(f"  · {err[:80]}")

    send_telegram("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════
# 9. CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=f"{TICKER_NAME} ({TICKER}) 데이터 수집 (한투 OpenAPI)"
    )
    parser.add_argument(
        "--mode", choices=["full", "incremental"],
        default="incremental",
    )
    parser.add_argument("--state-dir", type=str, default=".")
    args = parser.parse_args()

    STATE_DIR = Path(args.state_dir)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    OHLCV_FILE = STATE_DIR / "ohlcv_247540.json"
    SUPPLY_FILE = STATE_DIR / "supply_data_247540.json"
    PIPELINE_STATE_FILE = STATE_DIR / "pipeline_state_247540.json"

    result = run_daily_collection(mode=args.mode)
    exit(1 if result["errors"] else 0)
