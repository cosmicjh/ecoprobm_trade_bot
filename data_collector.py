"""
에코프로비엠 (247540) v4 자동트레이딩 — Phase 1-1: 데이터 수집 (v3.1 hotfix)
================================================================================
v3 → v3.1 변경:
  - mojito2 토큰 발급 완전 제거 → 직접 실전 서버에서 토큰 발급
  - 공매도 API: 설명서 기반으로 정확한 경로/tr_id/파라미터 적용
    URL:   /uapi/domestic-stock/v1/quotations/daily-short-sale
    tr_id: FHPST04830000 (실전 전용, 모의투자 미지원)
  - 날짜 포맷: YYYY-MM-DD (length=10, 하이픈 포함)
  - custtype: "P" 헤더 필수 추가
  
원인: mojito2가 내부적으로 캐시하는 토큰이 모의투자 서버용일 수 있음.
      공매도 API는 실전 서버 전용이므로 모의 토큰 → "유효하지 않은 token"
      → mojito2를 토큰 발급에서 완전 배제하여 해결
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

# ── 설정 ─────────────────────────────────────────────────────────
TICKER = "247540"
TICKER_NAME = "에코프로비엠"
ETF_TICKER = "305720"

# 한투 API — 실전 서버 고정 (공매도 API가 모의투자 미지원이므로)
KIS_BASE_URL = "https://openapi.koreainvestment.com:9443"
API_CALL_DELAY = 0.5

HIST_START = "2022-01-01"  # 하이픈 포함 (한투 API 날짜 포맷)

STATE_DIR = Path(os.getenv("STATE_DIR", "."))
OHLCV_FILE = STATE_DIR / "ohlcv_247540.json"
SUPPLY_FILE = STATE_DIR / "supply_data_247540.json"
PIPELINE_STATE_FILE = STATE_DIR / "pipeline_state_247540.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 1. 한투 API 클라이언트 (mojito2 완전 배제)
# ═══════════════════════════════════════════════════════════════════

class KISClient:
    """
    한투 OpenAPI 클라이언트.
    mojito2 없이 직접 실전 서버에서 토큰 발급.
    """

    def __init__(self):
        self.api_key = os.getenv("KIS_API_KEY", "")
        self.api_secret = os.getenv("KIS_API_SECRET", "")
        self.acc_no = os.getenv("KIS_ACC_NO", "")
        self.base_url = KIS_BASE_URL  # 항상 실전 서버
        self.access_token = None

        if not self.api_key or not self.api_secret:
            raise ValueError("KIS_API_KEY, KIS_API_SECRET 환경변수 필요")

        self._get_token()

    def _get_token(self):
      """토큰 발급 (캐시 재사용 지원)."""
      from kis_token_store import get_or_refresh_token
      self.access_token = get_or_refresh_token(
        api_key=self.api_key,
        api_secret=self.api_secret,
        base_url=self.base_url,
        state_dir=str(STATE_DIR),
      )
    def get(self, path: str, tr_id: str, params: dict,
        extra_headers: Optional[dict] = None) -> dict:
      """한투 API GET 요청. 토큰 만료 시 1회 자동 재발급."""
      from kis_token_store import is_token_error, invalidate_cache

      def _do_request():
        url = self.base_url + path
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.api_key,
            "appsecret": self.api_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }
        if extra_headers:
            headers.update(extra_headers)
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        try:
            rj = resp.json()
        except Exception:
            rj = {}
        return resp, rj

      # 1차 시도
      resp, data = _do_request()

      # 401/토큰만료 → 재발급 후 1회 재시도
      if is_token_error(resp.status_code, data):
        log.warning(f"[KIS] 토큰 거부 (status={resp.status_code}, "
                    f"msg_cd={data.get('msg_cd','')}), 재발급 후 재시도")
        invalidate_cache(str(STATE_DIR))
        self._get_token()
        resp, data = _do_request()

        if is_token_error(resp.status_code, data):
            raise RuntimeError(f"[KIS] 재인증 실패: {data}")

      # 기존 rt_cd 체크 유지
      rt_cd = data.get("rt_cd", "")
      if rt_cd != "0":
        msg = data.get("msg1", "Unknown")
        log.warning(f"[KIS] API 오류 (tr_id={tr_id}): rt_cd={rt_cd}, msg={msg}")

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
) -> pd.DataFrame:
    """
    국내주식기간별시세 (FHKST03010100)
    100건씩 페이징, .shift(1) 적용.
    """
    if end is None:
        end = datetime.now().strftime("%Y%m%d")

    # 이 API는 YYYYMMDD 포맷
    _start = start.replace("-", "")
    _end = end.replace("-", "")

    log.info(f"[OHLCV] {ticker}: {_start} ~ {_end}")

    path = "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
    tr_id = "FHKST03010100"

    all_rows = []
    current_end = _end

    for page in range(50):
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": ticker,
            "FID_INPUT_DATE_1": _start,
            "FID_INPUT_DATE_2": current_end,
            "FID_PERIOD_DIV_CODE": "D",
            "FID_ORG_ADJ_PRC": "0",
        }

        data = client.get(path, tr_id, params)
        output = data.get("output2", [])
        if not output:
            break

        for item in output:
            d = item.get("stck_bsop_date", "")
            if not d or d < _start:
                continue
            all_rows.append({
                "date": d,
                "Open": _safe_int(item.get("stck_oprc", 0)),
                "High": _safe_int(item.get("stck_hgpr", 0)),
                "Low": _safe_int(item.get("stck_lwpr", 0)),
                "Close": _safe_int(item.get("stck_clpr", 0)),
                "Volume": _safe_int(item.get("acml_vol", 0)),
            })

        last = output[-1].get("stck_bsop_date", "")
        if not last or last <= _start:
            break
        try:
            current_end = (datetime.strptime(last, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
        except ValueError:
            break
        if current_end < _start:
            break

    if not all_rows:
        raise ValueError(f"[OHLCV] {ticker}: 데이터 없음")

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.drop_duplicates(subset="date").sort_values("date").set_index("date")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[f"prev_{col}"] = df[col].shift(1)
    df = df.iloc[1:].copy()

    log.info(f"[OHLCV] {len(df)}일치 완료")
    return df


# ═══════════════════════════════════════════════════════════════════
# 3. 투자자별 매매동향
# ═══════════════════════════════════════════════════════════════════

def fetch_investor_trading(
    client: KISClient,
    ticker: str = TICKER,
) -> pd.DataFrame:
    """주식현재가 투자자 (FHKST01010900)."""
    log.info(f"[INVESTOR] {ticker} 수급 수집")

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
        d = item.get("stck_bsop_date", "")
        if not d:
            continue
        records.append({
            "date": f"{d[:4]}-{d[4:6]}-{d[6:]}",
            "foreign_net_qty": _safe_int(item.get("frgn_ntby_qty", 0)),
            "foreign_net_amt": _safe_int(item.get("frgn_ntby_tr_pbmn", 0)),
            "inst_net_qty": _safe_int(item.get("orgn_ntby_qty", 0)),
            "inst_net_amt": _safe_int(item.get("orgn_ntby_tr_pbmn", 0)),
            "individual_net_qty": _safe_int(item.get("prsn_ntby_qty", 0)),
            "individual_net_amt": _safe_int(item.get("prsn_ntby_tr_pbmn", 0)),
        })

    df = pd.DataFrame(records)
    log.info(f"[INVESTOR] {len(df)}일치 완료")
    return df


# ═══════════════════════════════════════════════════════════════════
# 4. 공매도 일별 추이 (설명서 기반 정확한 스펙)
# ═══════════════════════════════════════════════════════════════════

def fetch_short_selling(
    client: KISClient,
    ticker: str = TICKER,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    국내주식 공매도 일별추이 (국내주식-134)
    
    설명서 스펙:
      tr_id:  FHPST04830000 (실전 전용, 모의투자 미지원)
      URL:    /uapi/domestic-stock/v1/quotations/daily-short-sale
      날짜:   YYYY-MM-DD (length=10, 하이픈 포함)
      custtype: P (개인) — 헤더 필수

    output2 주요 필드:
      stck_bsop_date:       영업일자 (YYYYMMDD)
      ssts_cntg_qty:        공매도 체결 수량
      ssts_vol_rlim:        공매도 거래량 비중
      acml_vol:             누적 거래량
      ssts_tr_pbmn:         공매도 거래 대금
      ssts_tr_pbmn_rlim:    공매도 거래대금 비중
    """
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    if start is None:
        start = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

    # ★ 날짜 포맷: YYYY-MM-DD (하이픈 포함, length=10)
    # 하이픈이 없으면 추가
    if len(start) == 8:
        start = f"{start[:4]}-{start[4:6]}-{start[6:]}"
    if len(end) == 8:
        end = f"{end[:4]}-{end[4:6]}-{end[6:]}"

    log.info(f"[SHORT] {ticker} 공매도: {start} ~ {end}")

    # ★ 설명서 기반 정확한 경로 & tr_id
    path = "/uapi/domestic-stock/v1/quotations/daily-short-sale"
    tr_id = "FHPST04830000"

    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": ticker,
        "FID_INPUT_DATE_1": start,
        "FID_INPUT_DATE_2": end,
    }

    data = client.get(path, tr_id, params)

    rt_cd = data.get("rt_cd", "")
    if rt_cd != "0":
        msg = data.get("msg1", "")
        log.error(f"[SHORT] API 실패: rt_cd={rt_cd}, msg={msg}")
        return pd.DataFrame()

    output = data.get("output2", [])
    if not output:
        log.warning("[SHORT] output2 비어있음")
        return pd.DataFrame()

    records = []
    for item in output:
        d = item.get("stck_bsop_date", "")
        if not d:
            continue

        short_qty = _safe_int(item.get("ssts_cntg_qty", 0))
        total_vol = _safe_int(item.get("acml_vol", 0))
        short_ratio_vol = item.get("ssts_vol_rlim", "0")
        short_amt = _safe_int(item.get("ssts_tr_pbmn", 0))
        short_ratio_amt = item.get("ssts_tr_pbmn_rlim", "0")

        records.append({
            "date": f"{d[:4]}-{d[4:6]}-{d[6:]}",
            "short_volume": short_qty,
            "total_volume": total_vol,
            "short_ratio_vol": _safe_float(short_ratio_vol),
            "short_amount": short_amt,
            "short_ratio_amt": _safe_float(short_ratio_amt),
        })

    df = pd.DataFrame(records)
    log.info(f"[SHORT] {len(df)}일치 공매도 데이터 수집 완료")
    return df


# ═══════════════════════════════════════════════════════════════════
# 5. ETF 프록시
# ═══════════════════════════════════════════════════════════════════

def fetch_etf_proxy(
    client: KISClient,
    start: str = HIST_START,
    end: Optional[str] = None,
) -> pd.DataFrame:
    log.info(f"[ETF] {ETF_TICKER} 수집")
    try:
        df = fetch_ohlcv(client, ticker=ETF_TICKER, start=start, end=end)
    except Exception as e:
        log.warning(f"[ETF] 실패: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    df["vol_ma20"] = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = np.where(
        df["vol_ma20"] > 0, (df["Volume"] / df["vol_ma20"]).round(3), 1.0)
    df["is_bullish"] = (df["Close"] > df["Open"]).astype(int)
    return df


# ═══════════════════════════════════════════════════════════════════
# 6. 상태 관리 & 유틸리티
# ═══════════════════════════════════════════════════════════════════

def load_state(filepath: Path = PIPELINE_STATE_FILE) -> dict:
    default = {
        "last_ohlcv_update": None, "last_supply_update": None,
        "last_short_update": None, "total_ohlcv_rows": 0,
        "total_supply_rows": 0, "data_source": "KIS_OpenAPI_prod",
        "errors": [], "created_at": datetime.now().isoformat(),
    }
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return default


def save_state(state: dict, filepath: Path = PIPELINE_STATE_FILE) -> None:
    state["updated_at"] = datetime.now().isoformat()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, default=str)


def save_supply_data(new_data: dict, filepath: Path = SUPPLY_FILE) -> None:
    existing = {}
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            pass
    for k, v in new_data.items():
        if k in existing:
            existing[k].update(v)
        else:
            existing[k] = v
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"[SUPPLY] {len(new_data)}일치 → 총 {len(existing)}일치")


def save_ohlcv_data(df: pd.DataFrame, filepath: Path = OHLCV_FILE) -> None:
    data = {}
    for idx, row in df.iterrows():
        dk = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
        data[dk] = {
            "Open": int(row.get("Open", 0)), "High": int(row.get("High", 0)),
            "Low": int(row.get("Low", 0)), "Close": int(row.get("Close", 0)),
            "Volume": int(row.get("Volume", 0)),
            "prev_Open": int(row.get("prev_Open", 0)),
            "prev_High": int(row.get("prev_High", 0)),
            "prev_Low": int(row.get("prev_Low", 0)),
            "prev_Close": int(row.get("prev_Close", 0)),
            "prev_Volume": int(row.get("prev_Volume", 0)),
        }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info(f"[OHLCV] {len(data)}일치 저장")


def _safe_int(val) -> int:
    if val is None or val == "":
        return 0
    try:
        return int(float(str(val).replace(",", "")))
    except (ValueError, TypeError):
        return 0


def _safe_float(val) -> float:
    if val is None or val == "":
        return 0.0
    try:
        return float(str(val).replace(",", ""))
    except (ValueError, TypeError):
        return 0.0


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
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════
# 7. 메인 파이프라인
# ═══════════════════════════════════════════════════════════════════

def run_daily_collection(mode: str = "incremental") -> dict:
    log.info(f"{'='*60}")
    log.info(f"[PIPELINE] {TICKER_NAME} 수집 (한투 실전 서버 직접)")
    log.info(f"  mode={mode}, base_url={KIS_BASE_URL}")
    log.info(f"{'='*60}")

    state = load_state()
    result = {
        "timestamp": datetime.now().isoformat(), "mode": mode,
        "source": "KIS_OpenAPI_prod",
        "ohlcv": {"status": "skip", "rows": 0},
        "supply": {"status": "skip", "rows": 0},
        "short": {"status": "skip", "rows": 0},
        "etf": {"status": "skip", "rows": 0},
        "errors": [],
    }

    try:
        client = KISClient()
    except Exception as e:
        log.error(f"[AUTH] 실패: {e}")
        result["errors"].append(f"AUTH: {e}")
        _send_report(result)
        return result

    if mode == "full":
        ohlcv_start = HIST_START
    else:
        ohlcv_start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    # 1) OHLCV
    try:
        df = fetch_ohlcv(client, TICKER, start=ohlcv_start)
        save_ohlcv_data(df)
        result["ohlcv"] = {"status": "ok", "rows": len(df)}
        state["last_ohlcv_update"] = datetime.now().strftime("%Y-%m-%d")
        state["total_ohlcv_rows"] = len(df)
    except Exception as e:
        log.error(f"[OHLCV] 실패: {e}")
        result["ohlcv"] = {"status": "error", "error": str(e)}
        result["errors"].append(f"OHLCV: {e}")

    # 2) 투자자별 매매동향
    try:
        df = fetch_investor_trading(client, TICKER)
        if not df.empty:
            sd = {row["date"]: {k: v for k, v in row.items() if k != "date"}
                  for _, row in df.iterrows()}
            save_supply_data(sd)
            result["supply"] = {"status": "ok", "rows": len(df)}
            state["last_supply_update"] = datetime.now().strftime("%Y-%m-%d")
    except Exception as e:
        log.error(f"[INVESTOR] 실패: {e}")
        result["supply"] = {"status": "error", "error": str(e)}
        result["errors"].append(f"INVESTOR: {e}")

    # 3) 공매도
    try:
        short_start = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        df = fetch_short_selling(client, TICKER, start=short_start)
        if not df.empty:
            sd = {row["date"]: {k: v for k, v in row.items() if k != "date"}
                  for _, row in df.iterrows()}
            save_supply_data(sd)
            result["short"] = {"status": "ok", "rows": len(df)}
            state["last_short_update"] = datetime.now().strftime("%Y-%m-%d")
        else:
            result["short"] = {"status": "no_data", "rows": 0}
    except Exception as e:
        log.error(f"[SHORT] 실패: {e}")
        result["short"] = {"status": "error", "error": str(e)}
        result["errors"].append(f"SHORT: {e}")

    # 4) ETF
    try:
        df = fetch_etf_proxy(client, start=ohlcv_start)
        result["etf"] = {"status": "ok", "rows": len(df)}
    except Exception as e:
        log.error(f"[ETF] 실패: {e}")
        result["etf"] = {"status": "error", "error": str(e)}
        result["errors"].append(f"ETF: {e}")

    state["data_source"] = "KIS_OpenAPI_prod"
    state["errors"] = result["errors"][-10:]
    save_state(state)
    _send_report(result)
    return result


def _send_report(result: dict) -> None:
    emoji = {"ok": "✅", "error": "❌", "skip": "⏭️", "no_data": "⚠️"}
    labels = {"ohlcv": "OHLCV", "supply": "수급", "short": "공매도", "etf": "ETF"}
    lines = [f"📊 <b>{TICKER_NAME} 수집 (실전서버 직접)</b>",
             f"모드: {result['mode']}", ""]
    for k, l in labels.items():
        d = result.get(k, {"status": "skip"})
        lines.append(f"{emoji.get(d['status'], '❓')} {l}: {d.get('rows', 0)}건")
    if result["errors"]:
        lines.append("\n⚠️ 오류:")
        for e in result["errors"]:
            lines.append(f"  · {e[:80]}")
    send_telegram("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(...)  # 기존 그대로
    parser.add_argument("--mode", choices=["incremental", "full"], default="incremental")
    parser.add_argument("--state-dir", type=str, default="state")
    args = parser.parse_args()

    # 모듈 레벨 경로 상수를 CLI 인자로 교체
    STATE_DIR = Path(args.state_dir)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    OHLCV_FILE = STATE_DIR / "ohlcv_247540.json"
    SUPPLY_FILE = STATE_DIR / "supply_data_247540.json"
    PIPELINE_STATE_FILE = STATE_DIR / "pipeline_state_247540.json"

    # 함수 기본값에도 새 경로가 반영되도록 globals 업데이트
    globals()["STATE_DIR"] = STATE_DIR
    globals()["OHLCV_FILE"] = OHLCV_FILE
    globals()["SUPPLY_FILE"] = SUPPLY_FILE
    globals()["PIPELINE_STATE_FILE"] = PIPELINE_STATE_FILE

    run_daily_collection(mode=args.mode)
