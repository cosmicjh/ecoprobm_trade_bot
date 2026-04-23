"""
에코프로비엠 (247540) v4 — Phase 2-1: 자동트레이딩 봇 (+피라미딩)
======================================================
3-Layer 전략 실행 봇:
  Layer 1: 레짐 판별 (MA/BB/ATR 기반)
  Layer 2: 수급 시그널 (외국인·기관 순매수, 공매도)
  Layer 3: 레짐별 실행 전략 (TREND_UP/RANGE_BOUND/HIGH_VOL)
  Layer 4: 피라미딩 (1차 익절 후 수익 상태 + 강한 수급 시 추가 매수)

인프라: GitHub Actions + cron-job.org (workflow_dispatch)
  09:02 KST (00:02 UTC) — 장 시작: 시그널 판단 + 주문 실행
  15:35 KST (06:35 UTC) — 장 마감: 당일 체결 확인 + 상태 갱신
  18:30 KST (09:30 UTC) — 장 후: 확정 데이터 수집 + 지표 갱신

데이터: KISClient (토큰 매번 발급, 실전 서버)
"""

import os
import json
import logging
import requests
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional
from time import sleep

import numpy as np
import pandas as pd

from ai_layer import AILayer, ensemble_regime

from risk_monitor import assess_risk, adjust_invest_ratio, should_skip_entry, format_risk_telegram, save_risk_history

from order_manager import (
    PendingOrder,
    create_pending_from_response,
    handle_unfilled_orders,
)

from reporter import (
    log_trade,
    format_daily_report,
    format_weekly_report,
    format_monthly_report,
    should_send_weekly_report,
    should_send_monthly_report,
)

from accuracy_tracker import record_prediction, evaluate_pending_predictions

from news_sentiment import get_sentiment_signal, format_sentiment_telegram

# ── 로깅 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# 1. 설정 & 파라미터
# ═══════════════════════════════════════════════════════════════════

TICKER = "247540"
TICKER_NAME = "에코프로비엠"
MARKET = "KOSDAQ"

KIS_BASE_URL = "https://openapi.koreainvestment.com:9443"

# 파일 경로
STATE_DIR = Path(os.getenv("STATE_DIR", "state"))

# KRX 호가 단위 테이블 (코스닥)
KOSDAQ_TICK_TABLE = [
    (2_000,     1),
    (5_000,     5),
    (20_000,    10),
    (50_000,    50),
    (200_000,   100),
    (500_000,   500),
    (float('inf'), 1000),
]


@dataclass
class StrategyParams:
    """백테스트 최적화된 파라미터. Colab에서 내보낸 JSON으로 덮어씀."""
    # Layer 1: 레짐
    ma_short: int = 10
    ma_long: int = 80
    bb_squeeze_threshold: float = 0.8
    atr_hvol_threshold: float = 1.5

    # Layer 3A: TREND_UP
    tp1_pct: float = 0.11
    tp1_sell_ratio: float = 0.6
    trail_atr_mult: float = 1.75
    sl_pct: float = -0.04
    cooldown_days: int = 1

    # Layer 3B: RANGE_BOUND
    rsi_entry: float = 25.0
    sl_band_buffer: float = -0.02

    # Layer 3C: HIGH_VOL
    hvol_size_reduction: float = 0.5

    # 공통
    invest_ratio: float = 0.30
    max_invest_ratio: float = 0.60

    # 수급
    supply_lookback: int = 5
    supply_threshold: float = 2.0

    # 리스크 관리
    daily_loss_limit: float = -0.03    # 일일 -3%
    weekly_loss_limit: float = -0.07   # 주간 -7%
    monthly_loss_limit: float = -0.12  # 월간 -12%

    # Layer 4: 피라미딩 (추가 매수)
    enable_pyramiding: bool = True
    max_pyramiding: int = 2                      # 추가 매수 최대 횟수 (초기 매수 제외)
    pyramiding_min_profit: float = 0.02          # 현재가 - 평균진입가 이익률 최소치 (+2%)
    pyramiding_size_ratio: float = 0.5           # 추가 매수 규모 (최초 invest의 50%)

    # Layer 5: Isolation Forest 수급 이상 시그널 반영 — 신규
    # anomaly_score 기준점 (decision_function 출력: 낮을수록 이상)
    anomaly_strong_threshold: float = -0.20      # 이 값 이하면 강한 이상
    anomaly_moderate_threshold: float = -0.10    # 이 값 이하면 중간 이상
    anomaly_strong_bearish_multiplier: float = 0.0    # 0.0 = 진입 차단
    anomaly_moderate_bearish_multiplier: float = 0.7  # 포지션 30% 축소
    anomaly_moderate_bullish_multiplier: float = 1.15 # 포지션 15% 확대
    anomaly_strong_bullish_multiplier: float = 1.25   # 포지션 25% 확대


@dataclass
class BotState:
    """봇 상태 (JSON으로 persist)."""
    # 포지션
    position_qty: int = 0
    entry_price: float = 0.0
    entry_date: str = ""
    highest_since_entry: float = 0.0
    tp1_done: bool = False           # 1차 익절 완료 여부

    # 자금
    cash: float = 1_500_000.0
    initial_capital: float = 1_500_000.0

    # 리스크
    cooldown_until: str = ""         # 쿨다운 만료일 (YYYY-MM-DD)
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    last_trade_date: str = ""
    last_week_reset: str = ""
    last_month_reset: str = ""
    halted: bool = False             # 리스크 한도 초과 시 거래 중단
    halt_reason: str = ""

    # 메타
    last_regime: str = ""
    last_signal: str = ""
    last_run: str = ""
    total_trades: int = 0
    version: str = "v4.3.0"          # 피라미딩 반영

    # 미체결 주문 추적 리스트 (PendingOrder의 dict 형태로 저장)
    pending_orders: list = field(default_factory=list)

    # 피라미딩 추적 (신규)
    pyramiding_count: int = 0                                     # 현재 포지션에서 추가 매수한 횟수
    pyramiding_history: list = field(default_factory=list)        # [{date, price, qty, trigger, new_avg_entry}, ...]


def load_params(path: Optional[str] = None) -> StrategyParams:
    """Colab 최적화 결과 JSON에서 파라미터 로드."""
    if path is None:
        path = STATE_DIR / "optimized_params_247540.json"
    else:
        path = Path(path)

    if path.exists():
        with open(path, "r") as f:
            data = json.load(f)
        p = data.get("params", {})
        log.info(f"[PARAMS] 최적화 파라미터 로드: {path}")
        return StrategyParams(
            ma_short=p.get("ma_short", 10),
            ma_long=p.get("ma_long", 80),
            bb_squeeze_threshold=p.get("bb_squeeze_threshold", 0.8),
            atr_hvol_threshold=p.get("atr_hvol_threshold", 1.5),
            tp1_pct=p.get("tp1_pct", 0.11),
            tp1_sell_ratio=p.get("tp1_sell_ratio", 0.6),
            trail_atr_mult=p.get("trail_atr_mult", 1.75),
            sl_pct=p.get("sl_pct", -0.04),
            cooldown_days=p.get("cooldown_days", 1),
            rsi_entry=p.get("rsi_entry", 25.0),
            invest_ratio=p.get("invest_ratio", 0.30),
            max_invest_ratio=p.get("max_invest_ratio", 0.60),
            hvol_size_reduction=p.get("hvol_size_reduction", 0.5),
            daily_loss_limit=p.get("daily_loss_limit", -0.03),
            weekly_loss_limit=p.get("weekly_loss_limit", -0.07),
            monthly_loss_limit=p.get("monthly_loss_limit", -0.12),
            # 피라미딩 파라미터
            enable_pyramiding=p.get("enable_pyramiding", True),
            max_pyramiding=p.get("max_pyramiding", 2),
            pyramiding_min_profit=p.get("pyramiding_min_profit", 0.02),
            pyramiding_size_ratio=p.get("pyramiding_size_ratio", 0.5),
            # Isolation Forest 시그널 파라미터
            anomaly_strong_threshold=p.get("anomaly_strong_threshold", -0.20),
            anomaly_moderate_threshold=p.get("anomaly_moderate_threshold", -0.10),
            anomaly_strong_bearish_multiplier=p.get("anomaly_strong_bearish_multiplier", 0.0),
            anomaly_moderate_bearish_multiplier=p.get("anomaly_moderate_bearish_multiplier", 0.7),
            anomaly_moderate_bullish_multiplier=p.get("anomaly_moderate_bullish_multiplier", 1.15),
            anomaly_strong_bullish_multiplier=p.get("anomaly_strong_bullish_multiplier", 1.25),
        )
    else:
        log.warning(f"[PARAMS] 파일 없음, 기본 파라미터 사용")
        return StrategyParams()


def load_bot_state() -> BotState:
    path = STATE_DIR / "bot_state_247540.json"
    if path.exists():
        with open(path, "r") as f:
            data = json.load(f)
        s = BotState()
        for k, v in data.items():
            if hasattr(s, k):
                setattr(s, k, v)
        return s
    return BotState()


def save_bot_state(state: BotState):
    path = STATE_DIR / "bot_state_247540.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(state), f, ensure_ascii=False, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════
# 2. KIS API 클라이언트 (데이터 수집)
# ═══════════════════════════════════════════════════════════════════

class KISClient:
    """한투 OpenAPI 직접 호출 (실전 서버, 매 실행 토큰 신규 발급)."""

    def __init__(self):
        self.api_key = os.getenv("KIS_API_KEY", "")
        self.api_secret = os.getenv("KIS_API_SECRET", "")
        self.base_url = KIS_BASE_URL
        self.access_token = None
        self._get_token()

    def _get_token(self):
        from kis_token_store import get_or_refresh_token
        self.access_token = get_or_refresh_token(
            api_key=self.api_key,
            api_secret=self.api_secret,
            base_url=self.base_url,
            state_dir=str(STATE_DIR),
        )

    def get(self, path, tr_id, params, extra_headers=None):
        from kis_token_store import is_token_error, invalidate_cache

        def _do_request():
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
            url = self.base_url + path
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            try:
                rj = resp.json()
            except Exception:
                rj = {}
            return resp, rj

        # 1차 시도
        resp, data = _do_request()

        # 401/토큰만료 감지 → 재발급 후 1회 재시도
        if is_token_error(resp.status_code, data):
            log.warning(f"[KIS] 토큰 거부 감지 (status={resp.status_code}, "
                        f"msg_cd={data.get('msg_cd','')}), 재발급 후 재시도")
            invalidate_cache(str(STATE_DIR))
            self._get_token()
            resp, data = _do_request()
            if is_token_error(resp.status_code, data):
                raise RuntimeError(f"[KIS] 토큰 재발급 후에도 인증 실패: {data}")

        sleep(0.3)
        return data

    def get_hashkey(self, body):
        """POST 요청(주문)에 필요한 보안 해시키 발급"""
        url = f"{self.base_url}/uapi/hashkey"
        headers = {
            "content-type": "application/json",
            "appkey": self.api_key,
            "appsecret": self.api_secret,
        }
        resp = requests.post(url, headers=headers, json=body, timeout=10)
        return resp.json().get("HASH", "")

    def post(self, path, tr_id, body):
        """실전 주문용 POST 메서드"""
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.api_key,
            "appsecret": self.api_secret,
            "tr_id": tr_id,
            "custtype": "P",
            "hashkey": self.get_hashkey(body),
        }
        resp = requests.post(self.base_url + path, headers=headers, json=body, timeout=15)
        return resp.json()


# ═══════════════════════════════════════════════════════════════════
# 3. 시세 조회
# ═══════════════════════════════════════════════════════════════════

def get_current_price(client: KISClient, ticker=TICKER) -> dict:
    """주식현재가 시세 (FHKST01010100)."""
    data = client.get(
        "/uapi/domestic-stock/v1/quotations/inquire-price",
        "FHKST01010100",
        {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker},
    )
    output = data.get("output", {})
    return {
        "current": _si(output.get("stck_prpr", 0)),
        "open": _si(output.get("stck_oprc", 0)),
        "high": _si(output.get("stck_hgpr", 0)),
        "low": _si(output.get("stck_lwpr", 0)),
        "prev_close": _si(output.get("stck_sdpr", 0)),
        "volume": _si(output.get("acml_vol", 0)),
    }


def get_ohlcv_recent(client: KISClient, ticker=TICKER, days=100) -> pd.DataFrame:
    """최근 N일 일봉. shift(1) 적용."""
    start = (datetime.now() - timedelta(days=days + 10)).strftime("%Y%m%d")
    end = datetime.now().strftime("%Y%m%d")

    path = "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
    all_rows = []
    current_end = end

    for _ in range(10):
        data = client.get(path, "FHKST03010100", {
            "FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker,
            "FID_INPUT_DATE_1": start, "FID_INPUT_DATE_2": current_end,
            "FID_PERIOD_DIV_CODE": "D", "FID_ORG_ADJ_PRC": "0",
        })
        output = data.get("output2", [])
        if not output:
            break
        for item in output:
            d = item.get("stck_bsop_date", "")
            if not d or d < start:
                continue
            all_rows.append({
                "date": d,
                "Open": _si(item.get("stck_oprc", 0)),
                "High": _si(item.get("stck_hgpr", 0)),
                "Low": _si(item.get("stck_lwpr", 0)),
                "Close": _si(item.get("stck_clpr", 0)),
                "Volume": _si(item.get("acml_vol", 0)),
            })
        last = output[-1].get("stck_bsop_date", "")
        if not last or last <= start:
            break
        current_end = (datetime.strptime(last, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.drop_duplicates(subset="date").sort_values("date").set_index("date")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[f"prev_{col}"] = df[col].shift(1)
    df = df.iloc[1:].copy()
    return df


def get_investor_data(client: KISClient, ticker=TICKER) -> dict:
    """투자자별 매매동향 최근 데이터."""
    data = client.get(
        "/uapi/domestic-stock/v1/quotations/inquire-investor",
        "FHKST01010900",
        {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker},
    )
    output = data.get("output", [])
    if not output:
        return {}

    records = []
    for item in output[:10]:
        records.append({
            "date": item.get("stck_bsop_date", ""),
            "foreign_net": _si(item.get("frgn_ntby_qty", 0)),
            "inst_net": _si(item.get("orgn_ntby_qty", 0)),
        })

    # records[0]이 당일 0 값이면 D-1 사용
    target_idx = 0
    if len(records) > 1 and records[0]["foreign_net"] == 0 and records[0]["inst_net"] == 0:
        target_idx = 1

    return {
        "latest_foreign": records[target_idx]["foreign_net"] if records else 0,
        "latest_inst": records[target_idx]["inst_net"] if records else 0,
        "foreign_ma5": int(np.mean([r["foreign_net"] for r in records[target_idx: target_idx+5]])) if len(records) >= 5 else 0,
        "inst_ma5": int(np.mean([r["inst_net"] for r in records[target_idx: target_idx+5]])) if len(records) >= 5 else 0,
        "dual_buy": records[target_idx]["foreign_net"] > 0 and records[target_idx]["inst_net"] > 0 if records else False,
    }


# ═══════════════════════════════════════════════════════════════════
# 4. 기술적 지표 & 레짐 판별
# ═══════════════════════════════════════════════════════════════════

def compute_indicators(df, params: StrategyParams) -> pd.DataFrame:
    """prev_* 기반 지표 계산."""
    r = df.copy()
    pc = r["prev_Close"]
    ph = r["prev_High"]
    pl = r["prev_Low"]
    pv = r["prev_Volume"]

    # MA
    r["ma_s"] = pc.rolling(params.ma_short).mean()
    r["ma_l"] = pc.rolling(params.ma_long).mean()
    r["above_ma_s"] = (pc > r["ma_s"]).astype(int)
    r["ma_s_above_l"] = (r["ma_s"] > r["ma_l"]).astype(int)

    # BB
    bb_mid = pc.rolling(20).mean()
    bb_std = pc.rolling(20).std()
    r["bb_upper"] = bb_mid + bb_std * 2
    r["bb_lower"] = bb_mid - bb_std * 2
    r["bb_mid"] = bb_mid
    bb_w = np.where(bb_mid > 0, (r["bb_upper"] - r["bb_lower"]) / bb_mid * 100, 0)
    r["bb_width"] = bb_w
    bb_w_ma = pd.Series(bb_w, index=r.index).rolling(20).mean()
    r["bb_squeeze"] = np.where(bb_w_ma > 0, bb_w / bb_w_ma, 1.0)
    r["bb_pctb"] = np.where(
        (r["bb_upper"] - r["bb_lower"]) > 0,
        (pc - r["bb_lower"]) / (r["bb_upper"] - r["bb_lower"]),
        0.5,
    )

    # ATR
    prev_c = pc.shift(1)
    tr = pd.concat([ph - pl, (ph - prev_c).abs(), (pl - prev_c).abs()], axis=1).max(axis=1)
    r["atr"] = tr.ewm(span=14, adjust=False).mean()
    atr_ma = r["atr"].rolling(20).mean()
    r["atr_ratio"] = np.where(atr_ma > 0, r["atr"] / atr_ma, 1.0)

    # RSI
    delta = pc.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    rs = np.where(loss > 0, gain / loss, 100)
    r["rsi"] = 100 - 100 / (1 + rs)

    # Volume
    vol_ma = pv.rolling(20).mean()
    r["vol_ratio"] = np.where(vol_ma > 0, pv / vol_ma, 1.0)

    return r


def classify_regime(row, params: StrategyParams) -> str:
    """현재 레짐 판별."""
    if pd.isna(row.get("ma_l")):
        return "UNKNOWN"
    if row.get("atr_ratio", 1.0) >= params.atr_hvol_threshold:
        return "HIGH_VOLATILITY"
    if row.get("bb_squeeze", 1.0) <= params.bb_squeeze_threshold:
        return "RANGE_BOUND"
    if row.get("above_ma_s", 0) == 1 and row.get("ma_s_above_l", 0) == 1:
        return "TREND_UP"
    if row.get("above_ma_s", 0) == 0 and row.get("ma_s_above_l", 0) == 0:
        return "TREND_DOWN"
    return "NEUTRAL"


def compute_anomaly_multiplier(supply_anomaly: dict, params: StrategyParams) -> dict:
    """
    Isolation Forest 수급 이상 시그널을 포지션 multiplier로 변환.

    Returns:
        {
            "multiplier": float,   # invest_ratio에 곱할 배수 (0.0 = 차단)
            "block_entry": bool,   # True면 신규 진입 완전 차단
            "level": str,          # "strong_bearish" | "moderate_bearish" | "neutral" |
                                   #  "moderate_bullish" | "strong_bullish" | "no_anomaly"
            "detail": str,         # 로그/텔레그램용 설명
        }
    """
    if not supply_anomaly or not supply_anomaly.get("is_anomaly"):
        return {
            "multiplier": 1.0,
            "block_entry": False,
            "level": "no_anomaly",
            "detail": "정상 수급",
        }

    direction = supply_anomaly.get("direction", "neutral")
    score = supply_anomaly.get("anomaly_score", 0)

    # score가 양수이거나 0이면 실질적 이상 아님
    if score >= 0 or direction == "neutral":
        return {
            "multiplier": 1.0,
            "block_entry": False,
            "level": "no_anomaly",
            "detail": f"{direction} (score={score:.3f})",
        }

    is_strong = score <= params.anomaly_strong_threshold

    if direction == "bearish":
        if is_strong:
            mult = params.anomaly_strong_bearish_multiplier
            return {
                "multiplier": mult,
                "block_entry": mult <= 0.0,
                "level": "strong_bearish",
                "detail": f"🚨 강한 부정 수급 (score={score:.3f}, ×{mult})",
            }
        else:
            mult = params.anomaly_moderate_bearish_multiplier
            return {
                "multiplier": mult,
                "block_entry": False,
                "level": "moderate_bearish",
                "detail": f"⚠️ 중간 부정 수급 (score={score:.3f}, ×{mult})",
            }

    if direction == "bullish":
        if is_strong:
            mult = params.anomaly_strong_bullish_multiplier
            return {
                "multiplier": mult,
                "block_entry": False,
                "level": "strong_bullish",
                "detail": f"🟢 강한 긍정 수급 (score={score:.3f}, ×{mult})",
            }
        else:
            mult = params.anomaly_moderate_bullish_multiplier
            return {
                "multiplier": mult,
                "block_entry": False,
                "level": "moderate_bullish",
                "detail": f"🔵 중간 긍정 수급 (score={score:.3f}, ×{mult})",
            }

    return {
        "multiplier": 1.0,
        "block_entry": False,
        "level": "neutral",
        "detail": f"neutral (score={score:.3f})",
    }


# ═══════════════════════════════════════════════════════════════════
# 5. 주문 실행
# ═══════════════════════════════════════════════════════════════════

def kosdaq_tick_size(price: int) -> int:
    """코스닥 호가 단위."""
    for threshold, tick in KOSDAQ_TICK_TABLE:
        if price < threshold:
            return tick
    return 1000


def round_to_tick(price: int, direction: str = "down") -> int:
    """호가 단위로 반올림."""
    tick = kosdaq_tick_size(price)
    if direction == "down":
        return (price // tick) * tick
    else:
        return ((price + tick - 1) // tick) * tick


def _execute_buy(client: KISClient, state: BotState, price: int, qty: int, regime: str, today: str):
    """직접 KISClient로 매수 실행 (신규 포지션 개시)."""
    try:
        acc_no = os.getenv("KIS_ACC_NO", "")
        cano, acnt_prdt_cd = acc_no.split("-")

        body = {
            "CANO": cano, "ACNT_PRDT_CD": acnt_prdt_cd,
            "PDNO": TICKER, "ORD_DVSN": "00",   # 00: 지정가
            "ORD_QTY": str(qty), "ORD_UNPR": str(price),
        }

        log.info(f"[ORDER] 매수 주문: {TICKER} {qty}주 @ {price:,}원")
        resp = client.post("/uapi/domestic-stock/v1/trading/order-cash", "TTTC0802U", body)

        if resp.get("rt_cd") == "0":
            cost = qty * price
            state.cash -= cost
            state.position_qty = qty
            state.entry_price = float(price)
            state.entry_date = today
            state.highest_since_entry = 0.0
            state.tp1_done = False
            state.total_trades += 1
            # 신규 포지션이므로 피라미딩 상태 초기화
            state.pyramiding_count = 0
            state.pyramiding_history = []

            log.info(f"[BUY] 성공: {qty}주 @ {price:,} ({regime})")

            log_trade(
                state_dir=str(STATE_DIR),
                side="buy", reason="ENTRY",
                price=price, qty=qty,
                regime=regime, signal=state.last_signal,
            )

            pending = create_pending_from_response(resp, "buy", "ENTRY", qty, price)
            if pending:
                state.pending_orders.append(asdict(pending))
                log.info(f"[PENDING] 매수 추적 등록: {pending.order_no}")
        else:
            log.error(f"[BUY] KIS 응답 에러: {resp.get('msg1')}")
    except Exception as e:
        log.error(f"[BUY] 예외 발생: {e}")


def _execute_pyramiding_buy(client: KISClient, params: StrategyParams,
                              state: BotState, price: int, qty: int,
                              regime: str, today: str, trigger: str):
    """
    피라미딩 매수 실행 (기존 포지션에 추가).

    일반 _execute_buy와의 차이:
      - 기존 포지션을 덮어쓰지 않고 수량만 추가
      - entry_price는 가중평균으로 갱신 (손절/익절 기준도 자동 이동)
      - tp1_done, highest_since_entry 유지 (1차 익절 후 상태 보존)
      - pyramiding_count 증가, pyramiding_history에 이력 추가

    trigger: "DUAL" (외국인·기관 동반매수) | "ANOMALY" (bullish 수급 이상)
    """
    try:
        acc_no = os.getenv("KIS_ACC_NO", "")
        cano, acnt_prdt_cd = acc_no.split("-")

        body = {
            "CANO": cano, "ACNT_PRDT_CD": acnt_prdt_cd,
            "PDNO": TICKER, "ORD_DVSN": "00",
            "ORD_QTY": str(qty), "ORD_UNPR": str(price),
        }

        log.info(f"[PYRAMID] 추가 매수 주문: {qty}주 @ {price:,}원 (trigger={trigger})")
        resp = client.post("/uapi/domestic-stock/v1/trading/order-cash",
                            "TTTC0802U", body)

        if resp.get("rt_cd") == "0":
            # 가중평균 진입가 재계산
            old_cost = state.entry_price * state.position_qty
            new_cost = price * qty
            total_qty = state.position_qty + qty
            avg_entry = (old_cost + new_cost) / total_qty

            state.cash -= price * qty
            state.position_qty = total_qty
            state.entry_price = avg_entry
            state.pyramiding_count += 1
            state.pyramiding_history.append({
                "date": today,
                "price": price,
                "qty": qty,
                "trigger": trigger,
                "new_avg_entry": round(avg_entry, 2),
            })
            state.total_trades += 1

            log.info(f"[PYRAMID] 성공: +{qty}주 @ {price:,} | "
                     f"총 {total_qty}주 | 평균진입가 {avg_entry:,.0f} "
                     f"(count={state.pyramiding_count})")

            log_trade(
                state_dir=str(STATE_DIR),
                side="buy",
                reason=f"PYRAMID_{state.pyramiding_count}_{trigger}",
                price=price, qty=qty,
                regime=regime, signal=state.last_signal,
            )

            pending = create_pending_from_response(resp, "buy",
                                                    f"PYRAMID_{trigger}", qty, price)
            if pending:
                state.pending_orders.append(asdict(pending))
        else:
            log.error(f"[PYRAMID] KIS 응답 에러: {resp.get('msg1')}")
    except Exception as e:
        log.error(f"[PYRAMID] 예외 발생: {e}")


def _execute_sell(client: KISClient, params: StrategyParams, state: BotState, price: int, qty: int, reason: str, regime: str, today: str):
    """직접 KISClient로 매도 실행."""
    try:
        acc_no = os.getenv("KIS_ACC_NO", "")
        cano, acnt_prdt_cd = acc_no.split("-")

        body = {
            "CANO": cano, "ACNT_PRDT_CD": acnt_prdt_cd,
            "PDNO": TICKER, "ORD_DVSN": "00",
            "ORD_QTY": str(qty), "ORD_UNPR": str(price),
        }

        log.info(f"[ORDER] 매도 주문: {TICKER} {qty}주 @ {price:,}원")
        resp = client.post("/uapi/domestic-stock/v1/trading/order-cash", "TTTC0801U", body)

        if resp.get("rt_cd") == "0":
            proceeds = qty * price
            pnl = (price - state.entry_price) * qty

            entry_price_snapshot = state.entry_price
            entry_date_snapshot = state.entry_date

            state.cash += proceeds
            state.position_qty -= qty
            state.daily_pnl += pnl
            state.weekly_pnl += pnl
            state.monthly_pnl += pnl
            state.total_trades += 1

            if state.position_qty <= 0:
                state.position_qty = 0
                state.entry_price = 0.0
                state.tp1_done = False
                state.highest_since_entry = 0.0
                # 전량 청산 시 피라미딩 상태도 리셋
                state.pyramiding_count = 0
                state.pyramiding_history = []
                if reason == "SL":
                    cd = (datetime.strptime(today, "%Y-%m-%d") + timedelta(days=params.cooldown_days))
                    state.cooldown_until = cd.strftime("%Y-%m-%d")

            log.info(f"[SELL_{reason}] 성공: {qty}주 @ {price:,} | PnL={pnl:+,.0f} ({regime})")

            log_trade(
                state_dir=str(STATE_DIR),
                side="sell", reason=reason,
                price=price, qty=qty, pnl=int(pnl),
                regime=regime, signal=state.last_signal,
                entry_price=entry_price_snapshot,
                entry_date=entry_date_snapshot,
            )

            pending = create_pending_from_response(resp, "sell", reason, qty, price)
            if pending:
                state.pending_orders.append(asdict(pending))
                log.info(f"[PENDING] 매도 추적 등록: {pending.order_no} ({reason})")
        else:
            log.error(f"[SELL] KIS 응답 에러: {resp.get('msg1')}")
    except Exception as e:
        log.error(f"[SELL] 예외 발생: {e}")


# ═══════════════════════════════════════════════════════════════════
# 6. 리스크 관리
# ═══════════════════════════════════════════════════════════════════

def check_risk_limits(state: BotState, params: StrategyParams, today: str) -> bool:
    """리스크 한도 체크. 위반 시 True 반환 (거래 중단)."""
    today_dt = datetime.strptime(today, "%Y-%m-%d")

    # 주간 리셋 (월요일)
    if today_dt.weekday() == 0:
        if state.last_week_reset != today:
            state.weekly_pnl = 0.0
            state.last_week_reset = today

    # 월간 리셋 (1일)
    if today_dt.day == 1:
        if state.last_month_reset != today[:7]:
            state.monthly_pnl = 0.0
            state.last_month_reset = today[:7]

    capital = state.initial_capital

    if state.daily_pnl / capital <= params.daily_loss_limit:
        state.halted = True
        state.halt_reason = f"일일 손실 한도 초과 ({state.daily_pnl/capital*100:.1f}%)"
        return True

    if state.weekly_pnl / capital <= params.weekly_loss_limit:
        state.halted = True
        state.halt_reason = f"주간 손실 한도 초과 ({state.weekly_pnl/capital*100:.1f}%)"
        return True

    if state.monthly_pnl / capital <= params.monthly_loss_limit:
        state.halted = True
        state.halt_reason = f"월간 손실 한도 초과 ({state.monthly_pnl/capital*100:.1f}%)"
        return True

    state.halted = False
    state.halt_reason = ""
    return False


# ═══════════════════════════════════════════════════════════════════
# 7. 메인 봇 로직
# ═══════════════════════════════════════════════════════════════════

def run_bot(mode: str = "morning"):
    """봇 메인 루프."""
    log.info(f"{'='*60}")
    log.info(f"[BOT] {TICKER_NAME} v4 실행 (mode={mode})")
    log.info(f"{'='*60}")

    today = datetime.now().strftime("%Y-%m-%d")
    params = load_params()
    state = load_bot_state()

    # AI 레이어 로드
    ai = AILayer(str(STATE_DIR))
    ai.load_models()

    # 일일 PnL 리셋
    if state.last_trade_date != today:
        state.daily_pnl = 0.0
        state.last_trade_date = today

        # 이전 날짜의 추적 주문 정리
        if state.pending_orders:
            old_count = len(state.pending_orders)
            state.pending_orders = [
                p for p in state.pending_orders
                if p.get("ordered_date", "") == today.replace("-", "")
            ]
            if old_count != len(state.pending_orders):
                log.info(f"[CLEANUP] 이전 추적 주문 {old_count - len(state.pending_orders)}건 제거")

    # KIS 클라이언트
    if mode == "evening":
        _run_evening(None, params, state, today)
    else:
        try:
            client = KISClient()
        except Exception as e:
            log.error(f"[AUTH] 실패: {e}")
            send_telegram(f"❌ {TICKER_NAME} 봇 인증 실패: {e}")
            return

        if mode == "morning":
            _run_morning(client, params, state, today, ai)
        elif mode == "closing":
            _run_closing(client, params, state, today)

    state.last_run = datetime.now().isoformat()
    save_bot_state(state)
    log.info(f"[BOT] 완료. 포지션: {state.position_qty}주, 현금: {state.cash:,.0f}")


def _run_morning(client, params, state, today, ai):
    """09:02 — 시그널 판단 + 주문."""
    # 리스크 체크
    if check_risk_limits(state, params, today):
        log.warning(f"[RISK] 거래 중단: {state.halt_reason}")
        send_telegram(f"🚫 {TICKER_NAME} 거래 중단\n{state.halt_reason}")
        return

    # 쿨다운 체크
    if state.cooldown_until and today < state.cooldown_until:
        log.info(f"[COOLDOWN] {state.cooldown_until}까지 대기")
        return

    # 시세 조회
    price = get_current_price(client)
    current = price["current"]
    today_open = price["open"]

    if current <= 0 or today_open <= 0:
        log.warning("[PRICE] 시세 조회 실패")
        return

    log.info(f"[PRICE] 현재가={current:,}, 시가={today_open:,}, 전일종가={price['prev_close']:,}")

    # 갭다운 사전 경고
    risk = assess_risk(client, bot_state=state)
    save_risk_history(risk, str(STATE_DIR))
    send_telegram(format_risk_telegram(risk))

    # 진입 차단 판단
    skip, skip_reason = should_skip_entry(risk)
    if skip:
        log.warning(f"[RISK] 진입 차단: {skip_reason}")
        state.last_signal = f"BLOCKED: {skip_reason}"
        save_bot_state(state)
        return

    # 지표 계산
    df = get_ohlcv_recent(client, days=params.ma_long + 60)
    if df.empty or len(df) < params.ma_long:
        log.warning(f"[DATA] OHLCV 부족: {len(df)}일")
        return

    df_ind = compute_indicators(df, params)
    latest = df_ind.iloc[-1]
    regime = classify_regime(latest, params)

    # 수급 조회
    inv = get_investor_data(client)
    dual_buy = inv.get("dual_buy", False)

    # AI 앙상블 및 수급 이상 감지
    hmm_result = {"regime": "UNKNOWN", "confidence": 0}
    supply_anomaly = {"is_anomaly": False, "direction": "neutral"}

    if ai and ai.models_loaded:
        hmm_result = ai.get_hmm_regime(df)
        ensemble = ensemble_regime(regime, hmm_result)
        regime = ensemble["regime"]
        log.info(f"[AI] 앙상블: {ensemble['detail']} → {regime} ({ensemble['confidence']:.0%})")

        supply_anomaly = ai.detect_supply_anomaly(
            {"foreign_net_qty": inv.get("latest_foreign", 0),
             "inst_net_qty": inv.get("latest_inst", 0),
             "short_ratio_vol": 0},
            {"vol_ratio": latest.get("vol_ratio", 1.0)},
        )
        if supply_anomaly["is_anomaly"]:
            log.info(f"[AI] 수급 이상 감지: {supply_anomaly['direction']} (score={supply_anomaly['anomaly_score']:.3f})")

    # ── Isolation Forest 시그널을 multiplier로 변환 ──
    anomaly_info = compute_anomaly_multiplier(supply_anomaly, params)
    if anomaly_info["level"] != "no_anomaly":
        log.info(f"[IF_SIGNAL] {anomaly_info['detail']}")

    log.info(f"[REGIME] {regime} | RSI={latest.get('rsi', 0):.1f} | ATR비율={latest.get('atr_ratio', 0):.2f}")
    log.info(f"[SUPPLY] 외국인={inv.get('latest_foreign', 0):,} | 기관={inv.get('latest_inst', 0):,} | 동반매수={dual_buy}")

    state.last_regime = regime
    signal = "HOLD"

    # 뉴스 센티먼트 게이트
    sent = get_sentiment_signal(str(STATE_DIR))
    log.info(f"[NEWS] score={sent['score']:+d}, multiplier=×{sent['multiplier']}, "
             f"block={sent['block_entry']}, n={sent['n_articles']}")

    # ── 보유 중: 청산 판단 ──
    if state.position_qty > 0:
        pnl_pct = (today_open - state.entry_price) / state.entry_price

        # 손절
        if pnl_pct <= params.sl_pct:
            signal = "SELL_SL"
            sell_price = round_to_tick(today_open, "down")
            _execute_sell(client, params, state, sell_price, state.position_qty, "SL", regime, today)

        # 1차 익절 (미실행 시)
        elif pnl_pct >= params.tp1_pct and not state.tp1_done:
            signal = "SELL_TP1"
            sell_qty = max(1, int(state.position_qty * params.tp1_sell_ratio))
            sell_price = round_to_tick(today_open, "down")
            _execute_sell(client, params, state, sell_price, sell_qty, "TP1", regime, today)
            state.tp1_done = True
            state.highest_since_entry = today_open

        # 트레일링 스톱 (1차 익절 후)
        elif state.tp1_done and state.position_qty > 0:
            state.highest_since_entry = max(state.highest_since_entry, current)
            atr_val = latest.get("atr", 0)
            if pd.notna(atr_val) and atr_val > 0:
                trail_stop = state.highest_since_entry - atr_val * params.trail_atr_mult
                trail_stop = round_to_tick(int(trail_stop), "down")
                if current <= trail_stop:
                    signal = "SELL_TRAIL"
                    sell_price = round_to_tick(current, "down")
                    _execute_sell(client, params, state, sell_price, state.position_qty, "TRAIL", regime, today)

        # ── 피라미딩 매수 판단 (청산이 안 일어난 경우에만) ──
        # 조건이 엄격함: 하나라도 실패하면 미실행
        if (params.enable_pyramiding
                and state.position_qty > 0
                and signal not in ["SELL_SL", "SELL_TP1", "SELL_TRAIL"]
                and state.tp1_done                                     # 1차 익절 완료 후에만
                and state.pyramiding_count < params.max_pyramiding    # 횟수 미소진
                and regime in ["TREND_UP", "HIGH_VOLATILITY"]          # 유효 레짐
                and not anomaly_info["block_entry"]):                   # 강한 부정 수급 시 차단

            # 수익 상태 체크
            cur_profit = (current - state.entry_price) / state.entry_price
            if cur_profit >= params.pyramiding_min_profit:

                # 강한 수급 시그널 필수 (dual_buy OR bullish anomaly)
                strong_supply = (dual_buy or
                                  (supply_anomaly.get("is_anomaly") and
                                   supply_anomaly.get("direction") == "bullish"))

                if strong_supply:
                    # 누적 투자 한도 체크 (max_invest_ratio × initial_capital)
                    current_position_value = state.position_qty * current
                    max_position_value = state.initial_capital * params.max_invest_ratio
                    available_budget = max_position_value - current_position_value

                    if available_budget > 0:
                        # 초기 매수의 pyramiding_size_ratio 만큼 추가
                        initial_invest = state.initial_capital * params.invest_ratio
                        add_invest = min(initial_invest * params.pyramiding_size_ratio,
                                         available_budget,
                                         state.cash * 0.95)   # 현금 여유 확보

                        buy_price = round_to_tick(current, "up")
                        add_qty = int(add_invest / buy_price) if buy_price > 0 else 0

                        if add_qty > 0:
                            trigger = "DUAL" if dual_buy else "ANOMALY"
                            signal = f"BUY_PYRAMID_{trigger}"
                            _execute_pyramiding_buy(
                                client, params, state, buy_price, add_qty,
                                regime, today, trigger
                            )
                            log.info(f"[PYRAMID] 트리거={trigger}, "
                                     f"수익={cur_profit*100:.2f}%, "
                                     f"추가={add_qty}주, "
                                     f"count={state.pyramiding_count}/{params.max_pyramiding}")
                        else:
                            log.info(f"[PYRAMID] 수량 0 — add_invest={add_invest:,.0f}, "
                                     f"price={buy_price:,}")
                    else:
                        log.info(f"[PYRAMID] 한도 소진 — "
                                 f"current={current_position_value:,.0f}, "
                                 f"max={max_position_value:,.0f}")
                else:
                    log.debug(f"[PYRAMID] 수급 시그널 약함 (dual_buy={dual_buy})")
            else:
                log.debug(f"[PYRAMID] 수익 부족: {cur_profit*100:.2f}% "
                          f"< {params.pyramiding_min_profit*100:.1f}%")

    # ── 미보유: 진입 판단 ──
    elif state.position_qty == 0:
        # 뉴스 센티먼트로 진입 차단
        if sent["block_entry"]:
            log.warning("[NEWS] 부정 뉴스 과다, 진입 차단")
            signal = "BLOCKED_NEG_NEWS"
            state.last_signal = signal
        # Isolation Forest 강한 부정 시그널로 진입 차단
        elif anomaly_info["block_entry"]:
            log.warning(f"[IF_SIGNAL] 강한 부정 수급, 진입 차단")
            signal = "BLOCKED_SUPPLY_ANOMALY"
            state.last_signal = signal
        elif regime == "TREND_DOWN" or regime == "UNKNOWN":
            signal = "NO_ENTRY"

        elif regime == "TREND_UP":
            base_ratio = params.invest_ratio
            if dual_buy:
                base_ratio = params.max_invest_ratio
                signal = "BUY_TREND_DUAL"
            else:
                signal = "BUY_TREND"
            # sentiment × IF anomaly multiplier 합산
            effective_ratio = min(
                base_ratio * sent["multiplier"] * anomaly_info["multiplier"],
                params.max_invest_ratio,
            )
            invest = state.cash * effective_ratio
            buy_price = round_to_tick(current, "up")
            qty = int(invest / buy_price) if buy_price > 0 else 0
            if qty > 0:
                _execute_buy(client, state, buy_price, qty, regime, today)

        elif regime == "RANGE_BOUND":
            rsi_val = latest.get("rsi", 50)
            bb_pctb = latest.get("bb_pctb", 0.5)
            if rsi_val < params.rsi_entry and bb_pctb < 0.1:
                signal = "BUY_RB"
                effective_ratio = min(
                    params.invest_ratio * 0.5 * sent["multiplier"] * anomaly_info["multiplier"],
                    params.max_invest_ratio,
                )
                invest = state.cash * effective_ratio
                buy_price = round_to_tick(current, "up")
                qty = int(invest / buy_price) if buy_price > 0 else 0
                if qty > 0:
                    _execute_buy(client, state, buy_price, qty, regime, today)

        elif regime == "HIGH_VOLATILITY":
            rsi_val = latest.get("rsi", 50)
            if rsi_val < 25:
                signal = "BUY_HV"
                effective_ratio = min(
                    params.invest_ratio * params.hvol_size_reduction * sent["multiplier"] * anomaly_info["multiplier"],
                    params.max_invest_ratio,
                )
                invest = state.cash * effective_ratio
                buy_price = round_to_tick(current, "up")
                qty = int(invest / buy_price) if buy_price > 0 else 0
                if qty > 0:
                    _execute_buy(client, state, buy_price, qty, regime, today)

    state.last_signal = signal

    # ── Telegram 알림 ──
    equity = state.cash + state.position_qty * current
    pnl_total = (equity - state.initial_capital) / state.initial_capital * 100

    # 포지션 표시 (피라미딩 횟수 반영)
    if state.position_qty > 0:
        pos_str = f"{state.position_qty}주 @ 평균 {state.entry_price:,.0f}"
        if state.pyramiding_count > 0:
            pos_str += f" [P{state.pyramiding_count}]"
    else:
        pos_str = "0주"

    lines = [
        f"📊 <b>{TICKER_NAME} Morning</b>",
        f"시그널: <b>{signal}</b>",
        f"레짐: {regime}",
        f"HMM: {hmm_result.get('regime', 'N/A')} ({hmm_result.get('confidence', 0):.0%})",
        f"수급이상: {anomaly_info['detail']}",
        f"현재가: {current:,}원",
        f"RSI: {latest.get('rsi', 0):.1f} | ATR비율: {latest.get('atr_ratio', 0):.2f}",
        f"수급: 외{inv.get('latest_foreign',0):,} / 기{inv.get('latest_inst',0):,}",
        f"포지션: {pos_str}",
        f"평가: {equity:,.0f}원 ({pnl_total:+.1f}%)",
        format_sentiment_telegram(sent),
    ]

    # 예측 기록
    record_prediction(
        state_dir=str(STATE_DIR),
        date=today,
        regime=regime,
        signal=signal,
        hmm_regime=hmm_result.get("regime", "UNKNOWN"),
        hmm_confidence=hmm_result.get("confidence", 0),
        supply_anomaly=supply_anomaly.get("is_anomaly", False),
        supply_direction=supply_anomaly.get("direction", "neutral"),
        price_at_prediction=current,
    )

    send_telegram("\n".join(lines))


def _run_closing(client, params, state, today):
    """15:35 — 체결 확인 + 상태 갱신."""
    price = get_current_price(client)
    current = price["current"]

    if state.position_qty > 0 and current > 0:
        state.highest_since_entry = max(state.highest_since_entry, current)

    # 미체결 주문 처리
    unfilled_result = handle_unfilled_orders(client, state, current_price=current)

    # 어제 예측 라벨링
    evaluate_pending_predictions(str(STATE_DIR))

    # 일일 종합 리포트
    daily_msg = format_daily_report(state, current, str(STATE_DIR))

    equity = state.cash + state.position_qty * current
    pnl_total = (equity - state.initial_capital) / state.initial_capital * 100

    # 포지션 표시 (피라미딩 횟수 반영)
    if state.position_qty > 0:
        pos_str = f"{state.position_qty}주 @ 평균 {state.entry_price:,.0f}"
        if state.pyramiding_count > 0:
            pos_str += f" [P{state.pyramiding_count}]"
    else:
        pos_str = "0주"

    lines = [
        f"📊 <b>{TICKER_NAME} Closing</b>",
        f"종가: {current:,}원",
        f"포지션: {pos_str}",
        f"평가: {equity:,.0f}원 ({pnl_total:+.1f}%)",
        f"일일 PnL: {state.daily_pnl:+,.0f}원",
        f"레짐: {state.last_regime}",
    ]

    # 미체결 처리 결과
    if unfilled_result["checked"] > 0:
        lines.append("")
        lines.append(f"📝 주문 확인: {unfilled_result['checked']}건")
        if unfilled_result["filled"] > 0:
            lines.append(f"  ✅ 체결: {unfilled_result['filled']}건")
        if unfilled_result["cancelled"] > 0:
            lines.append(f"  🚫 취소: {unfilled_result['cancelled']}건")
        if unfilled_result["modified"] > 0:
            lines.append(f"  ✏️ 정정: {unfilled_result['modified']}건")
        if unfilled_result["errors"]:
            lines.append(f"  ❌ 오류: {len(unfilled_result['errors'])}건")

    send_telegram("\n".join(lines))

    # 금요일이면 주간 리포트 추가 발송
    if should_send_weekly_report():
        weekly_msg = format_weekly_report(state, str(STATE_DIR))
        send_telegram(weekly_msg)
        log.info("[REPORT] 주간 리포트 발송 완료")

    # 월말이면 월간 리포트 추가 발송
    if should_send_monthly_report():
        monthly_msg = format_monthly_report(state, str(STATE_DIR))
        send_telegram(monthly_msg)
        log.info("[REPORT] 월간 리포트 발송 완료")


def _run_evening(client, params, state, today):
    """18:30 — 확정 데이터 수집 (data_collector 호출)."""
    log.info("[EVENING] 데이터 수집은 별도 워크플로우에서 실행")


# ═══════════════════════════════════════════════════════════════════
# 8. Telegram
# ═══════════════════════════════════════════════════════════════════

def send_telegram(message: str):
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
# 유틸
# ═══════════════════════════════════════════════════════════════════

def _si(val) -> int:
    if val is None or val == "":
        return 0
    try:
        return int(float(str(val).replace(",", "")))
    except (ValueError, TypeError):
        return 0


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=f"{TICKER_NAME} v4 트레이딩 봇")
    parser.add_argument(
        "--mode", choices=["morning", "closing", "evening"],
        default="morning",
    )
    parser.add_argument("--state-dir", type=str, default="state")
    args = parser.parse_args()

    STATE_DIR = Path(args.state_dir)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    run_bot(mode=args.mode)
