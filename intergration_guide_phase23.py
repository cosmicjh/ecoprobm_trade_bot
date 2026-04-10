"""
Phase 2-3 미체결 주문 관리 — trading_bot.py 연동 가이드
========================================================
order_manager.py를 trading_bot.py에 통합하는 최소 수정 지점.
아래 5개 지점만 수정하면 됩니다.
"""

# ─────────────────────────────────────────────────────────
# 1. 파일 상단 import 추가
# ─────────────────────────────────────────────────────────
"""
from order_manager import (
    PendingOrder,
    create_pending_from_response,
    handle_unfilled_orders,
)
"""


# ─────────────────────────────────────────────────────────
# 2. BotState 클래스에 pending_orders 필드 추가
# ─────────────────────────────────────────────────────────
"""
@dataclass
class BotState:
    # ... 기존 필드들 ...
    version: str = "v4.2.1"

    # ★ 추가: 미체결 주문 추적 리스트 (PendingOrder의 dict 형태로 저장)
    pending_orders: list = field(default_factory=list)
"""
# 주의: dataclasses 모듈에서 field를 import해야 합니다.
#   from dataclasses import dataclass, asdict, field


# ─────────────────────────────────────────────────────────
# 3. _execute_buy 수정 — 주문 성공 시 PendingOrder 저장
# ─────────────────────────────────────────────────────────
"""
def _execute_buy(client: KISClient, state: BotState, price: int, qty: int, regime: str, today: str):
    try:
        acc_no = os.getenv("KIS_ACC_NO", "")
        cano, acnt_prdt_cd = acc_no.split("-")
        body = {
            "CANO": cano, "ACNT_PRDT_CD": acnt_prdt_cd,
            "PDNO": TICKER, "ORD_DVSN": "00",
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
            log.info(f"[BUY] 성공: {qty}주 @ {price:,} ({regime})")

            # ★ 추가: PendingOrder 생성 & 저장
            pending = create_pending_from_response(resp, "buy", "ENTRY", qty, price)
            if pending:
                state.pending_orders.append(asdict(pending))
                log.info(f"[PENDING] 매수 추적 등록: {pending.order_no}")
        else:
            log.error(f"[BUY] KIS 응답 에러: {resp.get('msg1')}")
    except Exception as e:
        log.error(f"[BUY] 예외 발생: {e}")
"""


# ─────────────────────────────────────────────────────────
# 4. _execute_sell 수정 — 주문 성공 시 PendingOrder 저장
# ─────────────────────────────────────────────────────────
"""
def _execute_sell(client: KISClient, params: StrategyParams, state: BotState,
                  price: int, qty: int, reason: str, regime: str, today: str):
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
                if reason == "SL":
                    cd = (datetime.strptime(today, "%Y-%m-%d")
                          + timedelta(days=params.cooldown_days))
                    state.cooldown_until = cd.strftime("%Y-%m-%d")

            log.info(f"[SELL_{reason}] 성공: {qty}주 @ {price:,} | PnL={pnl:+,.0f}")

            # ★ 추가: PendingOrder 생성 & 저장
            pending = create_pending_from_response(resp, "sell", reason, qty, price)
            if pending:
                state.pending_orders.append(asdict(pending))
                log.info(f"[PENDING] 매도 추적 등록: {pending.order_no} ({reason})")
        else:
            log.error(f"[SELL] KIS 응답 에러: {resp.get('msg1')}")
    except Exception as e:
        log.error(f"[SELL] 예외 발생: {e}")
"""


# ─────────────────────────────────────────────────────────
# 5. _run_closing에 미체결 주문 처리 추가
# ─────────────────────────────────────────────────────────
"""
def _run_closing(client, params, state, today):
    # 시세 조회
    price = get_current_price(client)
    current = price["current"]

    if state.position_qty > 0 and current > 0:
        state.highest_since_entry = max(state.highest_since_entry, current)

    # ★ 추가: 미체결 주문 처리
    unfilled_result = handle_unfilled_orders(client, state, current_price=current)

    equity = state.cash + state.position_qty * current
    pnl_total = (equity - state.initial_capital) / state.initial_capital * 100

    lines = [
        f"📊 <b>{TICKER_NAME} Closing</b>",
        f"종가: {current:,}원",
        f"포지션: {state.position_qty}주",
        f"평가: {equity:,.0f}원 ({pnl_total:+.1f}%)",
        f"일일 PnL: {state.daily_pnl:+,.0f}원",
        f"레짐: {state.last_regime}",
    ]

    # ★ 추가: 미체결 처리 결과를 Telegram에 포함
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

    send_telegram("\\n".join(lines))
"""


# ─────────────────────────────────────────────────────────
# 6. (선택) pending_orders 리셋 — 날짜 변경 시
# ─────────────────────────────────────────────────────────
"""
# run_bot() 내 일일 PnL 리셋 근처에 추가:

    if state.last_trade_date != today:
        state.daily_pnl = 0.0
        state.last_trade_date = today

        # ★ 추가: 이전 날짜의 추적 주문 정리
        # 당일 주문이 아닌 것은 조회 대상이 아니므로 제거
        if state.pending_orders:
            old_count = len(state.pending_orders)
            state.pending_orders = [
                p for p in state.pending_orders
                if p.get("ordered_date", "") == today.replace("-", "")
            ]
            if old_count != len(state.pending_orders):
                log.info(f"[CLEANUP] 이전 추적 주문 {old_count - len(state.pending_orders)}건 제거")
"""


# ─────────────────────────────────────────────────────────
# 동작 흐름 요약
# ─────────────────────────────────────────────────────────
"""
[09:05 morning]
  1. 시그널 판단 → _execute_buy/_execute_sell 호출
  2. state 낙관적 반영 (전량 체결 가정)
  3. PendingOrder 생성 → state.pending_orders에 append
  4. state 저장 (JSON persist)

[15:10 closing]
  1. 종가 조회
  2. handle_unfilled_orders 호출
     a. 당일 주문 조회 (TTTC0081R)
     b. pending_orders 순회하며 매칭
     c. 완전 체결 → 추적 종료
     d. 매수 미체결 → 취소 + state 복구
     e. 매도 SL 미체결 → 공격적 정정
     f. 매도 TP/TRAIL 미체결 → 취소 + state 복구
  3. Telegram 리포트에 처리 결과 포함
  4. state 저장

이로써:
  - 주문 후 state가 '낙관적'으로 선반영되어도 closing에서 반드시 실제와 동기화
  - 손절 미체결 리스크 (장 마감 후 보유 잔존) 자동 방지
  - 매수 추격 방지 (미체결 매수는 취소하여 재판단)
"""
