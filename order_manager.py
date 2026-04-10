"""
에코프로비엠 (247540) v4 — Phase 2-3: 미체결 주문 관리
========================================================
주문 생명주기 관리:
  1. morning: 주문 전송 → PendingOrder를 state에 저장
  2. closing: 주문 조회 → 미체결 주문 식별
  3. 미체결 처리 원칙:
     - 매수 미체결 (ENTRY)     → 취소 (시장 상황 변동)
     - 매도 미체결 (SL)        → 가격 하향 정정 (반드시 청산)
     - 매도 미체결 (TP1/TRAIL) → 취소 (다음날 재판단)

핵심 설계:
  trading_bot._execute_buy/_execute_sell은 주문 시점에 '전량 체결된 것처럼'
  state를 미리 반영합니다. sync_order_to_state()는 실제 체결량과의 차이(diff)만큼
  상태를 되돌리는 보정 역할을 합니다. 이로써 체결 지연에도 state가 항상
  "실제 보유 상황"에 맞게 유지됩니다.

한투 API:
  - 일별주문체결조회: TTTC0081R /uapi/domestic-stock/v1/trading/inquire-daily-ccld
  - 정정/취소 주문:    TTTC0803U /uapi/domestic-stock/v1/trading/order-rvsecncl
"""

import os
import logging
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional

log = logging.getLogger(__name__)

TICKER = "247540"


# ═══════════════════════════════════════════════════════════════════
# 1. PendingOrder 데이터 구조
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PendingOrder:
    """체결 대기 중인 주문."""
    order_no: str = ""          # 주문번호 (ODNO)
    org_no: str = ""            # 주문조직번호 (KRX_FWDG_ORD_ORGNO)
    side: str = ""              # "buy" / "sell"
    reason: str = ""            # "ENTRY" / "SL" / "TP1" / "TRAIL"
    qty: int = 0                # 원 주문 수량
    price: int = 0              # 원 주문 가격
    ordered_at: str = ""        # 주문 시각 (ISO)
    ordered_date: str = ""      # 주문일 (YYYYMMDD)

    # 체결 조회로 업데이트됨
    filled_qty: int = 0
    remaining_qty: int = 0
    status: str = "pending"     # pending / filled / partial / cancelled / modified


def create_pending_from_response(resp: dict, side: str, reason: str,
                                  qty: int, price: int) -> Optional[PendingOrder]:
    """
    한투 주문 API 응답에서 PendingOrder 생성.

    한투 주문 응답 output:
      KRX_FWDG_ORD_ORGNO: 주문조직번호
      ODNO: 주문번호
      ORD_TMD: 주문시각
    """
    if resp.get("rt_cd") != "0":
        return None

    output = resp.get("output", {}) or {}

    # 대소문자 모두 대응
    order_no = output.get("ODNO") or output.get("odno", "")
    org_no = output.get("KRX_FWDG_ORD_ORGNO") or output.get("krx_fwdg_ord_orgno", "")

    if not order_no:
        log.warning(f"[PENDING] 응답에 주문번호 없음: {resp}")
        return None

    return PendingOrder(
        order_no=str(order_no),
        org_no=str(org_no),
        side=side,
        reason=reason,
        qty=qty,
        price=price,
        ordered_at=datetime.now().isoformat(),
        ordered_date=datetime.now().strftime("%Y%m%d"),
        remaining_qty=qty,
        status="pending",
    )


# ═══════════════════════════════════════════════════════════════════
# 2. 주문 조회 (일별 체결 내역)
# ═══════════════════════════════════════════════════════════════════

def query_daily_orders(client, start_date: str = None) -> list:
    """
    주식일별주문체결조회 (TTTC0081R).

    Returns:
        [{order_no, side, order_qty, filled_qty, remaining_qty, status, ...}]
    """
    if start_date is None:
        start_date = datetime.now().strftime("%Y%m%d")
    end_date = datetime.now().strftime("%Y%m%d")

    acc_no = os.getenv("KIS_ACC_NO", "")
    if "-" not in acc_no:
        log.error("[ORDER_QUERY] KIS_ACC_NO 형식 오류 (하이픈 필요)")
        return []
    cano, acnt_prdt_cd = acc_no.split("-")

    path = "/uapi/domestic-stock/v1/trading/inquire-daily-ccld"
    tr_id = "TTTC0081R"

    params = {
        "CANO": cano,
        "ACNT_PRDT_CD": acnt_prdt_cd,
        "INQR_STRT_DT": start_date,
        "INQR_END_DT": end_date,
        "SLL_BUY_DVSN_CD": "00",    # 00:전체
        "INQR_DVSN": "00",          # 00:역순
        "PDNO": "",
        "CCLD_DVSN": "00",          # 00:전체
        "ORD_GNO_BRNO": "",
        "ODNO": "",
        "INQR_DVSN_3": "00",
        "INQR_DVSN_1": "",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }

    try:
        data = client.get(path, tr_id, params)
    except Exception as e:
        log.error(f"[ORDER_QUERY] 호출 실패: {e}")
        return []

    if data.get("rt_cd") != "0":
        log.warning(f"[ORDER_QUERY] {data.get('msg1', '')}")
        return []

    output1 = data.get("output1", [])
    orders = []

    for item in output1:
        pdno = item.get("pdno", "")
        if pdno != TICKER:
            continue

        order_qty = _si(item.get("ord_qty", 0))
        filled = _si(item.get("tot_ccld_qty", 0))
        remaining = _si(item.get("rmn_qty", 0))
        cncl_yn = item.get("cncl_yn", "N")

        if cncl_yn == "Y":
            status = "cancelled"
        elif remaining == 0 and filled > 0:
            status = "filled"
        elif remaining > 0 and filled > 0:
            status = "partial"
        elif remaining > 0:
            status = "pending"
        else:
            status = "unknown"

        orders.append({
            "order_no": item.get("odno", ""),
            "org_no": item.get("ord_gno_brno", ""),
            "pdno": pdno,
            "side": "buy" if item.get("sll_buy_dvsn_cd", "") == "02" else "sell",
            "order_qty": order_qty,
            "filled_qty": filled,
            "remaining_qty": remaining,
            "order_price": _si(item.get("ord_unpr", 0)),
            "avg_fill_price": _si(item.get("avg_prvs", 0)),
            "status": status,
            "ord_time": item.get("ord_tmd", ""),
        })

    return orders


# ═══════════════════════════════════════════════════════════════════
# 3. 주문 정정 / 취소
# ═══════════════════════════════════════════════════════════════════

def cancel_order(client, pending: PendingOrder) -> dict:
    """
    주문 취소 (TTTC0803U, RVSE_CNCL_DVSN_CD='02').
    """
    acc_no = os.getenv("KIS_ACC_NO", "")
    cano, acnt_prdt_cd = acc_no.split("-")

    body = {
        "CANO": cano,
        "ACNT_PRDT_CD": acnt_prdt_cd,
        "KRX_FWDG_ORD_ORGNO": pending.org_no,
        "ORGN_ODNO": pending.order_no,
        "ORD_DVSN": "00",
        "RVSE_CNCL_DVSN_CD": "02",  # 취소
        "ORD_QTY": "0",
        "ORD_UNPR": "0",
        "QTY_ALL_ORD_YN": "Y",
    }

    log.info(f"[CANCEL] {pending.order_no} ({pending.side}/{pending.reason}) 취소 요청")

    try:
        resp = client.post(
            "/uapi/domestic-stock/v1/trading/order-rvsecncl",
            "TTTC0803U",
            body,
        )
        if resp.get("rt_cd") == "0":
            log.info(f"[CANCEL] 성공: {pending.order_no}")
            pending.status = "cancelled"
            return {"success": True, "message": "cancelled"}
        else:
            log.warning(f"[CANCEL] 실패: {resp.get('msg1', '')}")
            return {"success": False, "message": resp.get("msg1", "")}
    except Exception as e:
        log.error(f"[CANCEL] 예외: {e}")
        return {"success": False, "message": str(e)}


def modify_order(client, pending: PendingOrder, new_price: int, new_qty: int = None) -> dict:
    """
    주문 정정 (TTTC0803U, RVSE_CNCL_DVSN_CD='01').
    매도 손절 미체결 시 가격을 공격적으로 낮춰 강제 청산.
    """
    acc_no = os.getenv("KIS_ACC_NO", "")
    cano, acnt_prdt_cd = acc_no.split("-")

    qty = new_qty if new_qty else (pending.remaining_qty or pending.qty)

    body = {
        "CANO": cano,
        "ACNT_PRDT_CD": acnt_prdt_cd,
        "KRX_FWDG_ORD_ORGNO": pending.org_no,
        "ORGN_ODNO": pending.order_no,
        "ORD_DVSN": "00",
        "RVSE_CNCL_DVSN_CD": "01",  # 정정
        "ORD_QTY": str(qty),
        "ORD_UNPR": str(new_price),
        "QTY_ALL_ORD_YN": "Y",
    }

    log.info(f"[MODIFY] {pending.order_no} 정정: {pending.price:,} → {new_price:,}")

    try:
        resp = client.post(
            "/uapi/domestic-stock/v1/trading/order-rvsecncl",
            "TTTC0803U",
            body,
        )
        if resp.get("rt_cd") == "0":
            log.info(f"[MODIFY] 성공")
            pending.status = "modified"
            pending.price = new_price
            return {"success": True, "message": "modified"}
        else:
            log.warning(f"[MODIFY] 실패: {resp.get('msg1', '')}")
            return {"success": False, "message": resp.get("msg1", "")}
    except Exception as e:
        log.error(f"[MODIFY] 예외: {e}")
        return {"success": False, "message": str(e)}


# ═══════════════════════════════════════════════════════════════════
# 4. 상태 동기화 (diff 기반 보정)
# ═══════════════════════════════════════════════════════════════════

def sync_order_to_state(state, pending: PendingOrder, fill_info: dict):
    """
    실제 체결 정보를 봇 상태에 반영.

    핵심 원칙:
      trading_bot._execute_buy/_execute_sell은 주문 시점에 '전량 체결된 것처럼'
      state를 미리 반영합니다. 이 함수는 실제 체결량과의 차이(diff)만큼
      상태를 되돌리는 '복구' 역할만 합니다.

      expected(주문량) - filled(체결량) = diff (복구할 수량)

    매수:
      diff > 0 → position -= diff, cash += diff × price
    매도:
      diff > 0 → position += diff, cash -= diff × price, pnl 복구
    """
    filled = fill_info.get("filled_qty", 0)
    avg_price = fill_info.get("avg_fill_price", 0)

    if avg_price == 0:
        avg_price = pending.price

    expected = pending.qty
    diff = expected - filled

    if diff <= 0:
        # 완전 체결
        return

    if pending.side == "buy":
        state.cash += diff * pending.price
        state.position_qty -= diff
        log.info(f"[SYNC] 매수 {filled}/{expected} 체결, {diff}주 상태 복구")

        # 부분 체결이면 평균 진입가 재계산
        if filled > 0 and avg_price != pending.price:
            state.entry_price = float(avg_price)

        # 완전 미체결로 포지션이 0이 되면 진입 관련 플래그 초기화
        if filled == 0 and state.position_qty == 0:
            state.entry_price = 0.0
            if hasattr(state, "tp1_done"):
                state.tp1_done = False
            if hasattr(state, "highest_since_entry"):
                state.highest_since_entry = 0.0

    elif pending.side == "sell":
        state.cash -= diff * pending.price
        state.position_qty += diff

        # _execute_sell에서 사전 반영한 PnL 복구
        undo_pnl = (pending.price - state.entry_price) * diff
        state.daily_pnl -= undo_pnl
        state.weekly_pnl -= undo_pnl
        state.monthly_pnl -= undo_pnl
        log.info(f"[SYNC] 매도 {filled}/{expected} 체결, {diff}주 복구 "
                 f"(pnl {undo_pnl:+,.0f} 복구)")


# ═══════════════════════════════════════════════════════════════════
# 5. 미체결 주문 처리 (closing 모드에서 호출)
# ═══════════════════════════════════════════════════════════════════

def handle_unfilled_orders(client, state, current_price: int) -> dict:
    """
    state.pending_orders를 순회하며 미체결 주문을 처리합니다.

    처리 원칙:
      - 완전 체결         → 추적 종료
      - 매수 미체결       → 취소 + state 복구
      - 매도 SL 미체결    → 공격적 정정 (반드시 청산)
      - 매도 TP/TRAIL 미체결 → 취소 + state 복구

    Returns:
        {checked, filled, cancelled, modified, errors}
    """
    result = {
        "checked": 0,
        "filled": 0,
        "cancelled": 0,
        "modified": 0,
        "errors": [],
    }

    pending_list = getattr(state, "pending_orders", [])
    if not pending_list:
        log.info("[UNFILLED] 추적 중인 주문 없음")
        return result

    # 당일 주문 조회
    try:
        all_orders = query_daily_orders(client)
    except Exception as e:
        log.error(f"[UNFILLED] 주문 조회 실패: {e}")
        result["errors"].append(f"query: {e}")
        return result

    order_map = {o["order_no"]: o for o in all_orders}
    updated_pending = []

    for p_data in pending_list:
        # dict → PendingOrder 변환
        if isinstance(p_data, dict):
            pending = PendingOrder(**p_data)
        else:
            pending = p_data

        result["checked"] += 1

        fill_info = order_map.get(pending.order_no)
        if not fill_info:
            log.warning(f"[UNFILLED] {pending.order_no} 조회 결과 없음 — 유지")
            updated_pending.append(asdict(pending))
            continue

        pending.filled_qty = fill_info["filled_qty"]
        pending.remaining_qty = fill_info["remaining_qty"]
        pending.status = fill_info["status"]

        # ── 완전 체결 ──
        if pending.status == "filled":
            log.info(f"[UNFILLED] {pending.order_no} 완전 체결 ({pending.filled_qty}주)")
            result["filled"] += 1
            # _execute_buy/sell이 이미 반영함 → 추적만 종료
            continue

        # ── 이미 취소됨 ──
        if pending.status == "cancelled":
            log.info(f"[UNFILLED] {pending.order_no} 이미 취소됨")
            sync_order_to_state(state, pending, fill_info)
            continue

        # ── 부분 체결 또는 미체결 ──
        remaining = pending.remaining_qty

        if remaining > 0:
            # 매수 미체결 → 취소
            if pending.side == "buy":
                cancel_result = cancel_order(client, pending)
                if cancel_result["success"]:
                    result["cancelled"] += 1
                    sync_order_to_state(state, pending, fill_info)
                else:
                    result["errors"].append(f"cancel_buy: {cancel_result['message']}")
                    updated_pending.append(asdict(pending))

            # 매도 손절 미체결 → 가격 정정 (강제 청산)
            elif pending.side == "sell" and pending.reason == "SL":
                try:
                    from trading_bot import round_to_tick, kosdaq_tick_size
                    tick = kosdaq_tick_size(current_price)
                    new_price = round_to_tick(current_price - tick * 2, "down")
                except ImportError:
                    new_price = current_price - 200

                log.warning(f"[UNFILLED] 손절 미체결 {remaining}주 → 정정 {new_price:,}")
                mod_result = modify_order(client, pending, new_price, remaining)
                if mod_result["success"]:
                    result["modified"] += 1
                    # 정정 후에도 추적 유지 (다음 closing에서 재확인)
                    updated_pending.append(asdict(pending))
                else:
                    result["errors"].append(f"modify_sl: {mod_result['message']}")
                    updated_pending.append(asdict(pending))

            # 매도 TP/TRAIL 미체결 → 취소
            elif pending.side == "sell":
                cancel_result = cancel_order(client, pending)
                if cancel_result["success"]:
                    result["cancelled"] += 1
                    sync_order_to_state(state, pending, fill_info)
                else:
                    result["errors"].append(f"cancel_sell: {cancel_result['message']}")
                    updated_pending.append(asdict(pending))

    state.pending_orders = updated_pending

    log.info(f"[UNFILLED] 결과: 확인={result['checked']}, 체결={result['filled']}, "
             f"취소={result['cancelled']}, 정정={result['modified']}, "
             f"오류={len(result['errors'])}")
    return result


# ═══════════════════════════════════════════════════════════════════
# 6. 유틸
# ═══════════════════════════════════════════════════════════════════

def _si(val) -> int:
    if val is None or val == "":
        return 0
    try:
        return int(float(str(val).replace(",", "")))
    except (ValueError, TypeError):
        return 0
