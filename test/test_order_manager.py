"""Phase 2-3 미체결 주문 관리 검증."""

import os
import sys
import json
from dataclasses import dataclass, field, asdict

sys.path.insert(0, os.path.dirname(__file__))
os.environ["KIS_ACC_NO"] = "12345678-01"

from order_manager import (
    PendingOrder, query_daily_orders, cancel_order, modify_order,
    sync_order_to_state, handle_unfilled_orders, create_pending_from_response,
)


# ─── Mock KISClient ───
class MockKISClient:
    def __init__(self):
        self.get_responses = {}
        self.post_responses = {}
        self.get_calls = []
        self.post_calls = []

    def set_get(self, tr_id, response):
        self.get_responses[tr_id] = response

    def set_post(self, tr_id, response):
        self.post_responses[tr_id] = response

    def get(self, path, tr_id, params, extra_headers=None):
        self.get_calls.append({"path": path, "tr_id": tr_id, "params": params})
        return self.get_responses.get(tr_id, {"rt_cd": "1", "msg1": "no mock"})

    def post(self, path, tr_id, body):
        self.post_calls.append({"path": path, "tr_id": tr_id, "body": body})
        return self.post_responses.get(tr_id, {"rt_cd": "1", "msg1": "no mock"})


# ─── Mock BotState ───
@dataclass
class MockBotState:
    position_qty: int = 0
    entry_price: float = 0.0
    cash: float = 1_500_000.0
    initial_capital: float = 1_500_000.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    tp1_done: bool = False
    highest_since_entry: float = 0.0
    pending_orders: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════
# 테스트
# ═══════════════════════════════════════════════════════════

def test_pending_creation_success():
    resp = {
        "rt_cd": "0",
        "output": {
            "KRX_FWDG_ORD_ORGNO": "00950",
            "ODNO": "0000123456",
            "ORD_TMD": "091530",
        },
    }
    p = create_pending_from_response(resp, "buy", "ENTRY", 10, 200000)
    assert p is not None
    assert p.order_no == "0000123456"
    assert p.org_no == "00950"
    assert p.side == "buy"
    assert p.reason == "ENTRY"
    assert p.qty == 10
    assert p.remaining_qty == 10
    print("✅ PendingOrder 생성 성공")


def test_pending_creation_failure():
    resp = {"rt_cd": "1", "msg1": "잔고 부족"}
    assert create_pending_from_response(resp, "buy", "ENTRY", 10, 200000) is None
    print("✅ PendingOrder 실패 응답 → None")


def test_query_filled():
    client = MockKISClient()
    client.set_get("TTTC0081R", {
        "rt_cd": "0",
        "output1": [{
            "odno": "0000001", "ord_gno_brno": "00950", "pdno": "247540",
            "sll_buy_dvsn_cd": "02",
            "ord_qty": "10", "tot_ccld_qty": "10", "rmn_qty": "0",
            "ord_unpr": "200000", "avg_prvs": "199800", "cncl_yn": "N",
        }],
    })
    orders = query_daily_orders(client)
    assert len(orders) == 1
    assert orders[0]["status"] == "filled"
    assert orders[0]["filled_qty"] == 10
    print("✅ 주문 조회 - 완전 체결")


def test_query_partial():
    client = MockKISClient()
    client.set_get("TTTC0081R", {
        "rt_cd": "0",
        "output1": [{
            "odno": "0000002", "ord_gno_brno": "00950", "pdno": "247540",
            "sll_buy_dvsn_cd": "02",
            "ord_qty": "10", "tot_ccld_qty": "6", "rmn_qty": "4",
            "ord_unpr": "200000", "avg_prvs": "200000", "cncl_yn": "N",
        }],
    })
    orders = query_daily_orders(client)
    assert orders[0]["status"] == "partial"
    assert orders[0]["filled_qty"] == 6
    assert orders[0]["remaining_qty"] == 4
    print("✅ 주문 조회 - 부분 체결")


def test_query_ticker_filter():
    client = MockKISClient()
    client.set_get("TTTC0081R", {
        "rt_cd": "0",
        "output1": [
            {"odno": "A1", "pdno": "005930", "sll_buy_dvsn_cd": "02",
             "ord_qty": "10", "tot_ccld_qty": "10", "rmn_qty": "0",
             "ord_unpr": "70000", "avg_prvs": "70000", "cncl_yn": "N"},
            {"odno": "A2", "pdno": "247540", "sll_buy_dvsn_cd": "02",
             "ord_qty": "5", "tot_ccld_qty": "5", "rmn_qty": "0",
             "ord_unpr": "200000", "avg_prvs": "200000", "cncl_yn": "N"},
        ],
    })
    orders = query_daily_orders(client)
    assert len(orders) == 1
    assert orders[0]["order_no"] == "A2"
    print("✅ 주문 조회 - 종목 필터")


def test_cancel_order():
    client = MockKISClient()
    client.set_post("TTTC0803U", {"rt_cd": "0"})

    p = PendingOrder(order_no="X001", org_no="00950",
                     side="buy", reason="ENTRY", qty=10, price=200000)
    r = cancel_order(client, p)
    assert r["success"] == True
    assert p.status == "cancelled"
    body = client.post_calls[0]["body"]
    assert body["RVSE_CNCL_DVSN_CD"] == "02"
    assert body["ORGN_ODNO"] == "X001"
    print("✅ 주문 취소")


def test_modify_order():
    client = MockKISClient()
    client.set_post("TTTC0803U", {"rt_cd": "0"})

    p = PendingOrder(order_no="X002", org_no="00950",
                     side="sell", reason="SL", qty=10, price=196000,
                     remaining_qty=10)
    r = modify_order(client, p, new_price=194500)
    assert r["success"] == True
    assert p.status == "modified"
    assert p.price == 194500
    body = client.post_calls[0]["body"]
    assert body["RVSE_CNCL_DVSN_CD"] == "01"
    assert body["ORD_UNPR"] == "194500"
    print("✅ 주문 정정")


def test_sync_buy_full_unfill():
    """매수 전량 미체결 시 state 완전 복구."""
    state = MockBotState(
        position_qty=10,           # _execute_buy가 반영
        entry_price=200000.0,
        cash=-500_000,             # 1.5M - 2M (차감)
        tp1_done=False,
        highest_since_entry=0.0,
    )
    p = PendingOrder(order_no="X", side="buy", reason="ENTRY",
                     qty=10, price=200000)
    fill = {"filled_qty": 0, "avg_fill_price": 0}

    sync_order_to_state(state, p, fill)

    assert state.position_qty == 0
    assert state.cash == 1_500_000.0
    assert state.entry_price == 0.0
    print("✅ sync: 매수 전량 미체결 → 전량 복구")


def test_sync_buy_partial():
    """매수 부분 체결 시 미체결분만 복구."""
    state = MockBotState(
        position_qty=10,
        entry_price=200000.0,
        cash=-500_000,
    )
    p = PendingOrder(order_no="X", side="buy", reason="ENTRY",
                     qty=10, price=200000)
    fill = {"filled_qty": 6, "avg_fill_price": 200000}

    sync_order_to_state(state, p, fill)

    assert state.position_qty == 6     # 10 - 4
    assert state.cash == 300_000.0     # -500K + 800K
    print("✅ sync: 매수 부분 체결 → 4주 복구")


def test_sync_sell_full_unfill_restores_pnl():
    """매도 전량 미체결 시 position + pnl 복구."""
    state = MockBotState(
        position_qty=0,              # _execute_sell이 차감
        entry_price=200000.0,
        cash=1_960_000.0,            # 196,000 × 10 수익 반영
        daily_pnl=-40_000,           # (196000-200000) × 10
        weekly_pnl=-40_000,
        monthly_pnl=-40_000,
    )
    p = PendingOrder(order_no="X", side="sell", reason="SL",
                     qty=10, price=196000)
    fill = {"filled_qty": 0, "avg_fill_price": 0}

    sync_order_to_state(state, p, fill)

    assert state.position_qty == 10
    assert state.cash == 0.0
    assert state.daily_pnl == 0.0
    assert state.weekly_pnl == 0.0
    assert state.monthly_pnl == 0.0
    print("✅ sync: 매도 전량 미체결 → position + pnl 복구")


def test_handle_buy_unfilled_cancel():
    """미체결 매수 → 취소 + state 복구 (회귀 테스트)."""
    client = MockKISClient()
    client.set_get("TTTC0081R", {
        "rt_cd": "0",
        "output1": [{
            "odno": "BUY001", "ord_gno_brno": "00950", "pdno": "247540",
            "sll_buy_dvsn_cd": "02",
            "ord_qty": "10", "tot_ccld_qty": "0", "rmn_qty": "10",
            "ord_unpr": "200000", "avg_prvs": "0", "cncl_yn": "N",
        }],
    })
    client.set_post("TTTC0803U", {"rt_cd": "0"})

    state = MockBotState(
        position_qty=10,
        entry_price=200000.0,
        cash=-500_000,
        pending_orders=[asdict(PendingOrder(
            order_no="BUY001", org_no="00950",
            side="buy", reason="ENTRY", qty=10, price=200000,
            remaining_qty=10, status="pending",
        ))],
    )

    result = handle_unfilled_orders(client, state, current_price=199000)

    assert result["checked"] == 1
    assert result["cancelled"] == 1
    assert state.position_qty == 0
    assert state.cash == 1_500_000.0
    assert len(state.pending_orders) == 0
    print("✅ handle: 매수 미체결 취소 + 완전 복구")


def test_handle_sell_sl_modify():
    """미체결 손절 매도 → 공격적 정정."""
    client = MockKISClient()
    client.set_get("TTTC0081R", {
        "rt_cd": "0",
        "output1": [{
            "odno": "SL001", "ord_gno_brno": "00950", "pdno": "247540",
            "sll_buy_dvsn_cd": "01",
            "ord_qty": "10", "tot_ccld_qty": "0", "rmn_qty": "10",
            "ord_unpr": "192000", "avg_prvs": "0", "cncl_yn": "N",
        }],
    })
    client.set_post("TTTC0803U", {"rt_cd": "0"})

    state = MockBotState(
        position_qty=0,
        entry_price=200000.0,
        pending_orders=[asdict(PendingOrder(
            order_no="SL001", org_no="00950",
            side="sell", reason="SL", qty=10, price=192000,
            remaining_qty=10, status="pending",
        ))],
    )

    # trading_bot 모듈을 모킹 (round_to_tick import)
    import types
    mock_tb = types.ModuleType("trading_bot")
    mock_tb.kosdaq_tick_size = lambda p: 500 if p >= 200000 else 100
    def _rtt(p, d):
        tick = mock_tb.kosdaq_tick_size(p)
        return (p // tick) * tick
    mock_tb.round_to_tick = _rtt
    sys.modules["trading_bot"] = mock_tb

    try:
        result = handle_unfilled_orders(client, state, current_price=190000)
        assert result["modified"] == 1
        assert result["cancelled"] == 0

        body = client.post_calls[0]["body"]
        assert body["RVSE_CNCL_DVSN_CD"] == "01"
        new_price = int(body["ORD_UNPR"])
        assert new_price < 192000
    finally:
        del sys.modules["trading_bot"]

    print("✅ handle: 손절 미체결 정정")


def test_handle_sell_tp_cancel():
    """미체결 TP1/TRAIL → 취소."""
    client = MockKISClient()
    client.set_get("TTTC0081R", {
        "rt_cd": "0",
        "output1": [{
            "odno": "TP001", "ord_gno_brno": "00950", "pdno": "247540",
            "sll_buy_dvsn_cd": "01",
            "ord_qty": "5", "tot_ccld_qty": "0", "rmn_qty": "5",
            "ord_unpr": "222000", "avg_prvs": "0", "cncl_yn": "N",
        }],
    })
    client.set_post("TTTC0803U", {"rt_cd": "0"})

    state = MockBotState(
        position_qty=5,              # 5주 복구 필요 (사전 5주 차감)
        entry_price=200000.0,
        cash=-390_000,               # 1.5M - 2M(매수) + 1.11M(TP1 5주) = 610K... 간이값
        daily_pnl=110_000,           # (222-200) × 5 사전반영
        weekly_pnl=110_000,
        monthly_pnl=110_000,
        pending_orders=[asdict(PendingOrder(
            order_no="TP001", org_no="00950",
            side="sell", reason="TP1", qty=5, price=222000,
            remaining_qty=5, status="pending",
        ))],
    )

    result = handle_unfilled_orders(client, state, current_price=220000)
    assert result["cancelled"] == 1
    assert state.position_qty == 10          # 5 + 5 복구
    assert state.daily_pnl == 0              # pnl 복구
    print("✅ handle: TP1 미체결 취소 + pnl 복구")


def test_handle_filled_no_action():
    """이미 완전 체결된 주문 → 추적 종료만."""
    client = MockKISClient()
    client.set_get("TTTC0081R", {
        "rt_cd": "0",
        "output1": [{
            "odno": "OK001", "ord_gno_brno": "00950", "pdno": "247540",
            "sll_buy_dvsn_cd": "02",
            "ord_qty": "10", "tot_ccld_qty": "10", "rmn_qty": "0",
            "ord_unpr": "200000", "avg_prvs": "199800", "cncl_yn": "N",
        }],
    })

    state = MockBotState(
        position_qty=10,
        entry_price=200000.0,
        cash=-500_000,
        pending_orders=[asdict(PendingOrder(
            order_no="OK001", org_no="00950",
            side="buy", reason="ENTRY", qty=10, price=200000,
            remaining_qty=10, status="pending",
        ))],
    )

    result = handle_unfilled_orders(client, state, current_price=200000)
    assert result["filled"] == 1
    assert result["cancelled"] == 0
    assert result["modified"] == 0
    assert state.position_qty == 10            # 변경 없음
    assert len(state.pending_orders) == 0      # 추적 종료
    print("✅ handle: 완전 체결 주문 추적 종료")


def run_all():
    print(f"\n{'='*55}")
    print("Phase 2-3 미체결 주문 관리 검증")
    print(f"{'='*55}\n")

    tests = [
        test_pending_creation_success,
        test_pending_creation_failure,
        test_query_filled,
        test_query_partial,
        test_query_ticker_filter,
        test_cancel_order,
        test_modify_order,
        test_sync_buy_full_unfill,
        test_sync_buy_partial,
        test_sync_sell_full_unfill_restores_pnl,
        test_handle_buy_unfilled_cancel,
        test_handle_sell_sl_modify,
        test_handle_sell_tp_cancel,
        test_handle_filled_no_action,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            print(f"❌ {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*55}")
    print(f"결과: {passed} 통과 / {failed} 실패 (총 {len(tests)}개)")
    print(f"{'='*55}")
    return failed == 0


if __name__ == "__main__":
    exit(0 if run_all() else 1)
