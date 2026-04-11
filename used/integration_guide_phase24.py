"""
Phase 2-4 리포트 & 성과 집계 — trading_bot.py 연동 가이드
==========================================================
3개 지점만 수정하면 됩니다.
"""

# ─────────────────────────────────────────────────────────
# 1. 파일 상단 import 추가
# ─────────────────────────────────────────────────────────
"""
from reporter import (
    log_trade,
    format_daily_report,
    format_weekly_report,
    format_monthly_report,
    should_send_weekly_report,
    should_send_monthly_report,
)
"""


# ─────────────────────────────────────────────────────────
# 2. _execute_buy / _execute_sell에 log_trade 호출 추가
# ─────────────────────────────────────────────────────────
"""
def _execute_buy(client, state, price, qty, regime, today):
    try:
        # ... 기존 주문 로직 ...
        if resp.get("rt_cd") == "0":
            cost = qty * price
            state.cash -= cost
            state.position_qty = qty
            state.entry_price = float(price)
            # ... 기존 상태 업데이트 ...

            # ★ 추가: 매매 이력 기록
            log_trade(
                state_dir=str(STATE_DIR),
                side="buy", reason="ENTRY",
                price=price, qty=qty,
                regime=regime, signal=state.last_signal,
            )

            # ★ Phase 2-3 PendingOrder 등록 (이미 있다면 유지)
            pending = create_pending_from_response(resp, "buy", "ENTRY", qty, price)
            if pending:
                state.pending_orders.append(asdict(pending))
"""

"""
def _execute_sell(client, params, state, price, qty, reason, regime, today):
    try:
        # ... 기존 주문 로직 ...
        if resp.get("rt_cd") == "0":
            proceeds = qty * price
            pnl = (price - state.entry_price) * qty
            entry_price_snapshot = state.entry_price  # ★ 청산 전 진입가 보관
            entry_date_snapshot = state.entry_date

            state.cash += proceeds
            state.position_qty -= qty
            # ... 기존 상태 업데이트 ...

            # ★ 추가: 매매 이력 기록 (entry_price/date는 reset 전 값 사용)
            log_trade(
                state_dir=str(STATE_DIR),
                side="sell", reason=reason,
                price=price, qty=qty, pnl=int(pnl),
                regime=regime, signal=state.last_signal,
                entry_price=entry_price_snapshot,
                entry_date=entry_date_snapshot,
            )
"""


# ─────────────────────────────────────────────────────────
# 3. _run_closing 수정 — 종합 리포트로 교체
# ─────────────────────────────────────────────────────────
"""
def _run_closing(client, params, state, today):
    # 시세 조회
    price = get_current_price(client)
    current = price["current"]

    if state.position_qty > 0 and current > 0:
        state.highest_since_entry = max(state.highest_since_entry, current)

    # Phase 2-3: 미체결 주문 처리
    unfilled_result = handle_unfilled_orders(client, state, current_price=current)

    # ★ Phase 2-4: 일일 종합 리포트
    daily_msg = format_daily_report(state, current, str(STATE_DIR))

    # 미체결 처리 결과 추가
    if unfilled_result["checked"] > 0:
        daily_msg += "\\n\\n📝 주문 확인: " + str(unfilled_result["checked"]) + "건"
        if unfilled_result["filled"] > 0:
            daily_msg += f"\\n  ✅ 체결: {unfilled_result['filled']}"
        if unfilled_result["cancelled"] > 0:
            daily_msg += f"\\n  🚫 취소: {unfilled_result['cancelled']}"
        if unfilled_result["modified"] > 0:
            daily_msg += f"\\n  ✏️ 정정: {unfilled_result['modified']}"

    send_telegram(daily_msg)

    # ★ 금요일이면 주간 리포트 추가 발송
    if should_send_weekly_report():
        weekly_msg = format_weekly_report(state, str(STATE_DIR))
        send_telegram(weekly_msg)
        log.info("[REPORT] 주간 리포트 발송 완료")

    # ★ 월말이면 월간 리포트 추가 발송
    if should_send_monthly_report():
        monthly_msg = format_monthly_report(state, str(STATE_DIR))
        send_telegram(monthly_msg)
        log.info("[REPORT] 월간 리포트 발송 완료")
"""


# ─────────────────────────────────────────────────────────
# 동작 흐름 요약
# ─────────────────────────────────────────────────────────
"""
[매매 발생 시]
  _execute_buy / _execute_sell
    → 한투 주문 전송 (성공 시)
    → state 업데이트 (cash, position 등)
    → log_trade() 호출 → trade_history_247540.json에 영구 기록

[15:10 closing 매일]
  _run_closing
    1. 종가 조회
    2. handle_unfilled_orders (Phase 2-3)
    3. format_daily_report → Telegram 발송
       포함 내용:
       - 평가금액, 현금, 포지션, 미실현 손익
       - 일/주/월 PnL
       - 금일 매매 내역
       - 현재 레짐
    4. 금요일이면 format_weekly_report 추가 발송
       포함 내용:
       - 주간 PnL, 매매 횟수
       - 승률, Profit Factor, 평균 익절/손절
       - 진입 레짐 분포, 청산 사유 분포
       - 백테스트 vs 실전 괴리
    5. 월말 평일이면 format_monthly_report 추가 발송
       포함 내용:
       - 월간 수익률, 매매 통계
       - 레짐별/사유별 분포
       - 백테스트 괴리 분석
       - 파라미터 재최적화 권장 여부

[누적 효과]
  - 매매 이력이 영구 보존되어 임의 기간 통계 추출 가능
  - 백테스트 기대값 대비 실전 성과의 괴리를 정량적으로 추적
  - 괴리가 임계치 초과 시 자동으로 재최적화 권장 메시지
  - CSV 내보내기로 외부 분석 도구 연동 가능
"""


# ─────────────────────────────────────────────────────────
# 부가 기능: 수동 리포트 생성
# ─────────────────────────────────────────────────────────
"""
# 임의 기간 리포트가 필요할 때 (예: 분기 리포트)
from reporter import format_monthly_report
report = format_monthly_report(state, "state",
                                month_start="2026-01-01",
                                month_end="2026-03-31")
print(report)

# CSV 내보내기 (분석/백업용)
from reporter import export_history_csv
csv_path = export_history_csv("state")  # state/trade_history_247540.csv 생성
"""
