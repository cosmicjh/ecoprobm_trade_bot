# 에코프로비엠 (247540) v4 자동트레이딩 — Phase 1-1

## 데이터 수집 파이프라인

### 수집 데이터

| 구분 | 소스 | 주기 | 용도 |
|------|------|------|------|
| 일봉 OHLCV | FinanceDataReader | 장 마감 후 1회 | 기술적 지표 계산, 레짐 판별 |
| 투자자별 매매동향 | 한투 API (`FHKST01010900`) | 장중 4회 | 외국인·기관 수급 시그널 |
| 공매도 거래량 | 한투 API (`FHKST03060100`) | 장 마감 후 1회 | 숏커버링 시그널 |
| KODEX 2차전지 ETF | FinanceDataReader | 장 마감 후 1회 | 섹터 자금 유입 프록시 |
| KOSDAQ 지수 (KQ11) | FinanceDataReader | 장 마감 후 1회 | 레짐 판별 보조 |

### 파일 구조

```
ecoprobm_v4/
├── data_collector.py           # 메인 수집 파이프라인
├── test_data_collector.py      # 로컬 검증 스크립트 (9개 테스트)
├── .github/
│   └── workflows/
│       └── ecoprobm_v4_collect.yml  # GitHub Actions 워크플로우
└── state/                      # 상태 파일 (자동 생성)
    ├── ohlcv_247540.json       # 일봉 데이터 (날짜별)
    ├── supply_data_247540.json # 수급+공매도 데이터 (날짜별 누적)
    └── pipeline_state_247540.json  # 파이프라인 메타 상태
```

### 핵심 설계 원칙

1. **look-ahead bias 방지**: 모든 FDR 데이터는 `.shift(1)` 적용.
   전략 시그널 계산에는 반드시 `prev_*` 컬럼만 사용.

2. **누적 저장**: 수급/공매도 데이터는 날짜를 키로 JSON에 누적.
   incremental 수집 시 동일 날짜는 최신 값으로 덮어씀.

3. **독립적 수집**: OHLCV, 수급, 공매도, ETF 각각 독립 try-except.
   하나가 실패해도 나머지는 정상 수집.

4. **상태 추적**: pipeline_state 파일에 마지막 수집 시점, 누적 건수, 오류 이력 기록.

### 환경변수

```bash
# 한투 API (필수)
KIS_API_KEY=your_api_key
KIS_API_SECRET=your_api_secret
KIS_ACC_NO=12345678-01    # 하이픈 필수!
KIS_MOCK=Y                # 모의투자: Y, 실전: N

# Telegram 알림 (선택)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# 상태 파일 경로 (선택, 기본값: 현재 디렉토리)
STATE_DIR=state
```

### 실행 방법

```bash
# 로컬 테스트 (API 호출 없이 로직 검증)
python test_data_collector.py

# 전체 데이터 수집 (초기 구축)
python data_collector.py --mode full --state-dir state

# 증분 수집 (일일 운영)
python data_collector.py --mode incremental --state-dir state
```

### cron-job.org 설정

| 시간 (KST) | 목적 | mode |
|-------------|------|------|
| 09:05 | 전일 확정 OHLCV + 장 초반 수급 | incremental |
| 15:10 | 당일 확정 데이터 전체 | incremental |

### 상태 파일 스키마

**supply_data_247540.json** (수급 시그널 핵심 데이터):
```json
{
  "2026-04-07": {
    "foreign_net_qty": 46800,
    "foreign_net_amt": 9500000000,
    "inst_net_qty": -3000,
    "inst_net_amt": -610000000,
    "individual_net_qty": -43800,
    "individual_net_amt": -8890000000,
    "short_volume": 25000,
    "short_amount": 5000000000,
    "total_volume": 500000,
    "short_ratio": 5.0
  }
}
```

### 다음 단계 (Phase 1-2)

Phase 1-1에서 수집한 데이터를 기반으로:
- MA(20/60), BB(20,2), ATR(14), RSI(14) 기술적 지표 계산 모듈
- 모든 지표는 `prev_*` 컬럼에서만 계산
- Colab 백테스트 환경에서 레짐 판별 정확도 검증
