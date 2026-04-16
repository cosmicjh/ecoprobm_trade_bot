"""
에코프로비엠 (247540) v4 — Phase 4-6: 대시보드 빌더
========================================================
state/ 디렉토리의 JSON 파일들을 읽어 docs/index.html을 생성한다.
GitHub Pages가 docs/를 자동으로 발행하므로,
빌드 후 git commit/push만 하면 즉시 웹에 반영된다.

설계:
  - 단일 HTML 파일 (의존성 없음, 데이터 인라인)
  - Chart.js (CDN) 만 사용
  - 다크 테마 (모바일 가독성)
  - 모든 상태 데이터를 window.__DATA__ 전역 객체에 박아넣음
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

TICKER = "247540"
TICKER_NAME = "에코프로비엠"


def _safe_load(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"[DASH] {path.name} 로드 실패: {e}")
        return default


def _read_jsonl(path: Path, n: int = 200) -> list:
    if not path.exists():
        return []
    runs = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines()[-n:]:
                try:
                    runs.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass
    return runs


def collect_dashboard_data(state_dir: str) -> dict:
    """state/ 의 모든 관련 파일을 읽어 단일 dict 로 합친다."""
    sd = Path(state_dir)

    bot_state = _safe_load(sd / f"bot_state_{TICKER}.json", {})
    trade_history = _safe_load(sd / f"trade_history_{TICKER}.json", [])
    predictions = _safe_load(sd / f"predictions_{TICKER}.json", [])
    retrain_history = _safe_load(sd / f"retrain_history_{TICKER}.json", [])
    opt_history = _safe_load(sd / f"optimization_history_{TICKER}.json", [])
    optimized_params = _safe_load(sd / f"optimized_params_{TICKER}.json", {})
    news_state = _safe_load(sd / f"news_sentiment_{TICKER}.json", {})
    run_logs = _read_jsonl(sd / f"run_log_{TICKER}.jsonl", n=200)

    # OHLCV (가격 차트용 — 최근 90일만)
    ohlcv_full = _safe_load(sd / f"ohlcv_{TICKER}.json", {})
    ohlcv_recent = {}
    if ohlcv_full:
        sorted_dates = sorted(ohlcv_full.keys())[-90:]
        ohlcv_recent = {d: ohlcv_full[d] for d in sorted_dates}

    # 자산 곡선 계산 (trade_history 기반 누적)
    equity_curve = _build_equity_curve(trade_history, bot_state)

    # 정확도 집계 (최근 30일)
    accuracy = _summarize_accuracy(predictions, days=30)

    # 뉴스 시계열 (점수만)
    news_series = _summarize_news(news_state)

    # ── KST 변환 ──
    def _to_kst(iso_str):
        if not iso_str:
            return ""
        try:
            dt = datetime.fromisoformat(iso_str.replace("Z", ""))
            return (dt + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return iso_str

    for t in trade_history[-100:]:
        if "timestamp" in t:
            t["timestamp_kst"] = _to_kst(t["timestamp"])
    for r in run_logs:
        if "timestamp" in r:
            r["timestamp_kst"] = _to_kst(r["timestamp"])
    for r in retrain_history:
        if "timestamp" in r:
            r["timestamp_kst"] = _to_kst(r["timestamp"])
  
    return {
        "ticker": TICKER,
        "ticker_name": TICKER_NAME,
        "generated_at": (datetime.utcnow() + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S KST"),
        "bot_state": bot_state,
        "trade_history": trade_history[-100:],   # 최근 100건
        "ohlcv": ohlcv_recent,
        "equity_curve": equity_curve,
        "accuracy": accuracy,
        "predictions": predictions[-60:],
        "retrain_history": retrain_history,
        "optimization_history": opt_history,
        "optimized_params": optimized_params,
        "news_series": news_series,
        "run_logs": run_logs,
    }


def _build_equity_curve(trade_history: list, bot_state: dict) -> list:
    """매매 이력으로 일별 누적 자산을 근사 계산."""
    initial = bot_state.get("initial_capital", 1_500_000)
    cash = initial
    pos = 0
    last_price = 0
    curve = []
    cur_day = None
    for t in trade_history:
        date = t.get("date", "")
        side = t.get("side", "")
        price = t.get("price", 0)
        qty = t.get("qty", 0)
        if side == "buy":
            cash -= price * qty
            pos += qty
        elif side == "sell":
            cash += price * qty
            pos -= qty
        last_price = price
        equity = cash + pos * last_price
        if cur_day != date:
            curve.append({"date": date, "equity": int(equity), "cash": int(cash), "pos": pos})
            cur_day = date
        else:
            curve[-1] = {"date": date, "equity": int(equity), "cash": int(cash), "pos": pos}
    return curve


def _summarize_accuracy(predictions: list, days: int = 30) -> dict:
    from datetime import timedelta
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    recent = [p for p in predictions if p.get("date", "") >= cutoff and p.get("evaluated")]
    if not recent:
        return {"n": 0}

    def acc(field):
        vals = [r[field] for r in recent if r.get(field) is not None]
        if not vals:
            return None
        return {"rate": round(sum(1 for v in vals if v) / len(vals), 3), "n": len(vals)}

    return {
        "n": len(recent),
        "regime_rule": acc("regime_correct"),
        "regime_hmm": acc("hmm_correct"),
        "signal": acc("signal_correct"),
        "supply": acc("supply_correct"),
    }


def _summarize_news(news_state: dict) -> list:
    """뉴스 캐시에서 일별 점수 합계 시계열 생성."""
    articles = news_state.get("articles", {})
    by_day = {}
    for art in articles.values():
        if "score" not in art:
            continue
        day = art.get("scored_at", "")[:10] or art.get("published", "")[:10]
        if not day:
            continue
        by_day.setdefault(day, []).append(art["score"])
    return [
        {"date": d, "sum": sum(scores), "n": len(scores)}
        for d, scores in sorted(by_day.items())
    ]


# ═══════════════════════════════════════════════════════════════════
# HTML 생성
# ═══════════════════════════════════════════════════════════════════

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{ticker_name} 자동매매 대시보드</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0f1419; --bg2: #1a1f2e; --border: #2a3040;
    --text: #e4e6eb; --text2: #8b949e;
    --green: #3fb950; --red: #f85149; --blue: #58a6ff; --yellow: #d29922;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    background: var(--bg); color: var(--text); padding: 16px; max-width: 1100px; margin: 0 auto;
  }}
  h1 {{ font-size: 22px; margin: 0 0 4px; font-weight: 600; }}
  h2 {{ font-size: 15px; margin: 0 0 12px; color: var(--text2); font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }}
  .meta {{ color: var(--text2); font-size: 12px; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 24px; }}
  .card {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px; }}
  .card .label {{ font-size: 11px; color: var(--text2); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }}
  .card .value {{ font-size: 22px; font-weight: 600; }}
  .card .sub {{ font-size: 11px; color: var(--text2); margin-top: 4px; }}
  .green {{ color: var(--green); }} .red {{ color: var(--red); }} .blue {{ color: var(--blue); }} .yellow {{ color: var(--yellow); }}
  .section {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 18px; margin-bottom: 20px; }}
  canvas {{ max-height: 280px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--text2); font-weight: 500; text-transform: uppercase; font-size: 10px; }}
  .tabs {{ display: flex; gap: 4px; margin-bottom: 16px; flex-wrap: wrap; }}
  .tab {{ padding: 6px 14px; background: transparent; border: 1px solid var(--border); color: var(--text2);
          border-radius: 6px; cursor: pointer; font-size: 13px; }}
  .tab.active {{ background: var(--bg2); color: var(--text); border-color: var(--blue); }}
  .tab-content {{ display: none; }} .tab-content.active {{ display: block; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 500; }}
  .badge-ok {{ background: rgba(63,185,80,0.15); color: var(--green); }}
  .badge-err {{ background: rgba(248,81,73,0.15); color: var(--red); }}
  .badge-warn {{ background: rgba(210,153,34,0.15); color: var(--yellow); }}
</style>
</head>
<body>
  <h1>{ticker_name} ({ticker}) 자동매매 대시보드</h1>
  <div class="meta">생성: <span id="genAt"></span></div>

  <div class="grid" id="kpiGrid"></div>

  <div class="tabs">
    <button class="tab active" data-tab="equity">자산</button>
    <button class="tab" data-tab="trades">매매</button>
    <button class="tab" data-tab="accuracy">정확도</button>
    <button class="tab" data-tab="ai">AI/최적화</button>
    <button class="tab" data-tab="news">뉴스</button>
    <button class="tab" data-tab="runs">실행로그</button>
  </div>

  <div class="tab-content active" id="tab-equity">
    <div class="section"><h2>자산 곡선</h2><canvas id="equityChart"></canvas></div>
    <div class="section"><h2>가격 (90일)</h2><canvas id="priceChart"></canvas></div>
  </div>

  <div class="tab-content" id="tab-trades">
    <div class="section"><h2>최근 매매 (최대 100건)</h2><div id="tradesTable"></div></div>
  </div>

  <div class="tab-content" id="tab-accuracy">
    <div class="section"><h2>예측 정확도 (최근 30일)</h2><div id="accuracyView"></div></div>
  </div>

  <div class="tab-content" id="tab-ai">
    <div class="section"><h2>AI 재학습 이력</h2><div id="retrainView"></div></div>
    <div class="section"><h2>파라미터 최적화 score 추이</h2><canvas id="optChart"></canvas></div>
    <div class="section"><h2>현재 적용 파라미터</h2><div id="paramsView"></div></div>
  </div>

  <div class="tab-content" id="tab-news">
    <div class="section"><h2>뉴스 센티먼트 (일별 합산)</h2><canvas id="newsChart"></canvas></div>
  </div>

  <div class="tab-content" id="tab-runs">
    <div class="section"><h2>최근 실행 로그</h2><div id="runsTable"></div></div>
  </div>

<script>
window.__DATA__ = {data_json};

// ── 유틸 ──
const fmt = n => n == null ? "-" : Number(n).toLocaleString();
const pct = n => n == null ? "-" : (n * 100).toFixed(1) + "%";
const cls = n => n > 0 ? "green" : n < 0 ? "red" : "";

document.getElementById("genAt").textContent = __DATA__.generated_at;

// ── KPI 카드 ──
const bs = __DATA__.bot_state;
const initial = bs.initial_capital || 1500000;
const lastEquity = (__DATA__.equity_curve.slice(-1)[0] || {{}}).equity || (bs.cash + bs.position_qty * bs.entry_price) || initial;
const totalRet = (lastEquity - initial) / initial;
const acc = __DATA__.accuracy;

const kpis = [
  {{ label: "총 자산", value: fmt(lastEquity) + "원", sub: (totalRet * 100).toFixed(2) + "%", cls: cls(totalRet) }},
  {{ label: "현금 / 보유", value: fmt(bs.cash) + "원", sub: bs.position_qty + "주" + (bs.entry_price ? " @ " + fmt(bs.entry_price) : "") }},
  {{ label: "총 매매", value: bs.total_trades || 0, sub: "최근: " + (bs.last_signal || "-") }},
  {{ label: "현재 레짐", value: bs.last_regime || "-", sub: bs.halted ? "⛔ " + bs.halt_reason : "정상" }},
  {{ label: "룰 정확도 (30d)", value: acc.regime_rule ? pct(acc.regime_rule.rate) : "-", sub: acc.regime_rule ? "n=" + acc.regime_rule.n : "" }},
  {{ label: "HMM 정확도 (30d)", value: acc.regime_hmm ? pct(acc.regime_hmm.rate) : "-", sub: acc.regime_hmm ? "n=" + acc.regime_hmm.n : "" }},
];
document.getElementById("kpiGrid").innerHTML = kpis.map(k =>
  `<div class="card"><div class="label">${{k.label}}</div><div class="value ${{k.cls || ''}}">${{k.value}}</div><div class="sub">${{k.sub || ''}}</div></div>`
).join("");

// ── 탭 전환 ──
document.querySelectorAll(".tab").forEach(t => {{
  t.onclick = () => {{
    document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(x => x.classList.remove("active"));
    t.classList.add("active");
    document.getElementById("tab-" + t.dataset.tab).classList.add("active");
  }};
}});

// ── Chart.js 공통 ──
Chart.defaults.color = "#8b949e";
Chart.defaults.borderColor = "#2a3040";
Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, sans-serif";

// ── 자산 곡선 ──
const eq = __DATA__.equity_curve;
new Chart(document.getElementById("equityChart"), {{
  type: "line",
  data: {{
    labels: eq.map(e => e.date),
    datasets: [{{ label: "Equity", data: eq.map(e => e.equity), borderColor: "#58a6ff",
                 backgroundColor: "rgba(88,166,255,0.1)", fill: true, tension: 0.2 }}]
  }},
  options: {{ plugins: {{ legend: {{ display: false }} }} }}
}});

// ── 가격 차트 ──
const ohlcvDates = Object.keys(__DATA__.ohlcv);
const ohlcvCloses = ohlcvDates.map(d => __DATA__.ohlcv[d].Close);
new Chart(document.getElementById("priceChart"), {{
  type: "line",
  data: {{
    labels: ohlcvDates,
    datasets: [{{ label: "Close", data: ohlcvCloses, borderColor: "#3fb950",
                 backgroundColor: "rgba(63,185,80,0.1)", fill: true, tension: 0.2 }}]
  }},
  options: {{ plugins: {{ legend: {{ display: false }} }} }}
}});

// ── 매매 테이블 ──
document.getElementById("tradesTable").innerHTML = `<table>
  <tr><th>일자</th><th>구분</th><th>사유</th><th>수량</th><th>가격</th><th>손익</th><th>레짐</th></tr>
  ${{__DATA__.trade_history.slice().reverse().map(t => `
    <tr>
      <td>${{t.date}}</td>
      <td>${{t.side === 'buy' ? '매수' : '매도'}}</td>
      <td>${{t.reason || '-'}}</td>
      <td>${{t.qty || 0}}</td>
      <td>${{fmt(t.price)}}</td>
      <td class="${{t.pnl > 0 ? 'green' : t.pnl < 0 ? 'red' : ''}}">${{t.pnl ? fmt(t.pnl) : '-'}}</td>
      <td>${{t.regime || '-'}}</td>
    </tr>`).join('')}}
</table>`;

// ── 정확도 ──
const renderAcc = () => {{
  const a = __DATA__.accuracy;
  if (!a.n) return '<div class="meta">평가 가능 데이터 없음</div>';
  const rows = [
    ["룰 레짐", a.regime_rule],
    ["HMM 레짐", a.regime_hmm],
    ["매수 시그널", a.signal],
    ["수급 이상", a.supply],
  ];
  return `<table><tr><th>지표</th><th>정확도</th><th>표본</th></tr>
    ${{rows.map(([k, v]) => `<tr><td>${{k}}</td><td>${{v ? pct(v.rate) : '-'}}</td><td>${{v ? v.n : 0}}</td></tr>`).join('')}}
  </table>`;
}};
document.getElementById("accuracyView").innerHTML = renderAcc();

// ── 재학습 이력 ──
document.getElementById("retrainView").innerHTML = `<table>
  <tr><th>일시</th><th>성공</th><th>HMM states</th><th>현재 레짐</th><th>LL</th></tr>
  ${{__DATA__.retrain_history.slice().reverse().slice(0, 20).map(r => `
    <tr>
      <td>${{(r.timestamp || '').slice(0, 16).replace('T', ' ')}}</td>
      <td>${{r.success ? '<span class="badge badge-ok">OK</span>' : '<span class="badge badge-err">FAIL</span>'}}</td>
      <td>${{r.hmm_n_states || '-'}}</td>
      <td>${{r.hmm_current_regime || '-'}} ${{r.hmm_current_confidence ? '(' + (r.hmm_current_confidence * 100).toFixed(0) + '%)' : ''}}</td>
      <td>${{r.hmm_log_likelihood ? r.hmm_log_likelihood.toFixed(1) : '-'}}</td>
    </tr>`).join('')}}
</table>`;

// ── 최적화 score 차트 ──
const opt = __DATA__.optimization_history;
new Chart(document.getElementById("optChart"), {{
  type: "line",
  data: {{
    labels: opt.map(o => (o.timestamp || '').slice(0, 10)),
    datasets: [
      {{ label: "기존 score", data: opt.map(o => o.old_score), borderColor: "#8b949e", borderDash: [5, 5] }},
      {{ label: "신규 score", data: opt.map(o => o.new_score), borderColor: "#58a6ff" }}
    ]
  }}
}});

// ── 파라미터 ──
const params = (__DATA__.optimized_params.params) || {{}};
document.getElementById("paramsView").innerHTML = `<table>
  ${{Object.entries(params).map(([k, v]) => `<tr><td>${{k}}</td><td>${{typeof v === 'number' ? v.toFixed(3) : v}}</td></tr>`).join('')}}
</table>`;

// ── 뉴스 차트 ──
const news = __DATA__.news_series;
new Chart(document.getElementById("newsChart"), {{
  type: "bar",
  data: {{
    labels: news.map(n => n.date),
    datasets: [{{
      label: "센티먼트 합계",
      data: news.map(n => n.sum),
      backgroundColor: news.map(n => n.sum >= 0 ? "rgba(63,185,80,0.6)" : "rgba(248,81,73,0.6)"),
    }}]
  }}
}});

// ── 실행 로그 ──
document.getElementById("runsTable").innerHTML = `<table>
  <tr><th>시각</th><th>모드</th><th>상태</th><th>소요</th><th>시그널</th><th>레짐</th><th>에러</th></tr>
  ${{__DATA__.run_logs.slice().reverse().slice(0, 80).map(r => `
    <tr>
      <td>${{(r.timestamp || '').slice(5, 16).replace('T', ' ')}}</td>
      <td>${{r.mode}}</td>
      <td><span class="badge badge-${{r.status === 'ok' ? 'ok' : r.status === 'error' ? 'err' : 'warn'}}">${{r.status}}</span></td>
      <td>${{r.duration_sec}}s</td>
      <td>${{r.signal || '-'}}</td>
      <td>${{r.regime || '-'}}</td>
      <td style="color:#f85149">${{r.error || ''}}</td>
    </tr>`).join('')}}
</table>`;
</script>
</body>
</html>"""


def build_dashboard(state_dir: str = "state",
                     output_dir: str = "docs") -> str:
    """대시보드 빌드. docs/index.html 생성."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = collect_dashboard_data(state_dir)
    data_json = json.dumps(data, ensure_ascii=False, default=str)

    html = HTML_TEMPLATE.format(
        ticker=TICKER,
        ticker_name=TICKER_NAME,
        data_json=data_json,
    )

    output_path = out / "index.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    log.info(f"[DASH] 대시보드 생성: {output_path} ({len(html):,} bytes)")
    return str(output_path)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=f"{TICKER_NAME} v4 대시보드 빌더")
    parser.add_argument("--state-dir", default="state")
    parser.add_argument("--output-dir", default="docs")
    args = parser.parse_args()
    build_dashboard(args.state_dir, args.output_dir)

def _to_kst(iso_str):
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", ""))
        return (dt + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return iso_str

# 매매 이력 timestamp 변환
for t in trade_history[-100:]:
    if "timestamp" in t:
        t["timestamp_kst"] = _to_kst(t["timestamp"])

# 실행 로그 변환
for r in run_logs:
    if "timestamp" in r:
        r["timestamp_kst"] = _to_kst(r["timestamp"])

# retrain 변환
for r in retrain_history:
    if "timestamp" in r:
        r["timestamp_kst"] = _to_kst(r["timestamp"])
