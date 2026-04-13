"""
에코프로비엠 (247540) v4 — Phase 3-3: 뉴스 센티먼트 (Claude Haiku)
====================================================================
Google News RSS로 2차전지/에코프로비엠 관련 뉴스를 수집하고,
Claude Haiku API로 각 뉴스의 주가 영향도를 -2~+2로 스코어링.

역할:
  - 새로운 진입 시그널 생성 X
  - 기존 시그널의 '게이트' 및 '포지션 크기 조절자' O

합산 점수에 따른 multiplier:
  score >= +3 → 1.2  (상승 뉴스 과다, 포지션 확대)
  -2 ≤ score ≤ +2 → 1.0 (중립)
  score ≤ -3  → 0.5  (부정 뉴스, 포지션 절반)
  score ≤ -6  → 진입 차단 (BLOCKED_NEG_NEWS)

비용 관리:
  - URL 기준 dedup → 이미 점수화한 기사 재호출 X
  - 일 실행 1회 (morning)만 호출
  - Haiku 기준 월 $1 미만 예상

의존성:
  pip install anthropic feedparser

환경변수:
  ANTHROPIC_API_KEY — 필수. 없으면 neutral fallback
"""

import json
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import quote

log = logging.getLogger(__name__)

TICKER = "247540"
STATE_FILE = f"news_sentiment_{TICKER}.json"

# 수집 쿼리 (Google News RSS)
DEFAULT_QUERIES = [
    "에코프로비엠",
    "양극재 2차전지",
    "SK온 배터리",
    "IRA 전기차 보조금",
    "니켈 가격 배터리",
]

# Claude Haiku 모델
HAIKU_MODEL = "claude-haiku-4-5"

# 스코어링 프롬프트
SCORING_PROMPT = """다음 뉴스가 코스닥 상장사 에코프로비엠(247540, 2차전지 양극재 제조)의 단기(1~5영업일) 주가에 미치는 영향을 평가하세요.

평가 척도:
  -2: 매우 부정 (직접적 악재, 실적 하향, 판매량 급감, 공급 차질 등)
  -1: 부정 (업황 둔화, 경쟁 심화, 원자재 가격 불리)
   0: 중립 (무관하거나 영향 미미)
  +1: 긍정 (업황 개선, 정책 지원, 수주 증가)
  +2: 매우 긍정 (대형 계약, 실적 상향, 강력한 정책 호재)

뉴스 제목: {title}
요약: {summary}

JSON 한 줄로만 응답하세요. 다른 텍스트 금지.
{{"score": N, "reason": "간단한 이유 (20자 이내)"}}"""


# ═══════════════════════════════════════════════════════════════════
# 1. 뉴스 수집
# ═══════════════════════════════════════════════════════════════════

def fetch_news(queries: list = None, lookback_hours: int = 24) -> list:
    """
    Google News RSS로 뉴스 수집. 최근 N시간 이내만 반환.

    Returns:
        list of {"title", "summary", "url", "published", "query"}
    """
    try:
        import feedparser
    except ImportError:
        log.warning("[NEWS] feedparser 미설치, 뉴스 수집 건너뜀")
        return []

    if queries is None:
        queries = DEFAULT_QUERIES

    cutoff = datetime.now() - timedelta(hours=lookback_hours)
    items = []
    seen_urls = set()

    for q in queries:
        url = f"https://news.google.com/rss/search?q={quote(q)}&hl=ko&gl=KR&ceid=KR:ko"
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            log.warning(f"[NEWS] RSS 실패 ({q}): {e}")
            continue

        for entry in feed.entries[:30]:
            link = entry.get("link", "")
            if not link or link in seen_urls:
                continue
            seen_urls.add(link)

            # 발행일 파싱
            published_dt = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    published_dt = datetime(*entry.published_parsed[:6])
                except Exception:
                    pass

            if published_dt and published_dt < cutoff:
                continue

            # HTML 태그 제거
            summary_raw = entry.get("summary", "")
            summary = re.sub(r"<[^>]+>", "", summary_raw)[:300]

            items.append({
                "title": entry.get("title", "")[:200],
                "summary": summary,
                "url": link,
                "published": published_dt.isoformat() if published_dt else "",
                "query": q,
            })

    log.info(f"[NEWS] 수집 완료: {len(items)}건 ({len(queries)}개 쿼리)")
    return items


# ═══════════════════════════════════════════════════════════════════
# 2. Claude Haiku 스코어링
# ═══════════════════════════════════════════════════════════════════

def score_article(client, title: str, summary: str) -> Optional[dict]:
    """
    단일 기사 스코어링. 실패 시 None.

    Returns:
        {"score": -2..+2, "reason": "..."} or None
    """
    prompt = SCORING_PROMPT.format(title=title, summary=summary[:400])
    try:
        resp = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()

        # JSON 추출 (모델이 ```json 감싸는 경우 대응)
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = text.replace("```", "").strip()

        parsed = json.loads(text)
        score = int(parsed.get("score", 0))
        score = max(-2, min(2, score))   # clamp
        return {
            "score": score,
            "reason": str(parsed.get("reason", ""))[:100],
        }
    except Exception as e:
        log.warning(f"[NEWS] 스코어링 실패: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# 3. 상태 저장 / 로드
# ═══════════════════════════════════════════════════════════════════

def load_state(state_dir: str) -> dict:
    path = Path(state_dir) / STATE_FILE
    if not path.exists():
        return {"articles": {}, "last_run": ""}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"articles": {}, "last_run": ""}


def save_state(state_dir: str, state: dict):
    path = Path(state_dir) / STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    # 7일 이상 지난 기사는 정리
    cutoff = (datetime.now() - timedelta(days=7)).isoformat()
    state["articles"] = {
        url: art for url, art in state["articles"].items()
        if art.get("published", "") >= cutoff or art.get("scored_at", "") >= cutoff
    }
    state["last_run"] = datetime.now().isoformat()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════
# 4. 메인: get_sentiment_signal
# ═══════════════════════════════════════════════════════════════════

def get_sentiment_signal(state_dir: str = "state",
                          queries: list = None,
                          max_new_per_run: int = 15) -> dict:
    """
    trading_bot._run_morning()에서 호출하는 메인 함수.

    동작:
      1. 뉴스 수집 (24h 이내)
      2. 기존 캐시와 비교, 신규 기사만 Haiku로 스코어링
      3. 최근 24h 기사들의 합산 점수 계산
      4. multiplier + block 플래그 반환

    Returns:
        {
            "score": int,           # -30..+30 정도 (합산)
            "multiplier": float,    # 0.5 | 1.0 | 1.2
            "block_entry": bool,
            "n_articles": int,
            "n_new": int,
            "n_cached": int,
            "top_negative": list,   # 최악 뉴스 3개
            "top_positive": list,   # 최고 뉴스 3개
            "source": "haiku" | "fallback_no_key" | "fallback_no_sdk",
        }
    """
    state = load_state(state_dir)
    articles_map = state.get("articles", {})

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        log.warning("[NEWS] ANTHROPIC_API_KEY 없음, neutral fallback")
        return _neutral_result(source="fallback_no_key")

    try:
        import anthropic
    except ImportError:
        log.warning("[NEWS] anthropic SDK 미설치, neutral fallback")
        return _neutral_result(source="fallback_no_sdk")

    # 1. 뉴스 수집
    new_items = fetch_news(queries)
    if not new_items and not articles_map:
        return _neutral_result(source="haiku", n_articles=0)

    # 2. 캐시에 없는 신규 기사만 스코어링
    to_score = [a for a in new_items if a["url"] not in articles_map]
    to_score = to_score[:max_new_per_run]   # 비용 상한

    if to_score:
        try:
            client = anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            log.error(f"[NEWS] Anthropic client 생성 실패: {e}")
            return _neutral_result(source="fallback_no_sdk")

        scored_count = 0
        for art in to_score:
            result = score_article(client, art["title"], art["summary"])
            if result is None:
                continue
            art["score"] = result["score"]
            art["reason"] = result["reason"]
            art["scored_at"] = datetime.now().isoformat()
            articles_map[art["url"]] = art
            scored_count += 1
        log.info(f"[NEWS] Haiku 스코어링: {scored_count}/{len(to_score)}건")

    state["articles"] = articles_map
    save_state(state_dir, state)

    # 3. 최근 24시간 기사 합산
    cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
    recent = [
        a for a in articles_map.values()
        if (a.get("published", "") >= cutoff or a.get("scored_at", "") >= cutoff)
        and "score" in a
    ]

    total_score = sum(a["score"] for a in recent)

    # 4. multiplier / block 결정
    if total_score <= -6:
        multiplier = 0.5
        block = True
    elif total_score <= -3:
        multiplier = 0.5
        block = False
    elif total_score >= 3:
        multiplier = 1.2
        block = False
    else:
        multiplier = 1.0
        block = False

    # 상위/하위 뉴스 추출
    sorted_desc = sorted(recent, key=lambda x: -x["score"])
    sorted_asc = sorted(recent, key=lambda x: x["score"])
    top_pos = [
        {"title": a["title"][:60], "score": a["score"], "reason": a.get("reason", "")}
        for a in sorted_desc if a["score"] > 0
    ][:3]
    top_neg = [
        {"title": a["title"][:60], "score": a["score"], "reason": a.get("reason", "")}
        for a in sorted_asc if a["score"] < 0
    ][:3]

    return {
        "score": total_score,
        "multiplier": multiplier,
        "block_entry": block,
        "n_articles": len(recent),
        "n_new": len(to_score),
        "n_cached": len(recent) - len(to_score) if len(recent) >= len(to_score) else 0,
        "top_positive": top_pos,
        "top_negative": top_neg,
        "source": "haiku",
    }


def _neutral_result(source: str, n_articles: int = 0) -> dict:
    return {
        "score": 0,
        "multiplier": 1.0,
        "block_entry": False,
        "n_articles": n_articles,
        "n_new": 0,
        "n_cached": 0,
        "top_positive": [],
        "top_negative": [],
        "source": source,
    }


# ═══════════════════════════════════════════════════════════════════
# 5. Telegram 포맷
# ═══════════════════════════════════════════════════════════════════

def format_sentiment_telegram(sent: dict) -> str:
    """morning 메시지에 끼워넣을 센티먼트 블록."""
    if sent.get("source") != "haiku":
        return f"📰 뉴스 센티먼트: N/A ({sent.get('source', '?')})"

    if sent["n_articles"] == 0:
        return "📰 뉴스 센티먼트: 최근 24h 뉴스 없음"

    emoji = "🟢" if sent["score"] >= 3 else "🔴" if sent["score"] <= -3 else "⚪"
    lines = [
        f"📰 <b>뉴스 센티먼트</b> {emoji}",
        f"  합산: {sent['score']:+d} (n={sent['n_articles']}, new={sent['n_new']})",
        f"  가중치: ×{sent['multiplier']}"
        + (" 🚫 진입차단" if sent["block_entry"] else ""),
    ]

    if sent["top_negative"]:
        lines.append("  ↓ 부정:")
        for a in sent["top_negative"][:2]:
            lines.append(f"    [{a['score']:+d}] {a['title']}")
    if sent["top_positive"]:
        lines.append("  ↑ 긍정:")
        for a in sent["top_positive"][:2]:
            lines.append(f"    [{a['score']:+d}] {a['title']}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# CLI (디버그용)
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="뉴스 센티먼트 디버그")
    parser.add_argument("--state-dir", default="state")
    args = parser.parse_args()

    result = get_sentiment_signal(args.state_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()
    print(format_sentiment_telegram(result))
