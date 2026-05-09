"""Collect contested and stable German Wikipedia article datasets.

This module mirrors the English dataset schema used in the project while
adapting collection sources and dispute detection to German Wikipedia.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests


LANG = "de"
API_URL = f"https://{LANG}.wikipedia.org/w/api.php"
LIFTWING_URL = (
    "https://api.wikimedia.org/service/lw/inference/v1/"
    "models/outlink-topic-model:predict"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"

CONTESTED_FILE = DATA_RAW_DIR / "final_contested_de.json"
STABLE_FILE = DATA_RAW_DIR / "final_stable_de.json"
STABLE_TITLES_FILE = DATA_RAW_DIR / "final_stable_titles_de.json"

TARGET_CONTESTED = 200
TARGET_STABLE = 200

MIN_WORDS = 1500
MIN_EDITORS = 15
MIN_AGE_DAYS = 365
MAX_WORDS_STABLE = 8000

REQUEST_SLEEP_SECONDS = 1.0
ARTICLE_SLEEP_MIN_SECONDS = 3.0
ARTICLE_SLEEP_MAX_SECONDS = 6.0

HEADERS = {
    "User-Agent": (
        "Wiki-Trends-Analysis/1.0 "
        "(German Wikipedia research dataset; contact: repository owner)"
    )
}

GERMAN_DISPUTE_TEMPLATES = [
    "Vorlage:Neutralität",
]

GERMAN_DISPUTE_TEMPLATE_NAMES = [
    "Neutralität",
    "Neutralitaet",
]

GERMAN_DISPUTE_HISTORY_TEMPLATE_NAMES = [
    "Neutralität",
    "Neutralitaet",
    "Überarbeiten",
    "Ueberarbeiten",
]

GERMAN_STABLE_CATEGORIES = [
    "Kategorie:Wikipedia:Exzellent",
    "Kategorie:Wikipedia:Lesenswert",
]

TOPIC_BROAD = {
    "STEM": "science",
    "Culture": "culture",
    "Geography": "geography",
    "History_and_Society": "politics_history",
}

TOPIC_QUOTAS_CONTESTED = {
    "politics_history": 50,
    "culture": 50,
    "geography": 50,
    "science": 50,
}

TOPIC_QUOTAS_STABLE = {
    "politics_history": 100,
    "culture": 100,
    "geography": 100,
    "science": 100,
}


def safe_get(
    url: str,
    params: dict[str, Any],
    *,
    retries: int = 5,
    base_wait: float = 3.0,
) -> dict[str, Any] | None:
    """Run a MediaWiki GET request with retries and rate-limit handling."""
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=30,
            )
            if response.status_code == 429:
                wait_seconds = 180
                print(f"Rate limited; waiting {wait_seconds}s")
                time.sleep(wait_seconds)
                continue
            if response.status_code == 200 and response.text.strip():
                return response.json()
            print(
                f"GET failed ({response.status_code}) on attempt "
                f"{attempt + 1}/{retries}: {params}"
            )
        except requests.RequestException as exc:
            print(f"GET error on attempt {attempt + 1}/{retries}: {exc}")
        except json.JSONDecodeError as exc:
            print(f"JSON decode error on attempt {attempt + 1}/{retries}: {exc}")
        time.sleep(base_wait * (attempt + 1))
    return None


def safe_post_json(
    url: str,
    payload: dict[str, Any],
    *,
    retries: int = 3,
    base_wait: float = 2.0,
) -> dict[str, Any] | None:
    """Run a JSON POST request with retries."""
    headers = {**HEADERS, "Content-Type": "application/json"}
    for attempt in range(retries):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=30,
            )
            if response.status_code == 200 and response.text.strip():
                return response.json()
            print(
                f"POST failed ({response.status_code}) on attempt "
                f"{attempt + 1}/{retries}: {payload}"
            )
        except requests.RequestException as exc:
            print(f"POST error on attempt {attempt + 1}/{retries}: {exc}")
        except json.JSONDecodeError as exc:
            print(f"JSON decode error on attempt {attempt + 1}/{retries}: {exc}")
        time.sleep(base_wait * (attempt + 1))
    return None


def save_json(data: Any, path: Path) -> None:
    """Atomically save JSON so interrupted writes do not corrupt outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_json(path: Path, default: Any | None = None) -> Any:
    if path.exists():
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    return [] if default is None else default


def clean_wikitext(text: str) -> str:
    """Strip common MediaWiki markup while preserving readable article text."""
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"==+[^=]+=+", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^/]*/\s*>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"'{2,}", "", text)
    return re.sub(r"\s+", " ", text).strip()


def get_page_value(data: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return next(iter(data["query"]["pages"].values()))
    except (KeyError, StopIteration, AttributeError):
        return None


def get_revision_content(revision: dict[str, Any]) -> str:
    slots = revision.get("slots")
    if slots and "main" in slots:
        return slots["main"].get("*", "")
    return revision.get("*", "")


def get_talk_title(title: str) -> str:
    return f"Diskussion:{title}"


def has_current_dispute_template(raw_text: str) -> bool:
    escaped = [re.escape(name) for name in GERMAN_DISPUTE_TEMPLATE_NAMES]
    pattern = r"\{\{\s*(?:" + "|".join(escaped) + r")\b"
    return re.search(pattern, raw_text, re.IGNORECASE) is not None


def has_dispute_history(title: str, *, revision_limit: int = 50) -> bool:
    """Reject stable articles with recent German dispute/cleanup templates."""
    escaped = [re.escape(name) for name in GERMAN_DISPUTE_HISTORY_TEMPLATE_NAMES]
    pattern = re.compile(r"\{\{\s*(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)
    data = safe_get(
        API_URL,
        {
            "action": "query",
            "titles": title,
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "rvlimit": revision_limit,
            "format": "json",
        },
    )
    if not data:
        return False
    page = get_page_value(data)
    if not page:
        return False
    for revision in page.get("revisions", []):
        if pattern.search(get_revision_content(revision)):
            return True
    return False


def get_topic_liftwing(title: str, *, retries: int = 3) -> tuple[str, str | None]:
    """Classify a German article with Wikimedia Lift Wing outlink topic model."""
    data = safe_post_json(
        LIFTWING_URL,
        {"page_title": title, "lang": LANG},
        retries=retries,
    )
    try:
        results = data["prediction"]["results"] if data else []
    except (KeyError, TypeError):
        return "other", None
    if not results:
        return "other", None

    top = max(results, key=lambda item: item.get("score", 0))
    if top.get("score", 0) < 0.5:
        return "other", top.get("topic")

    specific = top.get("topic")
    prefix = specific.split(".")[0] if specific else ""
    return TOPIC_BROAD.get(prefix, "other"), specific


def quota_reached(
    articles: list[dict[str, Any]],
    topic: str,
    quotas: dict[str, int],
) -> bool:
    current = sum(1 for article in articles if article.get("topic") == topic)
    return current >= quotas.get(topic, 999)


def topic_summary(articles: list[dict[str, Any]], label: str = "") -> None:
    if label:
        print(label)
    counts = Counter(article.get("topic", "unknown") for article in articles)
    total = max(len(articles), 1)
    for topic in ["politics_history", "culture", "geography", "science", "other"]:
        count = counts.get(topic, 0)
        if count or topic != "other":
            print(f"  {topic:<20} {count:>4} ({100 * count // total:>2}%)")


def fetch_template_titles(template: str, *, limit: int = 2000) -> list[str]:
    """Fetch article titles embedding a German maintenance template."""
    titles: list[str] = []
    params: dict[str, Any] = {
        "action": "query",
        "list": "embeddedin",
        "eititle": template,
        "eilimit": 500,
        "einamespace": 0,
        "format": "json",
    }
    while len(titles) < limit:
        data = safe_get(API_URL, params)
        if not data:
            break
        titles.extend(page["title"] for page in data["query"].get("embeddedin", []))
        if "continue" not in data:
            break
        params["eicontinue"] = data["continue"]["eicontinue"]
        time.sleep(REQUEST_SLEEP_SECONDS)
    return list(dict.fromkeys(titles[:limit]))


def fetch_category_titles(
    categories: list[str],
    *,
    cache_path: Path = STABLE_TITLES_FILE,
    limit_per_category: int | None = None,
) -> list[str]:
    """Fetch German stable article candidates from quality categories."""
    if cache_path.exists():
        titles = load_json(cache_path, [])
        random.shuffle(titles)
        print(f"Loaded {len(titles)} stable titles from {cache_path}")
        return titles

    all_titles: list[str] = []
    for category in categories:
        print(f"Fetching category members: {category}")
        category_titles: list[str] = []
        params: dict[str, Any] = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": 500,
            "cmtype": "page",
            "cmnamespace": 0,
            "format": "json",
        }
        while True:
            data = safe_get(API_URL, params)
            if not data:
                break
            batch = [
                page["title"]
                for page in data["query"].get("categorymembers", [])
                if not page["title"].startswith("Diskussion:")
            ]
            category_titles.extend(batch)
            if limit_per_category and len(category_titles) >= limit_per_category:
                category_titles = category_titles[:limit_per_category]
                break
            if "continue" not in data:
                break
            params["cmcontinue"] = data["continue"]["cmcontinue"]
            time.sleep(REQUEST_SLEEP_SECONDS)
        print(f"  {len(category_titles)} titles")
        all_titles.extend(category_titles)

    titles = list(dict.fromkeys(all_titles))
    random.shuffle(titles)
    save_json(titles, cache_path)
    return titles


def fetch_article(title: str, label: int) -> dict[str, Any] | None:
    """Fetch and validate one German Wikipedia article."""
    data_main = safe_get(
        API_URL,
        {
            "action": "query",
            "titles": title,
            "prop": "revisions|categories",
            "rvprop": "content|timestamp",
            "rvslots": "main",
            "rvlimit": 1,
            "cllimit": 50,
            "format": "json",
        },
    )
    if not data_main:
        return None

    page = get_page_value(data_main)
    if not page or "missing" in page or "revisions" not in page:
        return None

    raw = get_revision_content(page["revisions"][0])
    word_count = len(raw.split())
    if word_count < MIN_WORDS:
        return None
    if label == 1 and word_count > MAX_WORDS_STABLE:
        return None
    if label == 0 and not has_current_dispute_template(raw):
        return None

    clean = clean_wikitext(raw)
    citation_count = len(re.findall(r"<ref", raw, re.IGNORECASE))
    section_count = len(re.findall(r"^==+[^=]", raw, re.MULTILINE))

    topic, topic_specific = get_topic_liftwing(title)
    time.sleep(REQUEST_SLEEP_SECONDS)

    data_revisions = safe_get(
        API_URL,
        {
            "action": "query",
            "titles": title,
            "prop": "revisions",
            "rvprop": "user|timestamp",
            "rvlimit": 500,
            "format": "json",
        },
    )
    revisions = []
    if data_revisions:
        revision_page = get_page_value(data_revisions)
        if revision_page:
            revisions = revision_page.get("revisions", [])

    unique_editors = len(set(revision.get("user", "") for revision in revisions))
    total_edits = len(revisions)
    if unique_editors < MIN_EDITORS:
        return None

    age_days = 0
    if revisions:
        first_timestamp = revisions[-1].get("timestamp")
        if first_timestamp:
            first_dt = dt.datetime.fromisoformat(
                first_timestamp.replace("Z", "+00:00")
            )
            age_days = (dt.datetime.now(dt.timezone.utc) - first_dt).days
            if age_days < MIN_AGE_DAYS:
                return None

    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=365)
    recent_edits = sum(
        1
        for revision in revisions
        if dt.datetime.fromisoformat(
            revision["timestamp"].replace("Z", "+00:00")
        )
        >= cutoff
    )
    recency_ratio = round(recent_edits / total_edits, 3) if total_edits else 0
    time.sleep(REQUEST_SLEEP_SECONDS)

    talk_words, talk_editors, revert_count = fetch_talk_page_metrics(title)

    return {
        "title": title,
        "label": label,
        "label_name": "contested" if label == 0 else "stable",
        "raw_text": raw,
        "clean_text": clean,
        "word_count": word_count,
        "topic": topic,
        "topic_specific": topic_specific,
        "citation_count": citation_count,
        "section_count": section_count,
        "unique_editors": unique_editors,
        "total_edits": total_edits,
        "age_days": age_days,
        "recency_ratio": recency_ratio,
        "revert_count": revert_count,
        "talk_words": talk_words,
        "talk_editors": talk_editors,
        "lang": LANG,
        "collected_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def fetch_talk_page_metrics(title: str) -> tuple[int, int, int]:
    """Fetch German talk-page word/editor/revert metrics."""
    data_talk = safe_get(
        API_URL,
        {
            "action": "query",
            "titles": get_talk_title(title),
            "prop": "revisions",
            "rvprop": "comment|content|user",
            "rvslots": "main",
            "rvlimit": 500,
            "format": "json",
        },
    )
    if not data_talk:
        return 0, 0, 0

    page = get_page_value(data_talk)
    if not page or "missing" in page:
        return 0, 0, 0

    revisions = page.get("revisions", [])
    revert_terms = [
        "revert",
        "reverted",
        "undid",
        "undo",
        "restored",
        "zurückgesetzt",
        "rückgängig",
        "rv",
        "revertiert",
        "wiederhergestellt",
    ]
    revert_count = sum(
        1
        for revision in revisions
        if any(term in revision.get("comment", "").lower() for term in revert_terms)
    )
    talk_editors = len(set(revision.get("user", "") for revision in revisions))
    talk_words = 0
    if revisions:
        talk_words = len(get_revision_content(revisions[0]).split())
    return talk_words, talk_editors, revert_count


def collect_contested_articles(
    *,
    target: int = TARGET_CONTESTED,
    output_path: Path = CONTESTED_FILE,
) -> list[dict[str, Any]]:
    """Collect German articles currently marked with dispute templates."""
    articles: list[dict[str, Any]] = load_json(output_path, [])
    seen = {article["title"] for article in articles}
    print(f"Contested German articles: have {len(articles)}, target {target}")
    topic_summary(articles, "Current contested topic distribution:")

    if output_path.exists():
        print(f"Skipping contested collection because {output_path} already exists")
        return articles

    if len(articles) >= target:
        return articles

    candidates: list[str] = []
    for template in GERMAN_DISPUTE_TEMPLATES:
        titles = fetch_template_titles(template)
        print(f"{template}: {len(titles)} candidates")
        candidates.extend(titles)

    candidates = [title for title in dict.fromkeys(candidates) if title not in seen]
    random.shuffle(candidates)

    skipped_quality = 0
    skipped_topic = 0
    skipped_quota = 0

    for title in candidates:
        if len(articles) >= target:
            break
        time.sleep(random.uniform(ARTICLE_SLEEP_MIN_SECONDS, ARTICLE_SLEEP_MAX_SECONDS))

        article = fetch_article(title, label=0)
        if not article:
            skipped_quality += 1
            continue
        if article["topic"] == "other":
            skipped_topic += 1
            continue
        if quota_reached(articles, article["topic"], TOPIC_QUOTAS_CONTESTED):
            skipped_quota += 1
            continue

        articles.append(article)
        seen.add(title)
        print(
            f"[contested {len(articles)}/{target}] {title[:60]} "
            f"| {article['word_count']}w | {article['topic']}"
        )
        if len(articles) % 10 == 0:
            save_json(articles, output_path)

    save_json(articles, output_path)
    print(
        "Contested done: "
        f"{len(articles)} saved, {skipped_quality} quality skips, "
        f"{skipped_topic} topic skips, {skipped_quota} quota skips"
    )
    topic_summary(articles, "Final contested topic distribution:")
    return articles


def collect_stable_articles(
    *,
    target: int = TARGET_STABLE,
    output_path: Path = STABLE_FILE,
) -> list[dict[str, Any]]:
    """Collect German stable articles from quality-rated article categories."""
    articles: list[dict[str, Any]] = load_json(output_path, [])
    seen = {article["title"] for article in articles}
    print(f"Stable German articles: have {len(articles)}, target {target}")
    topic_summary(articles, "Current stable topic distribution:")

    if len(articles) >= target:
        return articles

    candidates = fetch_category_titles(GERMAN_STABLE_CATEGORIES)
    candidates = [title for title in candidates if title not in seen]

    skipped_dispute = 0
    skipped_quality = 0
    skipped_topic = 0
    skipped_quota = 0

    for title in candidates:
        if len(articles) >= target:
            break
        time.sleep(random.uniform(ARTICLE_SLEEP_MIN_SECONDS, ARTICLE_SLEEP_MAX_SECONDS))

        if has_dispute_history(title):
            skipped_dispute += 1
            continue

        article = fetch_article(title, label=1)
        if not article:
            skipped_quality += 1
            continue
        if article["topic"] == "other":
            skipped_topic += 1
            continue
        if quota_reached(articles, article["topic"], TOPIC_QUOTAS_STABLE):
            skipped_quota += 1
            continue

        articles.append(article)
        seen.add(title)
        print(
            f"[stable {len(articles)}/{target}] {title[:60]} "
            f"| {article['word_count']}w | {article['topic']}"
        )
        if len(articles) % 10 == 0:
            save_json(articles, output_path)

    save_json(articles, output_path)
    print(
        "Stable done: "
        f"{len(articles)} saved, {skipped_dispute} dispute skips, "
        f"{skipped_quality} quality skips, {skipped_topic} topic skips, "
        f"{skipped_quota} quota skips"
    )
    topic_summary(articles, "Final stable topic distribution:")
    return articles


def main() -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    contested = collect_contested_articles()
    stable = collect_stable_articles()
    print("German collection complete")
    print(f"  contested: {len(contested)} -> {CONTESTED_FILE}")
    print(f"  stable   : {len(stable)} -> {STABLE_FILE}")


if __name__ == "__main__":
    main()
