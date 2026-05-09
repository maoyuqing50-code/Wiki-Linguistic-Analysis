"""Validate the German matched Wikipedia dataset."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

FEATURES_FILE = PROCESSED_DATA_DIR / "final_features_de.csv"
PAIRS_FILE = PROCESSED_DATA_DIR / "final_pairs_de.csv"
CONTESTED_RAW_FILE = RAW_DATA_DIR / "final_contested_de.json"
STABLE_RAW_FILE = RAW_DATA_DIR / "final_stable_de.json"
REPORT_FILE = PROCESSED_DATA_DIR / "de_validation_report.txt"

RANDOM_SEED = 42
MANUAL_SAMPLE_SIZE = 10
NEAR_EMPTY_TEXT_WORDS = 100
EXTREMELY_SHORT_WORDS = 1500
EXTREMELY_LONG_WORDS = 30000

DISPUTE_TEMPLATE_PATTERN = re.compile(
    r"\{\{\s*(?:Neutralität|Neutralitaet|Überarbeiten|Ueberarbeiten)\b",
    re.IGNORECASE,
)

GERMAN_STOPWORDS = {
    "der",
    "die",
    "das",
    "den",
    "dem",
    "des",
    "ein",
    "eine",
    "einer",
    "einem",
    "einen",
    "und",
    "oder",
    "aber",
    "auch",
    "nicht",
    "mit",
    "von",
    "vom",
    "zu",
    "zur",
    "zum",
    "im",
    "in",
    "am",
    "an",
    "auf",
    "für",
    "als",
    "bei",
    "nach",
    "aus",
    "durch",
    "über",
    "unter",
    "wird",
    "wurde",
    "wurden",
    "ist",
    "sind",
    "war",
    "waren",
    "hat",
    "haben",
    "sich",
    "seine",
    "seiner",
    "ihre",
    "ihrer",
}

ENGLISH_STOPWORDS = {
    "the",
    "and",
    "or",
    "but",
    "not",
    "with",
    "from",
    "to",
    "of",
    "in",
    "on",
    "for",
    "as",
    "by",
    "after",
    "over",
    "under",
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "his",
    "her",
    "their",
    "which",
    "this",
    "that",
}


class Report:
    """Collect report lines and print important findings as they are added."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, text: str = "") -> None:
        self.lines.append(text)

    def section(self, title: str) -> None:
        self.add("")
        self.add("=" * 80)
        self.add(title)
        self.add("=" * 80)

    def suspicious(self, title: str, rows: pd.DataFrame | list[dict[str, Any]]) -> None:
        self.section(f"SUSPICIOUS: {title}")
        if isinstance(rows, pd.DataFrame):
            if rows.empty:
                self.add("None")
                return
            self.add(rows.to_string(index=False, max_colwidth=80))
            print(f"\nSUSPICIOUS: {title}")
            print(rows.to_string(index=False, max_colwidth=80))
            return
        if not rows:
            self.add("None")
            return
        for row in rows:
            self.add(str(row))
        print(f"\nSUSPICIOUS: {title}")
        for row in rows[:25]:
            print(row)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines).strip() + "\n", encoding="utf-8")


def load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"Missing features file: {FEATURES_FILE}")
    if not PAIRS_FILE.exists():
        raise FileNotFoundError(f"Missing pairs file: {PAIRS_FILE}")

    features = pd.read_csv(FEATURES_FILE)
    pairs = pd.read_csv(PAIRS_FILE)

    raw_articles = load_json_list(CONTESTED_RAW_FILE) + load_json_list(STABLE_RAW_FILE)
    raw_by_title = {
        str(article.get("title")): article
        for article in raw_articles
        if article.get("title") is not None
    }
    return features, pairs, raw_by_title


def article_text(article: dict[str, Any] | None) -> str:
    if not article:
        return ""
    text = article.get("clean_text") or article.get("text") or article.get("raw_text") or ""
    if isinstance(text, dict):
        return str(text.get("clean") or text.get("raw") or "")
    return str(text)


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÄÖÜäöüß]+", text.lower())


def dominant_language_scores(text: str) -> dict[str, Any]:
    words = tokenize_words(text[:30000])
    if not words:
        return {
            "word_count": 0,
            "de_hits": 0,
            "en_hits": 0,
            "umlaut_hits": 0,
            "dominant": "unknown",
        }
    counts = Counter(words)
    de_hits = sum(counts[word] for word in GERMAN_STOPWORDS)
    en_hits = sum(counts[word] for word in ENGLISH_STOPWORDS)
    umlaut_hits = sum(1 for word in words if any(char in word for char in "äöüß"))

    if de_hits >= max(5, en_hits):
        dominant = "de"
    elif en_hits > de_hits * 1.5 and en_hits >= 10:
        dominant = "non_de"
    else:
        dominant = "uncertain"

    return {
        "word_count": len(words),
        "de_hits": de_hits,
        "en_hits": en_hits,
        "umlaut_hits": umlaut_hits,
        "dominant": dominant,
    }


def check_duplicates(features: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "duplicate_feature_titles": features[
            features["title"].duplicated(keep=False)
        ].sort_values("title"),
        "duplicate_contested_pairs": pairs[
            pairs["contested_title"].duplicated(keep=False)
        ].sort_values("contested_title"),
        "duplicate_stable_pairs": pairs[
            pairs["stable_title"].duplicated(keep=False)
        ].sort_values("stable_title"),
    }


def check_missing_values(features: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    return features.isna().sum().sort_values(ascending=False), pairs.isna().sum().sort_values(
        ascending=False
    )


def check_text_quality(
    features: pd.DataFrame,
    raw_by_title: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, row in features.iterrows():
        title = row["title"]
        text = article_text(raw_by_title.get(title))
        text_words = len(tokenize_words(text))
        rows.append(
            {
                "title": title,
                "label": row.get("label"),
                "topic": row.get("topic"),
                "feature_word_count": row.get("word_count"),
                "text_word_count": text_words,
            }
        )
    text_df = pd.DataFrame(rows)
    empty_or_near_empty = text_df[text_df["text_word_count"] < NEAR_EMPTY_TEXT_WORDS]
    short_or_long = features[
        (features["word_count"] < EXTREMELY_SHORT_WORDS)
        | (features["word_count"] > EXTREMELY_LONG_WORDS)
    ][["title", "label", "topic", "word_count"]].sort_values("word_count")
    missing_raw = text_df[(text_df["text_word_count"] == 0)]
    return empty_or_near_empty, short_or_long, missing_raw


def check_language(
    features: pd.DataFrame,
    raw_by_title: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for _, row in features.iterrows():
        title = row["title"]
        scores = dominant_language_scores(article_text(raw_by_title.get(title)))
        if scores["dominant"] != "de":
            rows.append(
                {
                    "title": title,
                    "label": row.get("label"),
                    "topic": row.get("topic"),
                    **scores,
                }
            )
    columns = [
        "title",
        "label",
        "topic",
        "word_count",
        "de_hits",
        "en_hits",
        "umlaut_hits",
        "dominant",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["dominant", "en_hits"],
        ascending=[True, False],
    )


def check_topic_mismatches(features: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    by_title = features.set_index("title")
    rows = []
    for _, pair in pairs.iterrows():
        contested_title = pair["contested_title"]
        stable_title = pair["stable_title"]
        contested = by_title.loc[contested_title] if contested_title in by_title.index else None
        stable = by_title.loc[stable_title] if stable_title in by_title.index else None
        pair_topic = pair["topic"]
        contested_topic = contested["topic"] if contested is not None else None
        stable_topic = stable["topic"] if stable is not None else None
        if (
            contested is None
            or stable is None
            or pair_topic != contested_topic
            or pair_topic != stable_topic
        ):
            rows.append(
                {
                    "contested_title": contested_title,
                    "stable_title": stable_title,
                    "pair_topic": pair_topic,
                    "contested_topic": contested_topic,
                    "stable_topic": stable_topic,
                }
            )
    return pd.DataFrame(rows)


def unmatched_topic_diagnostics(features: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    contested = features[features["label"] == 0]
    stable = features[features["label"] == 1]
    matched_contested = set(pairs["contested_title"])
    matched_stable = set(pairs["stable_title"])
    rows = []
    topics = sorted(set(features["topic"].fillna("unknown")))
    for topic in topics:
        contested_topic = contested[contested["topic"] == topic]
        stable_topic = stable[stable["topic"] == topic]
        rows.append(
            {
                "topic": topic,
                "contested_total": len(contested_topic),
                "stable_total": len(stable_topic),
                "matched_contested": int(contested_topic["title"].isin(matched_contested).sum()),
                "unused_stable": int((~stable_topic["title"].isin(matched_stable)).sum()),
                "unmatched_contested": int(
                    (~contested_topic["title"].isin(matched_contested)).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def check_stable_dispute_templates(
    features: pd.DataFrame,
    raw_by_title: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    stable = features[features["label"] == 1]
    for _, row in stable.iterrows():
        title = row["title"]
        raw_text = str(raw_by_title.get(title, {}).get("raw_text", ""))
        match = DISPUTE_TEMPLATE_PATTERN.search(raw_text)
        if match:
            rows.append(
                {
                    "title": title,
                    "topic": row.get("topic"),
                    "template_match": match.group(0),
                }
            )
    return pd.DataFrame(rows)


def add_descriptive_statistics(report: Report, features: pd.DataFrame, pairs: pd.DataFrame) -> None:
    report.section("Descriptive Statistics")
    report.add("Feature matrix shape: " + str(features.shape))
    report.add("Pairs matrix shape  : " + str(pairs.shape))
    report.add("")
    report.add("Label counts:")
    report.add(features["label"].value_counts(dropna=False).to_string())
    report.add("")
    report.add("Topic counts by label:")
    report.add(pd.crosstab(features["topic"], features["label"]).to_string())
    report.add("")
    numeric_features = [
        column
        for column in ["word_count", "age_days", "hedging_density", "weasel_density", "def_ratio", "mtld"]
        if column in features.columns
    ]
    report.add("Feature numeric summary:")
    report.add(features[numeric_features].describe().to_string())
    report.add("")
    numeric_pairs = [
        column
        for column in ["word_count_diff", "age_days_diff", "mtld_diff"]
        if column in pairs.columns
    ]
    report.add("Pair-difference numeric summary:")
    report.add(pairs[numeric_pairs].describe().to_string())


def add_random_sample(report: Report, pairs: pd.DataFrame) -> None:
    report.section("Random Matched Pair Sample")
    sample_size = min(MANUAL_SAMPLE_SIZE, len(pairs))
    if sample_size == 0:
        report.add("No matched pairs available.")
        return
    sample = pairs.sample(n=sample_size, random_state=RANDOM_SEED)
    columns = [
        "contested_title",
        "stable_title",
        "topic",
        "contested_subtopic",
        "stable_subtopic",
        "exact_subtopic_match",
    ]
    sample = sample[columns]
    report.add(sample.to_string(index=False, max_colwidth=80))
    print("\n10 random matched pairs for manual inspection")
    print(sample.to_string(index=False, max_colwidth=80))


def add_balance_checks(report: Report, features: pd.DataFrame, pairs: pd.DataFrame) -> None:
    report.section("Balance Checks")
    label_counts = features["label"].value_counts().to_dict()
    contested_count = int(label_counts.get(0, 0))
    stable_count = int(label_counts.get(1, 0))
    ratio = contested_count / stable_count if stable_count else float("inf")
    report.add(f"Contested articles: {contested_count}")
    report.add(f"Stable articles   : {stable_count}")
    report.add(f"Label ratio C/S   : {ratio:.3f}")
    report.add("")
    report.add("Topic balance by label:")
    report.add(pd.crosstab(features["topic"], features["label"], margins=True).to_string())
    report.add("")
    report.add("Matched pair topic distribution:")
    report.add(pairs["topic"].value_counts().sort_index().to_string())


def main() -> None:
    random.seed(RANDOM_SEED)
    features, pairs, raw_by_title = load_inputs()

    report = Report()
    report.section("German Dataset Validation Report")
    report.add(f"Features file: {FEATURES_FILE}")
    report.add(f"Pairs file   : {PAIRS_FILE}")
    report.add(f"Raw articles : {len(raw_by_title)}")

    add_descriptive_statistics(report, features, pairs)
    add_balance_checks(report, features, pairs)
    add_random_sample(report, pairs)

    duplicate_results = check_duplicates(features, pairs)
    for title, rows in duplicate_results.items():
        report.suspicious(title.replace("_", " "), rows)

    feature_missing, pair_missing = check_missing_values(features, pairs)
    report.section("Missing Values")
    report.add("Feature missing values:")
    report.add(feature_missing.to_string())
    report.add("")
    report.add("Pair missing values:")
    report.add(pair_missing.to_string())
    suspicious_feature_missing = feature_missing[feature_missing > 0]
    suspicious_pair_missing = pair_missing[pair_missing > 0]
    if not suspicious_feature_missing.empty:
        print("\nSUSPICIOUS: feature columns with missing values")
        print(suspicious_feature_missing.to_string())
    if not suspicious_pair_missing.empty:
        print("\nSUSPICIOUS: pair columns with missing values")
        print(suspicious_pair_missing.to_string())

    empty_text, short_or_long, missing_raw = check_text_quality(features, raw_by_title)
    report.suspicious("empty or near-empty texts", empty_text)
    report.suspicious("extremely short or long articles", short_or_long)
    report.suspicious("articles missing raw text lookup", missing_raw)

    non_german = check_language(features, raw_by_title)
    report.suspicious("articles with non-German or uncertain dominant text", non_german)

    topic_mismatches = check_topic_mismatches(features, pairs)
    report.suspicious("suspicious topic mismatches", topic_mismatches)

    unmatched = unmatched_topic_diagnostics(features, pairs)
    report.section("Unmatched Topic Diagnostics")
    report.add(unmatched.to_string(index=False))
    print("\nUnmatched-topic diagnostics")
    print(unmatched.to_string(index=False))

    stable_disputes = check_stable_dispute_templates(features, raw_by_title)
    report.suspicious("stable articles containing dispute templates", stable_disputes)

    report.write(REPORT_FILE)
    print(f"\nValidation report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
