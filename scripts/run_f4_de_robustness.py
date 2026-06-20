#!/usr/bin/env python3
"""Run German F4 attribution-bias robustness diagnostics.

The original notebook F4 is preserved. This script recomputes attribution
counts from the existing German matched-pair dataset and writes separate
robustness outputs under ``outputs/f4_robustness_de``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


CURRENT_NEUTRAL_REPORT_DE = {
    "sagen",
    "berichten",
    "erklären",
    "beschreiben",
    "angeben",
    "mitteilen",
    "erwähnen",
    "bemerken",
    "schreiben",
    "bericht",
    "erklärung",
    "beschreibung",
    "mitteilung",
    "angabe",
    "erwähnung",
    "bemerkung",
}

CURRENT_BIASED_REPORT_DE = {
    "behaupten",
    "beschuldigen",
    "vorwerfen",
    "insistieren",
    "anklagen",
    "behauptung",
    "beschuldigung",
    "vorwurf",
    "anklage",
}

# The expansion remains within attribution/framing predicates: each term reports
# another actor's stance while adding argumentative, adversarial, concessive, or
# justificatory framing beyond neutral "said/reported" predicates.
EXPANDED_BIASED_REPORT_DE = CURRENT_BIASED_REPORT_DE | {
    "bestreiten",
    "bestreitung",
    "kritisieren",
    "kritik",
    "warnen",
    "warnung",
    "fordern",
    "forderung",
    "zurückweisen",
    "zurückweisung",
    "rechtfertigen",
    "rechtfertigung",
    "einräumen",
    "einräumung",
    "unterstellen",
    "unterstellung",
    "vorhalten",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="final_pairs_de.json")
    parser.add_argument("--outdir", default="outputs/f4_robustness_de")
    return parser.parse_args()


def load_articles(path: Path) -> tuple[list[dict], list[dict]]:
    pairs = json.loads(path.read_text(encoding="utf-8"))
    articles: list[dict] = []
    seen: set[str] = set()
    for pair in pairs:
        for role in ["contested", "stable"]:
            article = pair[role]
            title = article["title"]
            if title in seen:
                continue
            seen.add(title)
            articles.append(
                {
                    "title": title,
                    "label": 0 if role == "contested" else 1,
                    "label_name": role,
                    "topic": article.get("topic", "other"),
                    "clean_text": article.get("clean_text", ""),
                    "word_count": article.get("word_count", 0),
                    "age_days": article.get("age_days", 0),
                }
            )
    return articles, pairs


def count_attribution(
    nlp,
    text: str,
    *,
    biased_lexicon: set[str],
    neutral_lexicon: set[str],
) -> tuple[int, int]:
    if not text or not text.strip():
        return 0, 0
    doc = nlp(text)
    biased = 0
    neutral = 0
    for token in doc:
        if token.pos_ not in {"VERB", "NOUN"}:
            continue
        lemma = token.lemma_.lower()
        if lemma in biased_lexicon:
            biased += 1
        elif lemma in neutral_lexicon:
            neutral += 1
    return biased, neutral


def ratio_with_threshold(biased: int, neutral: int, threshold: int | None) -> float:
    total = biased + neutral
    if threshold is None:
        return biased / total if total > 0 else np.nan
    if total < threshold:
        return np.nan
    return biased / total


def build_feature_rows(articles: list[dict]) -> pd.DataFrame:
    nlp = spacy.load("de_core_news_sm", disable=["ner"])
    rows = []
    for index, article in enumerate(articles, start=1):
        current_biased, current_neutral = count_attribution(
            nlp,
            article["clean_text"],
            biased_lexicon=CURRENT_BIASED_REPORT_DE,
            neutral_lexicon=CURRENT_NEUTRAL_REPORT_DE,
        )
        expanded_biased, expanded_neutral = count_attribution(
            nlp,
            article["clean_text"],
            biased_lexicon=EXPANDED_BIASED_REPORT_DE,
            neutral_lexicon=CURRENT_NEUTRAL_REPORT_DE,
        )
        current_total = current_biased + current_neutral
        expanded_total = expanded_biased + expanded_neutral
        word_count = max(int(article.get("word_count") or 0), 1)
        rows.append(
            {
                **{key: article[key] for key in ["title", "label", "label_name", "topic", "word_count", "age_days"]},
                "f4_current_biased_count": current_biased,
                "f4_current_neutral_count": current_neutral,
                "f4_current_total_count": current_total,
                "f4_current_ratio_min3": ratio_with_threshold(current_biased, current_neutral, 3),
                "f4_current_ratio_min2": ratio_with_threshold(current_biased, current_neutral, 2),
                "f4_current_ratio_min1": ratio_with_threshold(current_biased, current_neutral, 1),
                "f4_current_ratio_no_threshold": ratio_with_threshold(current_biased, current_neutral, None),
                "f4_current_smoothed_ratio": (current_biased + 0.5) / (current_total + 1),
                "f4_current_biased_density": current_biased / word_count * 1000,
                "f4_current_neutral_density": current_neutral / word_count * 1000,
                "f4_expanded_biased_count": expanded_biased,
                "f4_expanded_neutral_count": expanded_neutral,
                "f4_expanded_total_count": expanded_total,
                "f4_expanded_ratio_min3": ratio_with_threshold(expanded_biased, expanded_neutral, 3),
                "f4_expanded_smoothed_ratio": (expanded_biased + 0.5) / (expanded_total + 1),
                "f4_expanded_biased_density": expanded_biased / word_count * 1000,
                "f4_expanded_neutral_density": expanded_neutral / word_count * 1000,
            }
        )
        if index % 25 == 0:
            print(f"Processed {index}/{len(articles)} articles", flush=True)
    return pd.DataFrame(rows)


def pair_differences(df: pd.DataFrame, pairs: list[dict], column: str) -> pd.DataFrame:
    by_title = df.set_index("title")
    rows = []
    for pair in pairs:
        contested_title = pair["contested"]["title"]
        stable_title = pair["stable"]["title"]
        if contested_title not in by_title.index or stable_title not in by_title.index:
            continue
        contested_value = by_title.loc[contested_title, column]
        stable_value = by_title.loc[stable_title, column]
        if pd.isna(contested_value) or pd.isna(stable_value):
            continue
        rows.append(
            {
                "feature": column,
                "contested_title": contested_title,
                "stable_title": stable_title,
                "contested_value": contested_value,
                "stable_value": stable_value,
                "diff": contested_value - stable_value,
            }
        )
    return pd.DataFrame(rows)


def wilcoxon_metrics(diffs: pd.DataFrame) -> dict:
    if len(diffs) < 5:
        return {
            "n_pairs": len(diffs),
            "mean_pair_diff": np.nan,
            "pct_contested_higher": np.nan,
            "wilcoxon_p": np.nan,
            "effect_r": np.nan,
        }
    values = diffs["diff"].dropna().values
    stat, p_value = wilcoxon(values)
    n = len(values)
    mean_w = n * (n + 1) / 4
    std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z_value = (stat - mean_w) / std_w
    return {
        "n_pairs": n,
        "mean_pair_diff": float(values.mean()),
        "pct_contested_higher": float((values > 0).mean() * 100),
        "wilcoxon_p": float(p_value),
        "effect_r": float(abs(z_value) / np.sqrt(n)),
    }


def logistic_metrics(df: pd.DataFrame, column: str) -> dict:
    topic_dummies = pd.get_dummies(df["topic"], prefix="topic", drop_first=True)
    model_df = pd.concat(
        [df[["label", column]].reset_index(drop=True), topic_dummies.reset_index(drop=True)],
        axis=1,
    ).dropna()
    y = model_df["label"].values
    topic_cols = [col for col in model_df.columns if col.startswith("topic_")]
    if len(model_df) < 10 or len(set(y)) < 2:
        return {
            "logistic_coefficient": np.nan,
            "macro_f1_with_feature_topic": np.nan,
            "macro_f1_topic_baseline": np.nan,
            "macro_f1_contribution": np.nan,
            "n_model_rows": len(model_df),
        }
    min_class_count = int(pd.Series(y).value_counts().min())
    cv = StratifiedKFold(n_splits=min(5, min_class_count), shuffle=True, random_state=42)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)

    def macro_f1(columns: list[str]) -> float:
        if not columns:
            return float(max(y.mean(), 1 - y.mean()))
        x_values = StandardScaler().fit_transform(model_df[columns].values)
        return float(cross_val_score(lr, x_values, y, cv=cv, scoring="f1_macro").mean())

    baseline_f1 = macro_f1(topic_cols)
    with_feature_f1 = macro_f1([column] + topic_cols)
    x_coef = StandardScaler().fit_transform(model_df[[column] + topic_cols].values)
    lr.fit(x_coef, y)
    return {
        "logistic_coefficient": float(lr.coef_[0][0]),
        "macro_f1_with_feature_topic": with_feature_f1,
        "macro_f1_topic_baseline": baseline_f1,
        "macro_f1_contribution": with_feature_f1 - baseline_f1,
        "n_model_rows": len(model_df),
    }


def summarize_variant(df: pd.DataFrame, pairs: list[dict], column: str, label: str) -> tuple[dict, pd.DataFrame]:
    contested_values = df[df["label"] == 0][column]
    stable_values = df[df["label"] == 1][column]
    diffs = pair_differences(df, pairs, column)
    contested_mean = contested_values.mean()
    stable_mean = stable_values.mean()
    return (
        {
            "Variant": label,
            "Column": column,
            "Contested mean": float(contested_mean),
            "Stable mean": float(stable_mean),
            "Percent difference": float(
                (contested_mean - stable_mean) / (abs(stable_mean) + 1e-9) * 100
            ),
            "NaN rate": float(df[column].isna().mean()),
            **wilcoxon_metrics(diffs),
            **logistic_metrics(df, column),
        },
        diffs,
    )


def count_distribution(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "f4_current_biased_count",
        "f4_current_neutral_count",
        "f4_current_total_count",
        "f4_expanded_biased_count",
        "f4_expanded_neutral_count",
        "f4_expanded_total_count",
    ]
    rows = []
    for label_name, group in df.groupby("label_name"):
        for column in columns:
            values = group[column]
            rows.append(
                {
                    "label_name": label_name,
                    "count_column": column,
                    "mean": values.mean(),
                    "median": values.median(),
                    "min": values.min(),
                    "p25": values.quantile(0.25),
                    "p75": values.quantile(0.75),
                    "max": values.max(),
                    "zero_rate": float((values == 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    articles, pairs = load_articles(Path(args.pairs))
    print(f"Articles: {len(articles)}", flush=True)

    features = build_feature_rows(articles)
    feature_path = outdir / "f4_robustness_article_features_de.csv"
    summary_path = outdir / "f4_robustness_summary_de.csv"
    pair_path = outdir / "f4_robustness_pair_diffs_de.csv"
    dist_path = outdir / "f4_attribution_count_distributions_de.csv"
    lexicon_path = outdir / "f4_lexicon_audit_de.json"

    features.to_csv(feature_path, index=False)
    count_distribution(features).to_csv(dist_path, index=False)

    variants = [
        ("f4_current_ratio_min3", "Current ratio, min total >=3"),
        ("f4_current_ratio_min2", "Current ratio, min total >=2"),
        ("f4_current_ratio_min1", "Current ratio, min total >=1"),
        ("f4_current_ratio_no_threshold", "Current ratio, no threshold"),
        ("f4_current_smoothed_ratio", "Current smoothed ratio"),
        ("f4_current_biased_density", "Current biased attribution density"),
        ("f4_current_neutral_density", "Current neutral attribution density"),
        ("f4_expanded_ratio_min3", "Expanded ratio, min total >=3"),
        ("f4_expanded_smoothed_ratio", "Expanded smoothed ratio"),
        ("f4_expanded_biased_density", "Expanded biased attribution density"),
        ("f4_expanded_neutral_density", "Expanded neutral attribution density"),
    ]
    summaries = []
    diff_frames = []
    for column, label in variants:
        summary, diffs = summarize_variant(features, pairs, column, label)
        summaries.append(summary)
        diff_frames.append(diffs)
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    pd.concat(diff_frames, ignore_index=True).to_csv(pair_path, index=False)

    lexicon_audit = {
        "current_biased_attribution_lexicon": sorted(CURRENT_BIASED_REPORT_DE),
        "current_neutral_attribution_lexicon": sorted(CURRENT_NEUTRAL_REPORT_DE),
        "expanded_biased_additions": sorted(EXPANDED_BIASED_REPORT_DE - CURRENT_BIASED_REPORT_DE),
        "expanded_biased_justification": (
            "Added terms are attribution/framing predicates that report another actor's "
            "stance while encoding adversarial, critical, concessive, justificatory, "
            "or demand/warning force; neutral reporting lexicon is unchanged."
        ),
        "minimum_count_rule": (
            "Original F4 returns NaN when biased + neutral attribution count < 3."
        ),
        "current_min3_affected_articles": int(features["f4_current_ratio_min3"].isna().sum()),
        "current_min3_affected_rate": float(features["f4_current_ratio_min3"].isna().mean()),
        "current_zero_attribution_articles": int((features["f4_current_total_count"] == 0).sum()),
        "current_zero_attribution_rate": float((features["f4_current_total_count"] == 0).mean()),
    }
    lexicon_path.write_text(json.dumps(lexicon_audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    print(f"Saved: {feature_path}", flush=True)
    print(f"Saved: {summary_path}", flush=True)
    print(f"Saved: {pair_path}", flush=True)
    print(f"Saved: {dist_path}", flush=True)
    print(f"Saved: {lexicon_path}", flush=True)


if __name__ == "__main__":
    main()
