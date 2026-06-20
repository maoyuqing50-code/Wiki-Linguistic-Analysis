#!/usr/bin/env python3
"""Audit and test German F1 epistemic hedge variants.

The original notebook F1 is preserved. This script recomputes lexical hedge
term counts and Konjunktiv-II modal counts from the existing German matched
dataset, then writes separate robustness outputs under
``outputs/f1_robustness_de``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


HEDGES_DE = [
    "anscheinend",
    "möglicherweise",
    "mutmaßlich",
    "scheinbar",
    "schwerlich",
    "vermutlich",
    "vielleicht",
    "wahrscheinlich",
    "wohl",
    "womöglich",
    "angeblich",
    "vorgeblich",
    "offenbar",
    "scheinen",
    "erscheinen",
    "vermuten",
    "annehmen",
    "glauben",
    "zweifeln",
    "bezweifeln",
    "möglich",
    "unwahrscheinlich",
    "fraglich",
    "zweifelhaft",
    "unsicher",
    "unklar",
    "meiner Meinung nach",
    "meines Erachtens",
    "nach meiner Ansicht",
    "nach meiner Einschätzung",
    "soweit ich weiß",
]

KONJUNKTIV_II_MODALS = {"können", "müssen", "dürfen", "mögen"}

GENERIC_VERBS = {"scheinen", "erscheinen", "glauben", "annehmen"}

STRONG_UNCERTAINTY = [
    "möglicherweise",
    "mutmaßlich",
    "vermutlich",
    "vielleicht",
    "wahrscheinlich",
    "womöglich",
    "angeblich",
    "vorgeblich",
    "fraglich",
    "zweifelhaft",
    "unsicher",
    "unklar",
    "zweifeln",
    "bezweifeln",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="final_pairs_de.json")
    parser.add_argument("--outdir", default="outputs/f1_robustness_de")
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


def count_term(text_lower: str, term: str) -> int:
    term = term.lower()
    if " " in term:
        return text_lower.count(term)
    return len(re.findall(r"\b" + re.escape(term) + r"\b", text_lower))


def lexical_counts(text: str, terms: list[str]) -> dict[str, int]:
    text_lower = text.lower()
    return {term: count_term(text_lower, term) for term in terms}


def konjunktiv_ii_count(nlp, text: str) -> int:
    if not text or not text.strip():
        return 0
    doc = nlp(text)
    count = 0
    for token in doc:
        morph = str(token.morph)
        if (
            token.lemma_.lower() in KONJUNKTIV_II_MODALS
            and "Mood=Sub" in morph
            and "Tense=Past" in morph
        ):
            count += 1
    return count


def density(count: int, word_count: int) -> float:
    return count / max(word_count, 1) * 1000


def build_article_features(articles: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    nlp = spacy.load("de_core_news_sm", disable=["ner"])
    article_rows = []
    term_rows = []

    no_generic_terms = [term for term in HEDGES_DE if term not in GENERIC_VERBS]

    for index, article in enumerate(articles, start=1):
        text = article["clean_text"]
        word_count = max(int(article.get("word_count") or len(text.split())), 1)
        counts = lexical_counts(text, HEDGES_DE)
        kii_count = konjunktiv_ii_count(nlp, text)

        for term, count in counts.items():
            term_rows.append(
                {
                    "title": article["title"],
                    "label": article["label"],
                    "label_name": article["label_name"],
                    "topic": article["topic"],
                    "word_count": word_count,
                    "term": term,
                    "count": count,
                    "density_per_1000": density(count, word_count),
                }
            )

        lexical_count = sum(counts.values())
        no_generic_count = sum(counts[term] for term in no_generic_terms)
        strong_count = sum(counts[term] for term in STRONG_UNCERTAINTY)

        article_rows.append(
            {
                **{
                    key: article[key]
                    for key in ["title", "label", "label_name", "topic", "word_count", "age_days"]
                },
                "f1_original_recomputed": density(lexical_count + kii_count, word_count),
                "f1_lexical_only": density(lexical_count, word_count),
                "f1_konjunktiv_ii_only": density(kii_count, word_count),
                "f1_no_generic_verbs": density(no_generic_count + kii_count, word_count),
                "f1_strong_uncertainty_only": density(strong_count, word_count),
                "f1_lexical_count": lexical_count,
                "f1_konjunktiv_ii_count": kii_count,
                "f1_no_generic_count": no_generic_count,
                "f1_strong_uncertainty_count": strong_count,
            }
        )
        if index % 25 == 0:
            print(f"Processed {index}/{len(articles)} articles", flush=True)

    return pd.DataFrame(article_rows), pd.DataFrame(term_rows)


def term_contribution(term_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        term_df.groupby(["term", "label_name"])
        .agg(
            frequency=("count", "sum"),
            density_sum=("density_per_1000", "sum"),
            article_hits=("count", lambda values: int((values > 0).sum())),
        )
        .reset_index()
    )
    wide = grouped.pivot(index="term", columns="label_name", values=["frequency", "density_sum", "article_hits"]).fillna(0)
    wide.columns = [f"{metric}_{label}" for metric, label in wide.columns]
    wide = wide.reset_index()
    for col in [
        "frequency_contested",
        "frequency_stable",
        "density_sum_contested",
        "density_sum_stable",
        "article_hits_contested",
        "article_hits_stable",
    ]:
        if col not in wide:
            wide[col] = 0
    total_density = wide["density_sum_contested"].sum() + wide["density_sum_stable"].sum()
    wide["total_frequency"] = wide["frequency_contested"] + wide["frequency_stable"]
    wide["total_density_sum"] = wide["density_sum_contested"] + wide["density_sum_stable"]
    wide["contribution_to_total_f1_pct"] = wide["total_density_sum"] / max(total_density, 1e-12) * 100
    wide["contested_minus_stable_density_sum"] = wide["density_sum_contested"] - wide["density_sum_stable"]
    return wide.sort_values("total_density_sum", ascending=False)


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
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    articles, pairs = load_articles(Path(args.pairs))
    print(f"Articles: {len(articles)}", flush=True)

    article_features, term_counts = build_article_features(articles)
    term_audit = term_contribution(term_counts)

    variants = [
        ("f1_original_recomputed", "Original F1 recomputed"),
        ("f1_lexical_only", "Lexical-only"),
        ("f1_konjunktiv_ii_only", "Konjunktiv-II-only"),
        ("f1_no_generic_verbs", "Remove generic verbs"),
        ("f1_strong_uncertainty_only", "Strong uncertainty only"),
    ]
    summaries = []
    diff_frames = []
    for column, label in variants:
        summary, diffs = summarize_variant(article_features, pairs, column, label)
        summaries.append(summary)
        diff_frames.append(diffs)

    article_features.to_csv(outdir / "f1_robustness_article_features_de.csv", index=False)
    term_counts.to_csv(outdir / "f1_term_article_counts_de.csv", index=False)
    term_audit.to_csv(outdir / "f1_term_contribution_audit_de.csv", index=False)
    pd.DataFrame(summaries).to_csv(outdir / "f1_robustness_summary_de.csv", index=False)
    pd.concat(diff_frames, ignore_index=True).to_csv(outdir / "f1_robustness_pair_diffs_de.csv", index=False)

    audit = {
        "hedges_de": HEDGES_DE,
        "konjunktiv_ii_modals": sorted(KONJUNKTIV_II_MODALS),
        "generic_verbs_removed": sorted(GENERIC_VERBS),
        "strong_uncertainty_markers": STRONG_UNCERTAINTY,
        "interpretation": (
            "Term contributions are based on per-article density sums, matching the "
            "feature's per-1000-word normalization rather than raw token counts alone."
        ),
    }
    (outdir / "f1_lexicon_audit_de.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    print(f"Saved outputs to: {outdir}", flush=True)


if __name__ == "__main__":
    main()
