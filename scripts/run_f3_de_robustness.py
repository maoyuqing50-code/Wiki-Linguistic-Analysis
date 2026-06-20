#!/usr/bin/env python3
"""Run German F3 weasel-word robustness diagnostics.

The original notebook F3 is preserved. This script separates lexical F3 into
vague attribution, vague quantifier, and evaluative/puffery groups, adds a
Konjunktiv-I morphology component, and writes separate robustness outputs under
``outputs/f3_robustness_de``.
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


VAGUE_ATTRIBUTION = [
    "einige menschen",
    "manche menschen",
    "viele menschen",
    "experten sagen",
    "experten behaupten",
    "es heißt",
    "es wird gesagt",
    "es wird behauptet",
    "es wurde vorgeschlagen",
    "es wird oft gesagt",
    "es wird berichtet",
    "es wird argumentiert",
    "es wird angenommen",
    "forscher sagen",
    "wissenschaftler sagen",
    "historiker sagen",
    "kritiker sagen",
    "studien zeigen",
    "forschungen zeigen",
    "laut einigen",
    "einige argumentieren",
    "einige glauben",
    "es gilt als",
    "man nimmt an",
]

VAGUE_QUANTIFIERS = [
    "viele",
    "die meisten",
    "mehrere",
    "eine reihe von",
    "eine mehrheit von",
    "zahlreich",
    "zahlreiche",
    "zahlreichen",
    "zahlreicher",
    "zahlreiches",
    "verschieden",
    "verschiedene",
    "verschiedenen",
    "verschiedener",
    "bestimmte",
    "einige",
    "wenige",
]

GENERIC_QUANTIFIERS = {
    "viele",
    "mehrere",
    "verschieden",
    "verschiedene",
    "verschiedenen",
    "verschiedener",
    "einige",
}

EVALUATIVE_PUFFERY = [
    "natürlich",
    "selbstverständlich",
    "offensichtlich",
    "eindeutig",
    "zweifellos",
    "es versteht sich von selbst",
    "unnötig zu sagen",
    "es ist wichtig zu beachten",
    "es ist erwähnenswert",
    "interessanterweise",
    "bemerkenswerterweise",
    "bedeutend",
    "signifikant",
    "legendär",
    "bahnbrechend",
    "revolutionär",
    "weltklasse",
    "einzigartig",
    "innovativ",
]

WEASEL_WORDS_DE = VAGUE_ATTRIBUTION + VAGUE_QUANTIFIERS + EVALUATIVE_PUFFERY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="final_pairs_de.json")
    parser.add_argument("--outdir", default="outputs/f3_robustness_de")
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


def count_terms(text: str, terms: list[str]) -> dict[str, int]:
    text_lower = text.lower()
    return {term: count_term(text_lower, term) for term in terms}


def konjunktiv_i_count(nlp, text: str) -> int:
    if not text or not text.strip():
        return 0
    doc = nlp(text)
    count = 0
    for token in doc:
        morph = str(token.morph)
        if token.pos_ == "VERB" and "Mood=Sub" in morph and "Tense=Pres" in morph:
            count += 1
    return count


def density(count: int, word_count: int) -> float:
    return count / max(word_count, 1) * 1000


def build_features(articles: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    nlp = spacy.load("de_core_news_sm", disable=["ner"])
    article_rows = []
    term_rows = []
    quant_no_generic = [term for term in VAGUE_QUANTIFIERS if term not in GENERIC_QUANTIFIERS]

    for index, article in enumerate(articles, start=1):
        text = article["clean_text"]
        word_count = max(int(article.get("word_count") or len(text.split())), 1)

        attribution_counts = count_terms(text, VAGUE_ATTRIBUTION)
        quantifier_counts = count_terms(text, VAGUE_QUANTIFIERS)
        evaluative_counts = count_terms(text, EVALUATIVE_PUFFERY)
        ki_count = konjunktiv_i_count(nlp, text)

        for group_name, counts in [
            ("vague_attribution", attribution_counts),
            ("vague_quantifier", quantifier_counts),
            ("evaluative_puffery", evaluative_counts),
        ]:
            for term, count in counts.items():
                term_rows.append(
                    {
                        "title": article["title"],
                        "label": article["label"],
                        "label_name": article["label_name"],
                        "topic": article["topic"],
                        "word_count": word_count,
                        "group": group_name,
                        "term": term,
                        "count": count,
                        "density_per_1000": density(count, word_count),
                    }
                )

        attr_count = sum(attribution_counts.values())
        quant_count = sum(quantifier_counts.values())
        eval_count = sum(evaluative_counts.values())
        quant_no_generic_count = sum(quantifier_counts[term] for term in quant_no_generic)
        lexical_count = attr_count + quant_count + eval_count
        lexical_no_generic_quant_count = attr_count + quant_no_generic_count + eval_count

        article_rows.append(
            {
                **{
                    key: article[key]
                    for key in ["title", "label", "label_name", "topic", "word_count", "age_days"]
                },
                "f3_original_recomputed": density(lexical_count + ki_count, word_count),
                "f3_lexical_only": density(lexical_count, word_count),
                "f3_vague_attribution_only": density(attr_count, word_count),
                "f3_vague_quantifiers_only": density(quant_count, word_count),
                "f3_evaluative_puffery_only": density(eval_count, word_count),
                "f3_konjunktiv_i_only": density(ki_count, word_count),
                "f3_no_generic_quantifiers": density(
                    lexical_no_generic_quant_count + ki_count, word_count
                ),
                "f3_attribution_plus_evaluative": density(attr_count + eval_count + ki_count, word_count),
                "f3_lexical_count": lexical_count,
                "f3_vague_attribution_count": attr_count,
                "f3_vague_quantifier_count": quant_count,
                "f3_evaluative_puffery_count": eval_count,
                "f3_konjunktiv_i_count": ki_count,
            }
        )
        if index % 25 == 0:
            print(f"Processed {index}/{len(articles)} articles", flush=True)

    return pd.DataFrame(article_rows), pd.DataFrame(term_rows)


def group_contribution(term_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        term_df.groupby(["group", "label_name"])
        .agg(
            frequency=("count", "sum"),
            density_sum=("density_per_1000", "sum"),
            article_hits=("count", lambda values: int((values > 0).sum())),
        )
        .reset_index()
    )
    wide = grouped.pivot(index="group", columns="label_name", values=["frequency", "density_sum", "article_hits"]).fillna(0)
    wide.columns = [f"{metric}_{label}" for metric, label in wide.columns]
    wide = wide.reset_index()
    wide["total_density_sum"] = wide.get("density_sum_contested", 0) + wide.get("density_sum_stable", 0)
    wide["contested_minus_stable_density_sum"] = wide.get("density_sum_contested", 0) - wide.get("density_sum_stable", 0)
    return wide.sort_values("total_density_sum", ascending=False)


def term_contribution(term_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        term_df.groupby(["group", "term", "label_name"])
        .agg(
            frequency=("count", "sum"),
            density_sum=("density_per_1000", "sum"),
            article_hits=("count", lambda values: int((values > 0).sum())),
        )
        .reset_index()
    )
    wide = grouped.pivot(index=["group", "term"], columns="label_name", values=["frequency", "density_sum", "article_hits"]).fillna(0)
    wide.columns = [f"{metric}_{label}" for metric, label in wide.columns]
    wide = wide.reset_index()
    wide["total_density_sum"] = wide.get("density_sum_contested", 0) + wide.get("density_sum_stable", 0)
    wide["contested_minus_stable_density_sum"] = wide.get("density_sum_contested", 0) - wide.get("density_sum_stable", 0)
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

    article_features, term_counts = build_features(articles)
    group_audit = group_contribution(term_counts)
    term_audit = term_contribution(term_counts)

    variants = [
        ("f3_original_recomputed", "Original F3 recomputed"),
        ("f3_lexical_only", "Lexical-only"),
        ("f3_vague_attribution_only", "Vague attribution only"),
        ("f3_vague_quantifiers_only", "Vague quantifiers only"),
        ("f3_evaluative_puffery_only", "Evaluative/puffery only"),
        ("f3_konjunktiv_i_only", "Konjunktiv-I only"),
        ("f3_no_generic_quantifiers", "Remove generic quantifiers"),
        ("f3_attribution_plus_evaluative", "Attribution + evaluative + Konjunktiv-I"),
    ]
    summaries = []
    diff_frames = []
    for column, label in variants:
        summary, diffs = summarize_variant(article_features, pairs, column, label)
        summaries.append(summary)
        diff_frames.append(diffs)

    summary_df = pd.DataFrame(summaries).sort_values(
        ["macro_f1_contribution", "wilcoxon_p"], ascending=[False, True]
    )
    article_features.to_csv(outdir / "f3_robustness_article_features_de.csv", index=False)
    term_counts.to_csv(outdir / "f3_term_article_counts_de.csv", index=False)
    group_audit.to_csv(outdir / "f3_group_contribution_audit_de.csv", index=False)
    term_audit.to_csv(outdir / "f3_term_contribution_audit_de.csv", index=False)
    summary_df.to_csv(outdir / "f3_robustness_summary_de.csv", index=False)
    pd.concat(diff_frames, ignore_index=True).to_csv(outdir / "f3_robustness_pair_diffs_de.csv", index=False)

    audit = {
        "vague_attribution_terms": VAGUE_ATTRIBUTION,
        "vague_quantifier_terms": VAGUE_QUANTIFIERS,
        "generic_quantifiers_removed": sorted(GENERIC_QUANTIFIERS),
        "evaluative_puffery_terms": EVALUATIVE_PUFFERY,
        "konjunktiv_i_rule": "token.pos_ == 'VERB' and Mood=Sub and Tense=Pres",
    }
    (outdir / "f3_lexicon_audit_de.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(summary_df.to_string(index=False), flush=True)
    print(f"Saved outputs to: {outdir}", flush=True)


if __name__ == "__main__":
    main()
