#!/usr/bin/env python3
"""Run German F2 affective-marker robustness diagnostics.

This script preserves the original notebook feature. It computes Hyland-only,
SentiWS-only, combined, and stricter-SentiWS variants when the original SentiWS
files are available. SentiWS is not vendored in this repository; pass the files
explicitly or place them under ``data/external/sentiws``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


HYLAND_ATTITUDE_DE = [
    "einräumen",
    "zugeben",
    "zustimmen",
    "ablehnen",
    "widersprechen",
    "erstaunt",
    "erstaunlich",
    "erstaunlicherweise",
    "angemessen",
    "angemessenerweise",
    "unangemessen",
    "unangemessenerweise",
    "richtigerweise",
    "merkwürdig",
    "merkwürdigerweise",
    "wünschenswert",
    "wünschenswerterweise",
    "enttäuscht",
    "enttäuschend",
    "enttäuschenderweise",
    "dramatisch",
    "dramatischerweise",
    "unerlässlich",
    "zu erwarten",
    "erwartungsgemäß",
    "glücklicherweise",
    "zum Glück",
    "hoffnungsvoll",
    "hoffentlich",
    "wichtig",
    "interessant",
    "interessanterweise",
    "bevorzugen",
    "vorzuziehen",
    "vorzugsweise",
    "bedauern",
    "bemerkenswert",
    "bemerkenswerterweise",
    "schockiert",
    "schockierend",
    "schockierenderweise",
    "auffallend",
    "auffällig",
    "auffallenderweise",
    "überrascht",
    "überraschend",
    "überraschenderweise",
    "unglaublich",
    "verständlich",
    "verständlicherweise",
    "unerwartet",
    "unerwarteterweise",
    "unglücklich",
    "unglücklicherweise",
    "bedauerlicherweise",
    "ungewöhnlich",
    "ungewöhnlicherweise",
    "gewöhnlich",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="final_pairs_de.json")
    parser.add_argument("--outdir", default="outputs/f2_robustness_de")
    parser.add_argument(
        "--sentiws-positive",
        default="data/external/sentiws/SentiWS_v2.0_Positive.txt",
    )
    parser.add_argument(
        "--sentiws-negative",
        default="data/external/sentiws/SentiWS_v2.0_Negative.txt",
    )
    parser.add_argument("--current-threshold", type=float, default=0.5)
    parser.add_argument("--strict-threshold", type=float, default=0.75)
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


def parse_sentiws(path: Path, threshold: float) -> tuple[list[str], dict[str, float]]:
    words: list[str] = []
    weights: dict[str, float] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            lemma = parts[0].split("|")[0].lower()
            weight = float(parts[1])
            if abs(weight) >= threshold:
                words.append(lemma)
                weights[lemma] = weight
                if len(parts) >= 3 and parts[2]:
                    for inflected in parts[2].split(","):
                        word = inflected.strip().lower()
                        words.append(word)
                        weights[word] = weight
    return list(dict.fromkeys(words)), weights


def density_count(text_lower: str, wordlist: list[str]) -> int:
    count = 0
    for word in wordlist:
        word = word.lower()
        if " " in word:
            count += text_lower.count(word)
        else:
            count += len(re.findall(r"\b" + re.escape(word) + r"\b", text_lower))
    return count


def density(text: str, wordlist: list[str]) -> float:
    if not text or not text.strip():
        return 0.0
    text_lower = text.lower()
    n_words = max(len(text_lower.split()), 1)
    return density_count(text_lower, wordlist) / n_words * 1000


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
    if len(model_df) < 10 or len(set(y)) < 2:
        return {
            "logistic_coefficient": np.nan,
            "macro_f1_with_feature_topic": np.nan,
            "macro_f1_topic_baseline": np.nan,
            "macro_f1_contribution": np.nan,
            "n_model_rows": len(model_df),
        }
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

    positive_path = Path(args.sentiws_positive)
    negative_path = Path(args.sentiws_negative)
    have_sentiws = positive_path.exists() and negative_path.exists()
    if not have_sentiws:
        print("WARNING: SentiWS files not found; running Hyland-only audit only.")
        print(f"Expected positive file: {positive_path}")
        print(f"Expected negative file: {negative_path}")

    sentiws_current: list[str] = []
    sentiws_strict: list[str] = []
    sentiws_weights_current: dict[str, float] = {}
    sentiws_weights_strict: dict[str, float] = {}
    if have_sentiws:
        pos_current, pos_current_weights = parse_sentiws(positive_path, args.current_threshold)
        neg_current, neg_current_weights = parse_sentiws(negative_path, args.current_threshold)
        sentiws_current = list(dict.fromkeys(pos_current + neg_current))
        sentiws_weights_current = {**pos_current_weights, **neg_current_weights}

        pos_strict, pos_strict_weights = parse_sentiws(positive_path, args.strict_threshold)
        neg_strict, neg_strict_weights = parse_sentiws(negative_path, args.strict_threshold)
        sentiws_strict = list(dict.fromkeys(pos_strict + neg_strict))
        sentiws_weights_strict = {**pos_strict_weights, **neg_strict_weights}

    hyland = list(dict.fromkeys(HYLAND_ATTITUDE_DE))
    combined_current = list(dict.fromkeys(hyland + sentiws_current))
    combined_strict = list(dict.fromkeys(hyland + sentiws_strict))

    rows = []
    for article in articles:
        text = article["clean_text"]
        row = {
            **{key: article[key] for key in ["title", "label", "label_name", "topic", "word_count", "age_days"]},
            "f2_hyland_only": density(text, hyland),
        }
        if have_sentiws:
            row.update(
                {
                    "f2_sentiws_only": density(text, sentiws_current),
                    "f2_current_combined": density(text, combined_current),
                    "f2_sentiws_strict_only": density(text, sentiws_strict),
                    "f2_combined_strict": density(text, combined_strict),
                }
            )
        rows.append(row)
    features = pd.DataFrame(rows)
    features.to_csv(outdir / "f2_robustness_article_features_de.csv", index=False)

    variants = [("f2_hyland_only", "Hyland-only")]
    if have_sentiws:
        variants.extend(
            [
                ("f2_sentiws_only", "SentiWS-only |weight|>=0.5"),
                ("f2_current_combined", "Current combined Hyland+SentiWS>=0.5"),
                ("f2_sentiws_strict_only", f"SentiWS-only |weight|>={args.strict_threshold}"),
                ("f2_combined_strict", f"Hyland+strict SentiWS |weight|>={args.strict_threshold}"),
            ]
        )

    summaries = []
    diff_frames = []
    for column, label in variants:
        summary, diffs = summarize_variant(features, pairs, column, label)
        summaries.append(summary)
        diff_frames.append(diffs)
    pd.DataFrame(summaries).to_csv(outdir / "f2_robustness_summary_de.csv", index=False)
    pd.concat(diff_frames, ignore_index=True).to_csv(outdir / "f2_robustness_pair_diffs_de.csv", index=False)

    overlap_current = sorted(set(hyland) & set(sentiws_current))
    overlap_strict = sorted(set(hyland) & set(sentiws_strict))
    lexicon_audit = {
        "sentiws_files_available": have_sentiws,
        "sentiws_positive_path": str(positive_path),
        "sentiws_negative_path": str(negative_path),
        "hyland_marker_list": hyland,
        "hyland_size": len(hyland),
        "sentiws_current_threshold": args.current_threshold,
        "sentiws_current_size": len(sentiws_current),
        "sentiws_strict_threshold": args.strict_threshold,
        "sentiws_strict_size": len(sentiws_strict),
        "combined_current_size": len(combined_current),
        "combined_strict_size": len(combined_strict),
        "hyland_sentiws_current_overlap": overlap_current,
        "hyland_sentiws_current_overlap_size": len(overlap_current),
        "hyland_sentiws_strict_overlap": overlap_strict,
        "hyland_sentiws_strict_overlap_size": len(overlap_strict),
        "sentiws_current_terms": sentiws_current,
        "sentiws_strict_terms": sentiws_strict,
        "sentiws_current_weights": sentiws_weights_current,
        "sentiws_strict_weights": sentiws_weights_strict,
    }
    (outdir / "f2_lexicon_audit_de.json").write_text(
        json.dumps(lexicon_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    print(f"Saved outputs to: {outdir}", flush=True)


if __name__ == "__main__":
    main()
