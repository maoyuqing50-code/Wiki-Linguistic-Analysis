#!/usr/bin/env python3
"""Compare available original vs improved German v4 feature sets.

The repository-level ``final_features_de.csv`` is used as the requested base
table for article metadata and labels. It does not contain the full v4
linguistic F1-F7 matrix, so this script merges the saved robustness outputs and
recomputes F6/F7 from ``final_pairs_de.json`` using the German v4 notebook
definitions. Original F2 is unchanged between models but is unavailable in the
current repository inputs; both models therefore omit F2 and the limitation is
recorded alongside the outputs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from lexicalrichness import LexicalRichness
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from statsmodels.stats.multitest import multipletests


CONTRASTIVE_TRANSITIONS_DE = [
    "aber",
    "allein",
    "doch",
    "jedoch",
    "sondern",
    "während",
    "allerdings",
    "im Gegensatz dazu",
    "demgegenüber",
    "dennoch",
    "hingegen",
    "wohingegen",
    "dagegen",
]

ORIGINAL_FEATURES = [
    "f1_original_recomputed",
    "f3_original_recomputed",
    "f4_current_ratio_min3",
    "f5_original_reproduced",
    "f6_contrastive_recomputed",
    "f7_mtld_recomputed",
]

IMPROVED_FEATURES = [
    "f1_original_recomputed",
    "f3_no_generic_quantifiers",
    "f4_expanded_smoothed_ratio",
    "f5_revised",
    "f6_contrastive_recomputed",
    "f7_mtld_recomputed",
]

FEATURE_LABELS = {
    "f1_original_recomputed": "F1 original",
    "f3_original_recomputed": "F3 original",
    "f3_no_generic_quantifiers": "F3 no generic quantifiers",
    "f4_current_ratio_min3": "F4 original",
    "f4_expanded_smoothed_ratio": "F4 expanded smoothed ratio",
    "f5_original_reproduced": "F5 original",
    "f5_revised": "F5 revised robustness",
    "f6_contrastive_recomputed": "F6 original recomputed",
    "f7_mtld_recomputed": "F7 original recomputed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="final_features_de.csv")
    parser.add_argument("--pairs", default="final_pairs_de.json")
    parser.add_argument("--f1", default="outputs/f1_robustness_de/f1_robustness_article_features_de.csv")
    parser.add_argument("--f3", default="outputs/f3_robustness_de/f3_robustness_article_features_de.csv")
    parser.add_argument("--f4", default="outputs/f4_robustness_de/f4_robustness_article_features_de.csv")
    parser.add_argument("--f5", default="outputs/f5_robustness_de/f5_full_article_scores_de.csv")
    parser.add_argument("--outdir", default="outputs/de_original_vs_improved")
    return parser.parse_args()


def _density_count(text_lower: str, wordlist: list[str]) -> int:
    count = 0
    for word in wordlist:
        word_lower = word.lower()
        if " " in word_lower:
            count += text_lower.count(word_lower)
        else:
            count += len(re.findall(r"\b" + re.escape(word_lower) + r"\b", text_lower))
    return count


def f6_contrastive_density(text: str) -> float:
    if not text or not text.strip():
        return 0.0
    text_lower = text.lower()
    n_words = max(len(text_lower.split()), 1)
    return round(_density_count(text_lower, CONTRASTIVE_TRANSITIONS_DE) / n_words * 1000, 4)


def f7_mtld(text: str) -> float:
    if not text or len(text.split()) < 50:
        return np.nan
    try:
        return round(LexicalRichness(text).mtld(threshold=0.72), 4)
    except Exception:
        return np.nan


def article_text_table(pairs: list[dict]) -> pd.DataFrame:
    rows = []
    seen = set()
    for pair in pairs:
        for side in ("contested", "stable"):
            article = pair[side]
            title = article["title"]
            if title in seen:
                continue
            seen.add(title)
            text = article.get("clean_text") or article.get("raw_text") or ""
            rows.append(
                {
                    "title": title,
                    "f6_contrastive_recomputed": f6_contrastive_density(text),
                    "f7_mtld_recomputed": f7_mtld(text),
                }
            )
    return pd.DataFrame(rows)


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, list[dict], dict]:
    paths = [args.features, args.pairs, args.f1, args.f3, args.f4, args.f5]
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    base = pd.read_csv(args.features)
    pairs = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
    f1 = pd.read_csv(args.f1)[["title", "f1_original_recomputed"]]
    f3 = pd.read_csv(args.f3)[["title", "f3_original_recomputed", "f3_no_generic_quantifiers"]]
    f4 = pd.read_csv(args.f4)[["title", "f4_current_ratio_min3", "f4_expanded_smoothed_ratio"]]
    f5 = pd.read_csv(args.f5)[["title", "f5_original_reproduced", "f5_revised"]]
    f6_f7 = article_text_table(pairs)

    merged = base[["title", "label", "label_name", "topic", "word_count", "age_days"]].copy()
    for feature_df in (f1, f3, f4, f5, f6_f7):
        merged = merged.merge(feature_df, on="title", how="left")

    required = {"title", "label", "topic", *ORIGINAL_FEATURES, *IMPROVED_FEATURES}
    missing_columns = sorted(required - set(merged.columns))
    if missing_columns:
        raise ValueError(f"Merged feature matrix is missing required columns: {missing_columns}")

    limitation = {
        "status": "completed_with_limitation",
        "base_features_loaded": args.features,
        "base_features_shape": list(base.shape),
        "omitted_unchanged_feature": "F2 original Hyland + SentiWS",
        "reason": (
            "final_features_de.csv exists, but it does not contain f2_affective or the "
            "full German v4 F1-F7 matrix. Because F2 is unchanged in the Original and "
            "Improved specifications, both fitted models omit F2. The comparison is "
            "therefore a valid estimate of the robustness-change delta for F3/F4/F5, "
            "but not an exact F2-inclusive reproduction of the original notebook model."
        ),
        "original_features_used": ORIGINAL_FEATURES,
        "improved_features_used": IMPROVED_FEATURES,
    }
    return merged, pairs, limitation


def pair_diffs(df: pd.DataFrame, pairs: list[dict], feature: str) -> pd.Series:
    by_title = df.set_index("title")
    diffs = []
    for pair in pairs:
        contested_title = pair["contested"]["title"]
        stable_title = pair["stable"]["title"]
        if contested_title not in by_title.index or stable_title not in by_title.index:
            continue
        contested_value = by_title.loc[contested_title, feature]
        stable_value = by_title.loc[stable_title, feature]
        if pd.isna(contested_value) or pd.isna(stable_value):
            continue
        diffs.append(contested_value - stable_value)
    return pd.Series(diffs, name=feature)


def wilcoxon_table(df: pd.DataFrame, pairs: list[dict], features: list[str], model_name: str) -> pd.DataFrame:
    rows = []
    for feature in features:
        diffs = pair_diffs(df, pairs, feature)
        if len(diffs) < 5:
            continue
        stat, p_raw = wilcoxon(diffs.values)
        n_pairs = len(diffs)
        mean_w = n_pairs * (n_pairs + 1) / 4
        std_w = np.sqrt(n_pairs * (n_pairs + 1) * (2 * n_pairs + 1) / 24)
        z_value = (stat - mean_w) / std_w
        rows.append(
            {
                "model": model_name,
                "feature": feature,
                "feature_label": FEATURE_LABELS.get(feature, feature),
                "n_pairs": n_pairs,
                "mean_pair_diff": float(diffs.mean()),
                "pct_contested_higher": float((diffs > 0).mean() * 100),
                "p_raw": float(p_raw),
                "effect_r": float(abs(z_value) / np.sqrt(n_pairs)),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        _, p_holm, _, _ = multipletests(result["p_raw"], method="holm")
        result["p_holm"] = p_holm
    return result


def model_frame(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    topic_dummies = pd.get_dummies(df["topic"], prefix="topic", drop_first=True)
    model_df = pd.concat(
        [df[["label"] + features].reset_index(drop=True), topic_dummies.reset_index(drop=True)],
        axis=1,
    ).dropna()
    return model_df, features + list(topic_dummies.columns)


def evaluate_model(df: pd.DataFrame, features: list[str], model_name: str) -> tuple[dict, pd.DataFrame]:
    model_df, feature_columns = model_frame(df, features)
    x_values = model_df[feature_columns].values
    y_values = model_df["label"].values

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred = cross_val_predict(pipeline, x_values, y_values, cv=cv, method="predict")
    proba = cross_val_predict(pipeline, x_values, y_values, cv=cv, method="predict_proba")[:, 1]

    pipeline.fit(x_values, y_values)
    lr = pipeline.named_steps["logistic"]
    coefs = pd.DataFrame(
        {
            "model": model_name,
            "feature": feature_columns,
            "feature_label": [FEATURE_LABELS.get(feature, feature) for feature in feature_columns],
            "coefficient": lr.coef_[0],
            "abs_coefficient": np.abs(lr.coef_[0]),
        }
    ).sort_values("abs_coefficient", ascending=False)

    metrics = {
        "model": model_name,
        "n_rows": int(len(model_df)),
        "n_features_without_topic_controls": int(len(features)),
        "n_topic_controls": int(len(feature_columns) - len(features)),
        "accuracy": float(accuracy_score(y_values, pred)),
        "macro_f1": float(f1_score(y_values, pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_values, proba)),
        "baseline_accuracy": float(max(np.mean(y_values), 1 - np.mean(y_values))),
    }
    return metrics, coefs


def write_interpretation(outdir: Path, metrics: pd.DataFrame, importance: pd.DataFrame) -> None:
    original = metrics.loc[metrics["model"] == "Original available"].iloc[0]
    improved = metrics.loc[metrics["model"] == "Improved available"].iloc[0]
    delta_f1 = improved["macro_f1"] - original["macro_f1"]
    delta_acc = improved["accuracy"] - original["accuracy"]
    changed = importance[
        (importance["model"] == "Improved available")
        & importance["feature"].isin(
            ["f3_no_generic_quantifiers", "f4_expanded_smoothed_ratio", "f5_revised"]
        )
    ].sort_values("abs_coefficient", ascending=False)
    top_changed = changed.iloc[0]["feature_label"] if not changed.empty else "n/a"
    practical = "yes" if delta_f1 >= 0.03 or delta_acc >= 0.03 else "limited"
    text = f"""German Original vs Improved Feature Evaluation

Result: the improved available model changes accuracy by {delta_acc:+.3f} and macro F1 by {delta_f1:+.3f} relative to the original available model.

Strongest changed feature by absolute logistic coefficient: {top_changed}.

Practical interpretation: {practical}. The robustness variants strengthen the German results only if the macro-F1/accuracy deltas are materially positive; otherwise they mostly improve feature validity and missingness diagnostics rather than downstream classification.

Limitation: F2 original is unchanged in the requested design but is absent from final_features_de.csv, so both models omit F2. This preserves the F3/F4/F5 robustness comparison delta but is not an exact F2-inclusive reproduction of the notebook's seven-feature model.
"""
    (outdir / "short_interpretation_original_vs_improved_de.txt").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df, pairs, limitation = load_inputs(args)
    (outdir / "limitations_original_vs_improved_de.json").write_text(
        json.dumps(limitation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    df.to_csv(outdir / "merged_feature_matrix_original_vs_improved_de.csv", index=False)

    wilcoxon_original = wilcoxon_table(df, pairs, ORIGINAL_FEATURES, "Original available")
    wilcoxon_improved = wilcoxon_table(df, pairs, IMPROVED_FEATURES, "Improved available")
    wilcoxon_results = pd.concat([wilcoxon_original, wilcoxon_improved], ignore_index=True)
    wilcoxon_results.to_csv(outdir / "wilcoxon_original_vs_improved_de.csv", index=False)

    original_metrics, original_importance = evaluate_model(df, ORIGINAL_FEATURES, "Original available")
    improved_metrics, improved_importance = evaluate_model(df, IMPROVED_FEATURES, "Improved available")
    metrics = pd.DataFrame([original_metrics, improved_metrics])
    metrics.to_csv(outdir / "model_metrics_original_vs_improved_de.csv", index=False)

    importance = pd.concat([original_importance, improved_importance], ignore_index=True)
    importance.to_csv(outdir / "feature_importance_original_vs_improved_de.csv", index=False)

    comparison = pd.DataFrame(
        [
            {
                "Metric": "Rows used",
                "Original Model": original_metrics["n_rows"],
                "Improved Model": improved_metrics["n_rows"],
                "Delta": improved_metrics["n_rows"] - original_metrics["n_rows"],
            },
            {
                "Metric": "Accuracy",
                "Original Model": original_metrics["accuracy"],
                "Improved Model": improved_metrics["accuracy"],
                "Delta": improved_metrics["accuracy"] - original_metrics["accuracy"],
            },
            {
                "Metric": "Macro F1",
                "Original Model": original_metrics["macro_f1"],
                "Improved Model": improved_metrics["macro_f1"],
                "Delta": improved_metrics["macro_f1"] - original_metrics["macro_f1"],
            },
            {
                "Metric": "ROC-AUC",
                "Original Model": original_metrics["roc_auc"],
                "Improved Model": improved_metrics["roc_auc"],
                "Delta": improved_metrics["roc_auc"] - original_metrics["roc_auc"],
            },
            {
                "Metric": "Baseline accuracy",
                "Original Model": original_metrics["baseline_accuracy"],
                "Improved Model": improved_metrics["baseline_accuracy"],
                "Delta": improved_metrics["baseline_accuracy"] - original_metrics["baseline_accuracy"],
            },
        ]
    )
    comparison.to_csv(outdir / "comparison_table_original_vs_improved_de.csv", index=False)
    write_interpretation(outdir, metrics, importance)
    print(comparison.to_string(index=False))
    print(f"\nSaved outputs to {outdir}")


if __name__ == "__main__":
    main()
