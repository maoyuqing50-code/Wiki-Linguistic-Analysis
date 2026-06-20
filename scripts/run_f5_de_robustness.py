#!/usr/bin/env python3
"""Run full-dataset German F5 robustness scoring without overwriting originals.

This script uses only the existing ``final_pairs_de.json`` dataset. It writes
separate outputs under ``outputs/f5_robustness_de`` and checkpoints after each
article so interrupted runs can resume safely.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
import torch
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

ORIGINAL_LABELS = ["biased", "neutral"]
ORIGINAL_HYPOTHESIS = (
    "This sentence contains a word or phrase whose meaning presupposes or "
    "entails a one-sided stance towards a contested event, person, or claim. "
    "The sentence is {}."
)

REVISED_LABELS = ["einseitig", "neutral"]
REVISED_HYPOTHESIS = (
    "Dieser Satz enthält ein Wort oder eine Formulierung, deren Bedeutung "
    "eine einseitige Haltung gegenüber einem umstrittenen Ereignis, einer "
    "Person oder einer Behauptung voraussetzt oder nahelegt. Der Satz ist {}."
)
REVISED_THRESHOLD = 0.75
REVISED_MARGIN = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="final_pairs_de.json")
    parser.add_argument("--outdir", default="outputs/f5_robustness_de")
    parser.add_argument("--batch-pairs", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
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


def is_classifiable(nlp, sentence: str) -> bool:
    tokens = sentence.strip().split()
    if len(tokens) < 6:
        return False
    doc = nlp(sentence)
    if not any(token.pos_ == "VERB" for token in doc):
        return False
    num_ratio = sum(1 for token in doc if token.like_num or token.pos_ == "NUM") / max(
        len(tokens), 1
    )
    return num_ratio <= 0.4


def classifiable_sentences(nlp, text: str) -> list[str]:
    if not text or not text.strip():
        return []
    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        text = sent.text.strip()
        if is_classifiable(nlp, text):
            sentences.append(text[:512])
    return sentences


def entailment_id(model) -> int:
    for label, idx in model.config.label2id.items():
        if "entail" in label.lower():
            return int(idx)
    return 0


def score_zero_shot(
    *,
    sentences: list[str],
    labels: list[str],
    hypothesis: str,
    tokenizer,
    model,
    device: torch.device,
    entail_id: int,
    batch_pairs: int,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not sentences:
        return np.array([]), np.array([])

    hypotheses = [hypothesis.format(label) for label in labels]
    total_pairs = len(sentences) * len(labels)
    logits: list[float] = []

    for start in range(0, total_pairs, batch_pairs):
        end = min(total_pairs, start + batch_pairs)
        premises = []
        hypothesis_batch = []
        for flat_index in range(start, end):
            sent_index = flat_index // len(labels)
            label_index = flat_index % len(labels)
            premises.append(sentences[sent_index])
            hypothesis_batch.append(hypotheses[label_index])

        encoded = tokenizer(
            premises,
            hypothesis_batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            batch_logits = model(**encoded).logits[:, entail_id].detach().cpu().float()
        logits.extend(batch_logits.numpy().tolist())

    logits_array = np.array(logits).reshape(len(sentences), len(labels))
    logits_array = logits_array - logits_array.max(axis=1, keepdims=True)
    probs = np.exp(logits_array) / np.exp(logits_array).sum(axis=1, keepdims=True)
    return probs[:, 0], probs[:, 1]


def score_article(
    article: dict,
    *,
    nlp,
    tokenizer,
    model,
    device: torch.device,
    entail_id_value: int,
    batch_pairs: int,
    max_length: int,
) -> dict:
    sentences = classifiable_sentences(nlp, article["clean_text"])
    n_sentences = len(sentences)
    base = {
        "title": article["title"],
        "label": article["label"],
        "label_name": article["label_name"],
        "topic": article["topic"],
        "word_count": article["word_count"],
        "age_days": article["age_days"],
        "n_classifiable_sentences": n_sentences,
    }
    if n_sentences == 0:
        return {
            **base,
            "f5_original_reproduced": np.nan,
            "f5_original_biased_mean": np.nan,
            "f5_original_neutral_mean": np.nan,
            "f5_revised": np.nan,
            "f5_revised_biased_mean": np.nan,
            "f5_revised_neutral_mean": np.nan,
            "f5_revised_margin_mean": np.nan,
        }

    original_biased, original_neutral = score_zero_shot(
        sentences=sentences,
        labels=ORIGINAL_LABELS,
        hypothesis=ORIGINAL_HYPOTHESIS,
        tokenizer=tokenizer,
        model=model,
        device=device,
        entail_id=entail_id_value,
        batch_pairs=batch_pairs,
        max_length=max_length,
    )
    revised_biased, revised_neutral = score_zero_shot(
        sentences=sentences,
        labels=REVISED_LABELS,
        hypothesis=REVISED_HYPOTHESIS,
        tokenizer=tokenizer,
        model=model,
        device=device,
        entail_id=entail_id_value,
        batch_pairs=batch_pairs,
        max_length=max_length,
    )
    revised_margin = revised_biased - revised_neutral
    revised_passes = (revised_biased > REVISED_THRESHOLD) & (
        revised_margin > REVISED_MARGIN
    )

    return {
        **base,
        "f5_original_reproduced": round(float((original_biased > 0.50).mean()), 4),
        "f5_original_biased_mean": round(float(original_biased.mean()), 4),
        "f5_original_neutral_mean": round(float(original_neutral.mean()), 4),
        "f5_revised": round(float(revised_passes.mean()), 4),
        "f5_revised_biased_mean": round(float(revised_biased.mean()), 4),
        "f5_revised_neutral_mean": round(float(revised_neutral.mean()), 4),
        "f5_revised_margin_mean": round(float(revised_margin.mean()), 4),
    }


def append_checkpoint(path: Path, row: dict) -> None:
    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


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
            "macro_f1_with_f5_topic": np.nan,
            "macro_f1_topic_baseline": np.nan,
            "macro_f1_contribution": np.nan,
            "n_model_rows": len(model_df),
        }

    min_class_count = int(pd.Series(y).value_counts().min())
    n_splits = min(5, min_class_count)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)

    def macro_f1(columns: list[str]) -> float:
        if not columns:
            return float(max(y.mean(), 1 - y.mean()))
        x_values = StandardScaler().fit_transform(model_df[columns].values)
        return float(cross_val_score(lr, x_values, y, cv=cv, scoring="f1_macro").mean())

    baseline_f1 = macro_f1(topic_cols)
    with_f5_f1 = macro_f1([column] + topic_cols)
    x_coef = StandardScaler().fit_transform(model_df[[column] + topic_cols].values)
    lr.fit(x_coef, y)
    return {
        "logistic_coefficient": float(lr.coef_[0][0]),
        "macro_f1_with_f5_topic": with_f5_f1,
        "macro_f1_topic_baseline": baseline_f1,
        "macro_f1_contribution": with_f5_f1 - baseline_f1,
        "n_model_rows": len(model_df),
    }


def summarize(df: pd.DataFrame, pairs: list[dict], column: str, label: str) -> tuple[dict, pd.DataFrame]:
    contested_values = df[df["label"] == 0][column]
    stable_values = df[df["label"] == 1][column]
    diffs = pair_differences(df, pairs, column)
    return (
        {
            "Feature": label,
            "Column": column,
            "Contested mean": float(contested_values.mean()),
            "Stable mean": float(stable_values.mean()),
            "Percent difference": float(
                (contested_values.mean() - stable_values.mean())
                / (abs(stable_values.mean()) + 1e-9)
                * 100
            ),
            "NaN rate": float(df[column].isna().mean()),
            "Saturation rate >0.90": float((df[column].dropna() > 0.90).mean()),
            **wilcoxon_metrics(diffs),
            **logistic_metrics(df, column),
        },
        diffs,
    )


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    article_path = outdir / "f5_full_article_scores_de.csv"
    summary_path = outdir / "f5_robustness_summary_de.csv"
    pair_path = outdir / "f5_robustness_pair_diffs_de.csv"

    articles, pairs = load_articles(Path(args.pairs))
    if args.limit:
        articles = articles[: args.limit]

    print(
        f"Articles: {len(articles)} "
        f"(contested={sum(a['label'] == 0 for a in articles)}, "
        f"stable={sum(a['label'] == 1 for a in articles)})",
        flush=True,
    )
    print(f"Output directory: {outdir}", flush=True)

    done_titles: set[str] = set()
    if article_path.exists():
        existing = pd.read_csv(article_path)
        done_titles = set(existing["title"].astype(str))
        print(f"Resuming from checkpoint: {len(done_titles)} articles done", flush=True)

    nlp = spacy.load("de_core_news_sm", disable=["ner"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    entail_id_value = entailment_id(model)
    print(f"Device: {device}; entailment id: {entail_id_value}", flush=True)

    start_time = time.time()
    for index, article in enumerate(articles, start=1):
        if article["title"] in done_titles:
            continue
        row = score_article(
            article,
            nlp=nlp,
            tokenizer=tokenizer,
            model=model,
            device=device,
            entail_id_value=entail_id_value,
            batch_pairs=args.batch_pairs,
            max_length=args.max_length,
        )
        append_checkpoint(article_path, row)
        done_titles.add(article["title"])
        print(
            f"[{len(done_titles):>3}/{len(articles)}] "
            f"{article['label_name']:<9} "
            f"{row['n_classifiable_sentences']:>4} sentences "
            f"orig={row['f5_original_reproduced']!s:<6} "
            f"rev={row['f5_revised']!s:<6} "
            f"{article['title'][:55]} "
            f"elapsed={time.time() - start_time:.1f}s",
            flush=True,
        )

    score_df = pd.read_csv(article_path).drop_duplicates("title", keep="last")
    original_summary, original_diffs = summarize(
        score_df, pairs, "f5_original_reproduced", "Original F5 reproduced"
    )
    revised_summary, revised_diffs = summarize(score_df, pairs, "f5_revised", "Revised F5")

    summary_df = pd.DataFrame([original_summary, revised_summary])
    summary_df.to_csv(summary_path, index=False)
    pd.concat([original_diffs, revised_diffs], ignore_index=True).to_csv(
        pair_path, index=False
    )
    print(summary_df.to_string(index=False), flush=True)
    print(f"Saved: {article_path}", flush=True)
    print(f"Saved: {summary_path}", flush=True)
    print(f"Saved: {pair_path}", flush=True)


if __name__ == "__main__":
    main()
