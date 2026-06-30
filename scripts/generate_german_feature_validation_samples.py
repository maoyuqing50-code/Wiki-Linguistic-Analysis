#!/usr/bin/env python3
"""Generate qualitative validation samples for German linguistic features.

This script samples positive detections from the final German matched dataset.
It is intentionally descriptive: no significance tests, regressions, or model
comparison statistics are computed.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import spacy
from transformers import pipeline


RANDOM_SEED = 42
N_SAMPLES = 20

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAIRS_PATH = PROJECT_ROOT / "final_pairs_de.json"
OUT_CSV = PROJECT_ROOT / "outputs" / "german_feature_validation_samples.csv"
OUT_MD = PROJECT_ROOT / "outputs" / "german_feature_validation_summary.md"
F5_ARTICLE_SCORES = PROJECT_ROOT / "outputs" / "f5_robustness_de" / "f5_full_article_scores_de.csv"
F1_ARTICLE_FEATURES = PROJECT_ROOT / "outputs" / "f1_robustness_de" / "f1_robustness_article_features_de.csv"
F3_ARTICLE_FEATURES = PROJECT_ROOT / "outputs" / "f3_robustness_de" / "f3_robustness_article_features_de.csv"
F4_ARTICLE_FEATURES = PROJECT_ROOT / "outputs" / "f4_robustness_de" / "f4_robustness_article_features_de.csv"


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

WEASEL_WORDS_DE_ORIGINAL = [
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
GENERIC_QUANTIFIERS_REMOVED = {
    "viele",
    "mehrere",
    "verschieden",
    "verschiedene",
    "verschiedenen",
    "verschiedener",
    "einige",
}
WEASEL_WORDS_DE_FINAL = [
    word for word in WEASEL_WORDS_DE_ORIGINAL if word not in GENERIC_QUANTIFIERS_REMOVED
]

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
EXPANDED_BIASED_REPORT_DE = {
    "behaupten",
    "beschuldigen",
    "vorwerfen",
    "insistieren",
    "anklagen",
    "behauptung",
    "beschuldigung",
    "vorwurf",
    "anklage",
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

F5_LABEL_BIASED = "einseitig"
F5_LABEL_NEUTRAL = "neutral"
F5_LABELS = [F5_LABEL_BIASED, F5_LABEL_NEUTRAL]
F5_THRESHOLD = 0.75
F5_MARGIN = 0.15
F5_HYPOTHESIS = (
    "Dieser Satz enthält ein Wort oder eine Formulierung, deren Bedeutung "
    "eine einseitige Haltung gegenüber einem umstrittenen Ereignis, einer "
    "Person oder einer Behauptung voraussetzt oder nahelegt. Der Satz ist {}."
)


def load_articles() -> list[dict]:
    pairs = json.loads(PAIRS_PATH.read_text(encoding="utf-8"))
    articles = []
    seen = set()
    for pair in pairs:
        for role in ("contested", "stable"):
            article = pair[role]
            title = article["title"]
            if title in seen:
                continue
            seen.add(title)
            articles.append(
                {
                    "title": title,
                    "text": article.get("clean_text") or article.get("raw_text") or "",
                }
            )
    return articles


def match_wordlist(sentence: str, wordlist: list[str]) -> list[str]:
    sentence_lower = sentence.lower()
    matches = []
    for word in wordlist:
        word_lower = word.lower()
        if " " in word_lower:
            if word_lower in sentence_lower:
                matches.append(word)
        elif re.search(r"\b" + re.escape(word_lower) + r"\b", sentence_lower):
            matches.append(word)
    return matches


def split_sentences(text: str) -> list[str]:
    return [
        sent.strip()
        for sent in re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ0-9])", text)
        if len(sent.strip().split()) >= 4
    ]


def is_classifiable(sent_text: str, nlp) -> bool:
    tokens = sent_text.strip().split()
    if len(tokens) < 6:
        return False
    doc = nlp(sent_text)
    if not any(token.pos_ == "VERB" for token in doc):
        return False
    num_ratio = sum(1 for token in doc if token.like_num or token.pos_ == "NUM") / max(
        len(tokens), 1
    )
    return num_ratio <= 0.4


def load_total_counts(articles: list[dict]) -> Counter:
    counts = Counter()
    if F1_ARTICLE_FEATURES.exists():
        f1 = pd.read_csv(F1_ARTICLE_FEATURES)
        counts["F1"] = int((f1["f1_lexical_count"] + f1["f1_konjunktiv_ii_count"]).sum())
    if F3_ARTICLE_FEATURES.exists():
        f3 = pd.read_csv(F3_ARTICLE_FEATURES)
        counts["F3"] = int((f3["f3_no_generic_quantifiers"] * f3["word_count"] / 1000).round().sum())
    if F4_ARTICLE_FEATURES.exists():
        f4 = pd.read_csv(F4_ARTICLE_FEATURES)
        counts["F4"] = int(f4["f4_expanded_total_count"].sum())
    counts["F6"] = sum(
        len(match_wordlist(article["text"], CONTRASTIVE_TRANSITIONS_DE)) for article in articles
    )
    if F5_ARTICLE_SCORES.exists():
        f5_scores = pd.read_csv(F5_ARTICLE_SCORES)
        counts["F5"] = int(
            (f5_scores["f5_revised"] * f5_scores["n_classifiable_sentences"])
            .round()
            .fillna(0)
            .sum()
        )
    return counts


def collect_symbolic_detections(articles: list[dict], nlp, rng: random.Random) -> tuple[list[dict], list[str]]:
    detections = []
    issues = []
    sentence_rows = []
    for article in articles:
        for sentence in split_sentences(article["text"]):
            sentence_rows.append((article["title"], sentence))
    rng.shuffle(sentence_rows)

    feature_sample_counts = Counter()
    for title, sentence in sentence_rows:
        if all(feature_sample_counts[feature] >= N_SAMPLES for feature in ["F1", "F3", "F4", "F6"]):
            break
        lexical_hits = (
            match_wordlist(sentence, HEDGES_DE)
            + match_wordlist(sentence, WEASEL_WORDS_DE_FINAL)
            + match_wordlist(sentence, CONTRASTIVE_TRANSITIONS_DE)
        )
        needs_nlp = feature_sample_counts["F4"] < N_SAMPLES or bool(lexical_hits)
        sent_doc = nlp(sentence) if needs_nlp else None

        if feature_sample_counts["F1"] < N_SAMPLES:
            for phrase in match_wordlist(sentence, HEDGES_DE):
                detections.append(
                    {
                        "feature": "F1",
                        "article_title": title,
                        "matched_phrase": phrase,
                        "sentence": sentence,
                    }
                )
                feature_sample_counts["F1"] += 1
                if feature_sample_counts["F1"] >= N_SAMPLES:
                    break
            if feature_sample_counts["F1"] < N_SAMPLES and sent_doc is not None:
                for token in sent_doc:
                    if (
                        token.lemma_.lower() in KONJUNKTIV_II_MODALS
                        and "Mood=Sub" in str(token.morph)
                        and "Tense=Past" in str(token.morph)
                    ):
                        detections.append(
                            {
                                "feature": "F1",
                                "article_title": title,
                                "matched_phrase": f"Konjunktiv-II:{token.text}",
                                "sentence": sentence,
                            }
                        )
                        feature_sample_counts["F1"] += 1
                        if feature_sample_counts["F1"] >= N_SAMPLES:
                            break

        if feature_sample_counts["F3"] < N_SAMPLES:
            for phrase in match_wordlist(sentence, WEASEL_WORDS_DE_FINAL):
                detections.append(
                    {
                        "feature": "F3",
                        "article_title": title,
                        "matched_phrase": phrase,
                        "sentence": sentence,
                    }
                )
                feature_sample_counts["F3"] += 1
                if feature_sample_counts["F3"] >= N_SAMPLES:
                    break
            if feature_sample_counts["F3"] < N_SAMPLES and sent_doc is not None:
                for token in sent_doc:
                    if (
                        token.pos_ == "VERB"
                        and "Mood=Sub" in str(token.morph)
                        and "Tense=Pres" in str(token.morph)
                    ):
                        detections.append(
                            {
                                "feature": "F3",
                                "article_title": title,
                                "matched_phrase": f"Konjunktiv-I:{token.text}",
                                "sentence": sentence,
                            }
                        )
                        feature_sample_counts["F3"] += 1
                        if feature_sample_counts["F3"] >= N_SAMPLES:
                            break

        if feature_sample_counts["F4"] < N_SAMPLES and sent_doc is not None:
            for token in sent_doc:
                if token.pos_ in ("VERB", "NOUN"):
                    lemma = token.lemma_.lower()
                    if lemma in EXPANDED_BIASED_REPORT_DE or lemma in CURRENT_NEUTRAL_REPORT_DE:
                        role = "biased" if lemma in EXPANDED_BIASED_REPORT_DE else "neutral"
                        detections.append(
                            {
                                "feature": "F4",
                                "article_title": title,
                                "matched_phrase": f"{role}:{token.text}/{lemma}",
                                "sentence": sentence,
                            }
                        )
                        feature_sample_counts["F4"] += 1
                        if feature_sample_counts["F4"] >= N_SAMPLES:
                            break

        if feature_sample_counts["F6"] < N_SAMPLES:
            for phrase in match_wordlist(sentence, CONTRASTIVE_TRANSITIONS_DE):
                detections.append(
                    {
                        "feature": "F6",
                        "article_title": title,
                        "matched_phrase": phrase,
                        "sentence": sentence,
                    }
                )
                feature_sample_counts["F6"] += 1
                if feature_sample_counts["F6"] >= N_SAMPLES:
                    break

    if any(
        d["matched_phrase"] in GENERIC_QUANTIFIERS_REMOVED
        for d in detections
        if d["feature"] == "F3"
    ):
        issues.append("F3 generic quantifier removal did not fully apply.")
    for feature in ["F1", "F3", "F4", "F6"]:
        if feature_sample_counts[feature] < N_SAMPLES:
            issues.append(f"{feature} produced only {feature_sample_counts[feature]} sampled detections.")
    return detections, issues


def collect_f5_detections(articles: list[dict], nlp, rng: random.Random) -> tuple[list[dict], int, list[str]]:
    issues = []
    candidates = []
    for article in articles:
        for sentence in split_sentences(article["text"]):
            if is_classifiable(sentence, nlp):
                candidates.append({"article_title": article["title"], "sentence": sentence})

    rng.shuffle(candidates)
    detections = []
    processed = 0
    nli = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        tokenizer="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        device=-1,
    )

    for start in range(0, len(candidates), 16):
        batch = candidates[start : start + 16]
        processed += len(batch)
        results = nli(
            [item["sentence"][:512] for item in batch],
            candidate_labels=F5_LABELS,
            hypothesis_template=F5_HYPOTHESIS,
            batch_size=16,
            multi_label=False,
        )
        if isinstance(results, dict):
            results = [results]
        for item, result in zip(batch, results):
            scores = dict(zip(result["labels"], result["scores"]))
            biased = scores.get(F5_LABEL_BIASED, 0.0)
            neutral = scores.get(F5_LABEL_NEUTRAL, 0.0)
            if biased > F5_THRESHOLD and (biased - neutral) > F5_MARGIN:
                detections.append(
                    {
                        "feature": "F5",
                        "article_title": item["article_title"],
                        "matched_phrase": f"einseitig_score={biased:.3f};margin={biased-neutral:.3f}",
                        "sentence": item["sentence"],
                    }
                )
                if len(detections) >= N_SAMPLES:
                    return detections, processed, issues

    if len(detections) < N_SAMPLES:
        issues.append(f"F5 produced only {len(detections)} sampled positives before candidates ended.")
    return detections, processed, issues


def sample_feature(detections: list[dict], feature: str, rng: random.Random) -> list[dict]:
    rows = [row for row in detections if row["feature"] == feature]
    if len(rows) <= N_SAMPLES:
        return rows
    return rng.sample(rows, N_SAMPLES)


def write_outputs(samples: list[dict], counts: Counter, issues: list[str], f5_processed: int) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["feature", "article_title", "matched_phrase", "sentence"])
        writer.writeheader()
        writer.writerows(samples)

    plausible = {
        "F1": "Yes: sampled matches are lexical hedge terms or Konjunktiv-II modal forms.",
        "F3": "Mostly yes: sampled matches are vague/evaluative terms or Konjunktiv-I markers; generic quantifiers are excluded.",
        "F4": "Yes: sampled matches are attribution verbs/nominalisations from the expanded final lexicon.",
        "F5": "Requires human review: positives follow the final German NLI rule, but the matched phrase is a model score rather than a lexical span.",
        "F6": "Yes: sampled matches are German adversative/contrastive connectors.",
    }

    lines = [
        "# German Feature Extraction Validation Samples",
        "",
        "Purpose: qualitative validation evidence for Methods > Validation. No significance tests, p-values, or regressions were computed.",
        "",
        "Final implementation choices used for sampling:",
        "",
        "- F1: original German hedge lexicon plus Konjunktiv-II modal morphology.",
        "- F3: no-generic-quantifiers robustness variant.",
        "- F4: expanded attribution lexicon used by the final improved German model.",
            "- F5: revised German NLI robustness rule (`einseitig > 0.75` and margin `> 0.15`).",
        "- F6: original German contrastive-transition lexicon.",
        "",
        "## Detection Counts",
        "",
        "| Feature | Total detected instances | Sampled instances | Plausibility note |",
        "|---|---:|---:|---|",
    ]
    for feature in ["F1", "F3", "F4", "F5", "F6"]:
        sampled = sum(1 for row in samples if row["feature"] == feature)
        total = counts[feature]
        lines.append(f"| {feature} | {total} | {sampled} | {plausible[feature]} |")
    lines.extend(
        [
            "",
            "## Automatic Issue Checks",
            "",
        ]
    )
    if issues:
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.append("- No obvious automatic implementation issues were detected in the sampled extraction pass.")
    lines.extend(
        [
            f"- F5 sentence-level validation processed {f5_processed} randomly ordered classifiable sentences to obtain sampled positives.",
            "- F5 total detected instances are reconstructed from the saved full-dataset article scores as `round(f5_revised * n_classifiable_sentences)` because sentence-level decisions were not persisted.",
            "- F5 validation does not identify a lexical span; the `matched_phrase` column records the model decision score and margin.",
            "",
            "## Qualitative Assessment",
            "",
            "The symbolic feature samples are linguistically plausible under their implemented definitions: F1 finds epistemic uncertainty markers, F3 finds vague/evaluative or indirect-speech markers, F4 finds reporting/attribution items, and F6 finds contrastive discourse connectors. F5 samples are plausible only as model-level sentence detections and should be reviewed manually because the NLI feature does not expose the exact lexical item responsible for the decision.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    articles = load_articles()
    counts = load_total_counts(articles)
    nlp = spacy.load("de_core_news_sm", disable=["parser", "ner"])
    detections, issues = collect_symbolic_detections(articles, nlp, rng)
    f5_detections, f5_processed, f5_issues = collect_f5_detections(articles, nlp, rng)
    if "F5" not in counts:
        counts["F5"] = len(f5_detections)
        issues.append("F5 article-level score file was missing; total F5 count is sampled detections only.")
    issues.extend(f5_issues)

    samples = []
    for feature in ["F1", "F3", "F4", "F6"]:
        samples.extend(sample_feature(detections, feature, rng))
    samples.extend(f5_detections[:N_SAMPLES])
    samples.sort(key=lambda row: (row["feature"], row["article_title"]))
    write_outputs(samples, counts, issues, f5_processed)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
