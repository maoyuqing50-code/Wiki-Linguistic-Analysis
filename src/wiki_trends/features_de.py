"""German linguistic feature extraction for Wikipedia article records.

The functions in this module mirror the English notebook outputs while using
German lexical resources and a German spaCy pipeline for tokenization and
sentence segmentation.
"""

from __future__ import annotations

import re
import json
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover - only needed for matrix construction
    pd = None

try:
    import spacy
except ImportError:  # pragma: no cover - handled by load_german_nlp
    spacy = None

try:
    from lexicalrichness import LexicalRichness
except ImportError:  # pragma: no cover - returns NaN if unavailable
    LexicalRichness = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - progress bar is optional
    tqdm = None


SPACY_MODEL = "de_core_news_sm"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
CONTESTED_FILE = RAW_DATA_DIR / "final_contested_de.json"
STABLE_FILE = RAW_DATA_DIR / "final_stable_de.json"
FEATURES_FILE = PROCESSED_DATA_DIR / "final_features_de.csv"

FEATURE_COLUMNS = [
    "hedging_density",
    "weasel_density",
    "def_ratio",
    "mtld",
]

HEDGE_PATTERNS = [
    r"\bangeblich(?:e[snmr]?)?\b",
    r"\banscheinend\b",
    r"\boffenbar\b",
    r"\bvermutlich\b",
    r"\bwomöglich\b",
    r"\bmöglicherweise\b",
    r"\beventuell\b",
    r"\bmutma(?:ß|ss)lich(?:e[snmr]?)?\b",
    r"\bwahrscheinlich\b",
    r"\bwohl\b",
    r"\bumstritten(?:e[snmr]?)?\b",
    r"\bkontrovers(?:e[snmr]?)?\b",
    r"\bstrittig(?:e[snmr]?)?\b",
    r"\bfraglich(?:e[snmr]?)?\b",
    r"\bnicht eindeutig\b",
    r"\bnicht abschließend geklärt\b",
    r"\b(?:es\s+)?wird\s+(?:behauptet|vermutet|angenommen|diskutiert)\b",
    r"\b(?:es\s+)?wurde\s+(?:behauptet|vermutet|angenommen|diskutiert)\b",
    r"\b(?:einige|manche|mehrere)\s+(?:behaupten|vermuten|meinen|argumentieren)\b",
    r"\b(?:kritiker|gegner|befürworter)\s+(?:behaupten|meinen|argumentieren)\b",
]

WEASEL_PATTERNS = [
    r"\beinige\b",
    r"\bmanche\b",
    r"\bviele\b",
    r"\bzahlreiche\b",
    r"\bmehrere\b",
    r"\bverschiedene\b",
    r"\bgewisse\b",
    r"\bbestimmte\b",
    r"\bhäufig\b",
    r"\boft\b",
    r"\bweitgehend\b",
    r"\ballgemein\b",
    r"\bangeblich\b",
    r"\boffensichtlich\b",
    r"\bzweifellos\b",
    r"\bbemerkenswert(?:erweise)?\b",
    r"\binteressanterweise\b",
    r"\bwichtig(?:erweise)?\b",
    r"\bes\s+heißt\b",
    r"\bes\s+wird\s+(?:gesagt|angenommen|behauptet|vermutet)\b",
    r"\b(?:experten|forscher|wissenschaftler|historiker|kritiker)\s+(?:sagen|meinen|behaupten|argumentieren)\b",
    r"\b(?:studien|untersuchungen|forschungen)\s+(?:zeigen|belegen|legen nahe)\b",
]

DEFINITION_PATTERNS = [
    r"\bist\s+(?:ein|eine|eines|einer|einem|einen)\b",
    r"\bwar\s+(?:ein|eine|eines|einer|einem|einen)\b",
    r"\bsind\s+(?:ein|eine|eines|einer|einem|einen)?\b",
    r"\bbezeichnet\s+(?:einen|eine|ein|den|die|das)?\b",
    r"\bwird\s+(?:als|auch\s+als)\b",
    r"\bwurde\s+(?:als|auch\s+als)\b",
    r"\bwird\s+definiert\s+als\b",
    r"\bist\s+definiert\s+als\b",
    r"\bist\s+bekannt\s+als\b",
    r"\bwird\s+beschrieben\s+als\b",
    r"\bgilt\s+als\b",
    r"\bversteht\s+man\s+(?:unter|darunter)\b",
]


def _compile(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


HEDGE_REGEXES = _compile(HEDGE_PATTERNS)
WEASEL_REGEXES = _compile(WEASEL_PATTERNS)
DEFINITION_REGEXES = _compile(DEFINITION_PATTERNS)


@lru_cache(maxsize=1)
def load_german_nlp():
    """Load the German spaCy model used for tokenization and sentences."""
    if spacy is None:
        raise ImportError("spaCy is required. Install spacy and de_core_news_sm.")
    try:
        return spacy.load(SPACY_MODEL)
    except OSError as exc:
        raise OSError(
            "German spaCy model is missing. Install it with: "
            "python -m spacy download de_core_news_sm"
        ) from exc


def _nan() -> float:
    return float("nan")


def _word_count(text: str, nlp=None) -> int:
    if not text:
        return 0
    if nlp is None:
        return len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))
    doc = nlp(text)
    return sum(1 for token in doc if token.is_alpha)


def _sentences(text: str, nlp=None) -> list[str]:
    if not text:
        return []
    if nlp is None:
        return [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 10]
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]


def regex_density(text: str, regexes: Iterable[re.Pattern[str]], n_words: int) -> float:
    """Count regex matches per 1,000 words."""
    if not text or n_words <= 0:
        return 0.0
    count = sum(len(pattern.findall(text)) for pattern in regexes)
    return round(count / n_words * 1000, 4)


def hedging_density(text: str, n_words: int | None = None, nlp=None) -> float:
    """German epistemic hedge density per 1,000 words."""
    words = n_words if n_words is not None else _word_count(text, nlp=nlp)
    return regex_density(text, HEDGE_REGEXES, words)


def weasel_density(text: str, n_words: int | None = None, nlp=None) -> float:
    """German weasel-word density per 1,000 words."""
    words = n_words if n_words is not None else _word_count(text, nlp=nlp)
    return regex_density(text, WEASEL_REGEXES, words)


def definition_density(text: str, nlp=None) -> float:
    """Share of sentences containing a German definitional construction."""
    sentences = _sentences(text, nlp=nlp)
    if not sentences:
        return 0.0
    n_definition = sum(
        1
        for sent in sentences
        if any(pattern.search(sent) for pattern in DEFINITION_REGEXES)
    )
    return round(n_definition / len(sentences), 4)


def lexical_diversity_mtld(text: str) -> float:
    """Compute MTLD, matching the English v2 notebook's lexical diversity column."""
    if not text or len(text.split()) < 50 or LexicalRichness is None:
        return _nan()
    try:
        return round(LexicalRichness(text).mtld(threshold=0.72), 4)
    except Exception:
        return _nan()


def extract_german_linguistic_features(
    article: Mapping[str, Any],
    *,
    nlp=None,
) -> dict[str, Any] | None:
    """Extract German linguistic features from one article record.

    The returned columns intentionally match the English feature matrix naming:
    ``hedging_density``, ``weasel_density``, ``def_ratio``, and ``mtld``.
    """
    text = article.get("clean_text") or article.get("text") or ""
    if isinstance(text, Mapping):
        text = text.get("clean") or text.get("raw") or ""
    if not text:
        return None

    nlp = nlp or load_german_nlp()
    n_words = _word_count(text, nlp=nlp)
    if n_words == 0:
        return None

    return {
        "title": article.get("title"),
        "label": article.get("label"),
        "label_name": article.get("label_name"),
        "topic": article.get("topic", "other"),
        "topic_specific": article.get("topic_specific"),
        "lang": article.get("lang", "de"),
        "hedging_density": hedging_density(text, n_words=n_words),
        "weasel_density": weasel_density(text, n_words=n_words),
        "def_ratio": definition_density(text, nlp=nlp),
        "mtld": lexical_diversity_mtld(text),
        "word_count": article.get("word_count", n_words),
        "age_days": article.get("age_days", 0),
    }


def build_german_feature_matrix(
    articles: Iterable[Mapping[str, Any]],
    *,
    output_csv: str | None = None,
    show_progress: bool = True,
):
    """Build a pandas feature matrix from German article records."""
    if pd is None:
        raise ImportError("pandas is required to build a feature matrix.")
    nlp = load_german_nlp()
    article_iterable = list(articles)
    if show_progress and tqdm is not None:
        article_iterable = tqdm(article_iterable, desc="Extracting German features")
    records = [
        record
        for article in article_iterable
        if (record := extract_german_linguistic_features(article, nlp=nlp))
    ]
    df = pd.DataFrame(records)
    if output_csv:
        df.to_csv(output_csv, index=False)
    return df


def load_json(path: Path) -> list[dict[str, Any]]:
    """Load a JSON list of article records."""
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def main() -> None:
    contested = load_json(CONTESTED_FILE)
    stable = load_json(STABLE_FILE)
    articles = contested + stable

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = build_german_feature_matrix(articles, output_csv=str(FEATURES_FILE))

    print(f"Contested articles : {len(contested)}")
    print(f"Stable articles    : {len(stable)}")
    print(f"Total articles     : {len(articles)}")
    print(f"Output path        : {FEATURES_FILE}")
    print(f"Dataframe shape    : {df.shape}")


if __name__ == "__main__":
    main()
