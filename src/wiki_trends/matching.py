"""Match contested German Wikipedia articles to comparable stable articles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - progress bar is optional
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_FILE = PROCESSED_DATA_DIR / "final_features_de.csv"
PAIRS_FILE = PROCESSED_DATA_DIR / "final_pairs_de.csv"

MATCH_FEATURES = ["word_count", "age_days", "mtld"]
FEATURE_WEIGHTS = {
    "word_count": 1.0,
    "age_days": 1.0,
    "mtld": 1.0,
}


@dataclass(frozen=True)
class MatchDiagnostics:
    total_contested: int
    total_stable: int
    matched: int
    unmatched: int
    exact_subtopic_matches: int


def load_feature_matrix(path: Path = FEATURES_FILE) -> pd.DataFrame:
    """Load and validate the German feature matrix."""
    if not path.exists():
        raise FileNotFoundError(f"Missing feature CSV: {path}")

    df = pd.read_csv(path)
    required = {
        "title",
        "label",
        "topic",
        "topic_specific",
        "word_count",
        "age_days",
        "mtld",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Feature CSV is missing required columns: {missing}")

    df = df.copy()
    df["topic"] = df["topic"].fillna("other")
    df["topic_specific"] = df["topic_specific"].fillna("")
    for column in MATCH_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def split_by_label(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a feature matrix into contested and stable rows."""
    contested = df[df["label"] == 0].copy()
    stable = df[df["label"] == 1].copy()
    if contested.empty:
        raise ValueError("No contested articles found with label 0.")
    if stable.empty:
        raise ValueError("No stable articles found with label 1.")
    return contested, stable


def _feature_scales(df: pd.DataFrame) -> dict[str, float]:
    """Compute robust scales for weighted distance normalization."""
    scales: dict[str, float] = {}
    for column in MATCH_FEATURES:
        values = df[column].dropna()
        if values.empty:
            scales[column] = 1.0
            continue
        iqr = values.quantile(0.75) - values.quantile(0.25)
        std = values.std()
        scale = iqr if iqr and iqr > 0 else std
        scales[column] = float(scale) if scale and scale > 0 else 1.0
    return scales


def weighted_distance(
    contested_row: pd.Series,
    stable_row: pd.Series,
    scales: dict[str, float],
    weights: dict[str, float] = FEATURE_WEIGHTS,
) -> float:
    """Compute weighted normalized distance over matching features."""
    total = 0.0
    used = 0
    for column in MATCH_FEATURES:
        left = contested_row[column]
        right = stable_row[column]
        if pd.isna(left) or pd.isna(right):
            continue
        diff = (float(left) - float(right)) / scales[column]
        total += weights[column] * diff * diff
        used += 1
    if used == 0:
        return float("inf")
    return float(np.sqrt(total / used))


def _candidate_pools(
    contested_row: pd.Series,
    stable_available: pd.DataFrame,
) -> list[pd.DataFrame]:
    """Return prioritized same-topic pools, preferring exact subtopic matches."""
    same_topic = stable_available[stable_available["topic"] == contested_row["topic"]]
    if same_topic.empty:
        return []

    contested_subtopic = contested_row.get("topic_specific", "")
    same_subtopic = same_topic[same_topic["topic_specific"] == contested_subtopic]
    other_subtopic = same_topic[same_topic["topic_specific"] != contested_subtopic]

    pools = []
    if not same_subtopic.empty:
        pools.append(same_subtopic)
    if not other_subtopic.empty:
        pools.append(other_subtopic)
    return pools


def match_articles(
    contested: pd.DataFrame,
    stable: pd.DataFrame,
    *,
    scales: dict[str, float],
) -> tuple[pd.DataFrame, MatchDiagnostics, list[dict[str, Any]]]:
    """Greedily match each contested article to one stable article."""
    used_stable_indices: set[Any] = set()
    pair_records: list[dict[str, Any]] = []
    unmatched_records: list[dict[str, Any]] = []

    iterable = contested.iterrows()
    if tqdm is not None:
        iterable = tqdm(
            list(iterable),
            total=len(contested),
            desc="Matching contested articles",
        )

    for _, contested_row in iterable:
        stable_available = stable.drop(index=list(used_stable_indices), errors="ignore")
        pools = _candidate_pools(contested_row, stable_available)

        best_index = None
        best_row = None
        best_distance = float("inf")

        for pool in pools:
            for stable_index, stable_row in pool.iterrows():
                distance = weighted_distance(contested_row, stable_row, scales)
                if distance < best_distance:
                    best_index = stable_index
                    best_row = stable_row
                    best_distance = distance
            if best_row is not None:
                break

        if best_row is None or best_index is None:
            unmatched_records.append(
                {
                    "contested_title": contested_row["title"],
                    "topic": contested_row["topic"],
                    "contested_subtopic": contested_row.get("topic_specific", ""),
                    "reason": "no available stable article with same topic",
                }
            )
            continue

        used_stable_indices.add(best_index)
        contested_subtopic = contested_row.get("topic_specific", "")
        stable_subtopic = best_row.get("topic_specific", "")
        pair_records.append(
            {
                "contested_title": contested_row["title"],
                "stable_title": best_row["title"],
                "topic": contested_row["topic"],
                "contested_subtopic": contested_subtopic,
                "stable_subtopic": stable_subtopic,
                "word_count_diff": abs(
                    float(contested_row["word_count"]) - float(best_row["word_count"])
                ),
                "age_days_diff": abs(
                    float(contested_row["age_days"]) - float(best_row["age_days"])
                ),
                "mtld_diff": abs(float(contested_row["mtld"]) - float(best_row["mtld"]))
                if not pd.isna(contested_row["mtld"]) and not pd.isna(best_row["mtld"])
                else np.nan,
                "exact_subtopic_match": contested_subtopic == stable_subtopic,
                "match_distance": best_distance,
            }
        )

    pairs = pd.DataFrame(pair_records)
    diagnostics = MatchDiagnostics(
        total_contested=len(contested),
        total_stable=len(stable),
        matched=len(pairs),
        unmatched=len(unmatched_records),
        exact_subtopic_matches=int(pairs["exact_subtopic_match"].sum())
        if not pairs.empty
        else 0,
    )
    return pairs, diagnostics, unmatched_records


def print_topic_distribution(
    contested: pd.DataFrame,
    stable: pd.DataFrame,
    pairs: pd.DataFrame,
) -> None:
    """Print topic counts for inputs and matched output."""
    print("\nTopic distribution")
    topics = sorted(set(contested["topic"]) | set(stable["topic"]))
    for topic in topics:
        contested_count = int((contested["topic"] == topic).sum())
        stable_count = int((stable["topic"] == topic).sum())
        matched_count = int((pairs["topic"] == topic).sum()) if not pairs.empty else 0
        print(
            f"  {topic:<20} contested={contested_count:>3} "
            f"stable={stable_count:>3} matched={matched_count:>3}"
        )


def print_summary(
    pairs: pd.DataFrame,
    diagnostics: MatchDiagnostics,
    unmatched: list[dict[str, Any]],
) -> None:
    """Print summary statistics and matching diagnostics."""
    print("\nMatching summary")
    print(f"  Contested articles       : {diagnostics.total_contested}")
    print(f"  Stable articles          : {diagnostics.total_stable}")
    print(f"  Matched pairs            : {diagnostics.matched}")
    print(f"  Unmatched contested      : {diagnostics.unmatched}")
    print(f"  Exact subtopic matches   : {diagnostics.exact_subtopic_matches}")

    if diagnostics.matched:
        exact_rate = diagnostics.exact_subtopic_matches / diagnostics.matched * 100
        print(f"  Exact subtopic match rate: {exact_rate:.1f}%")

    if not pairs.empty:
        print("\nDistance diagnostics")
        for column in ["word_count_diff", "age_days_diff", "mtld_diff", "match_distance"]:
            values = pairs[column].dropna()
            if values.empty:
                print(f"  {column:<18}: no valid values")
                continue
            print(
                f"  {column:<18}: "
                f"mean={values.mean():.3f} "
                f"median={values.median():.3f} "
                f"max={values.max():.3f}"
            )

    if unmatched:
        print("\nUnmatched diagnostics")
        reasons = pd.DataFrame(unmatched)["reason"].value_counts()
        for reason, count in reasons.items():
            print(f"  {reason}: {count}")
        print("  First unmatched articles:")
        for record in unmatched[:10]:
            print(f"    - {record['contested_title']} [{record['topic']}]")


def run_matching(
    *,
    input_csv: Path = FEATURES_FILE,
    output_csv: Path = PAIRS_FILE,
) -> pd.DataFrame:
    """Load German features, match articles, save pair CSV, and report diagnostics."""
    df = load_feature_matrix(input_csv)
    contested, stable = split_by_label(df)
    scales = _feature_scales(df)

    print(f"Loaded feature matrix: {input_csv}")
    print(f"  Shape: {df.shape}")
    print(f"  Matching scales: {scales}")

    pairs, diagnostics, unmatched = match_articles(contested, stable, scales=scales)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_columns = [
        "contested_title",
        "stable_title",
        "topic",
        "contested_subtopic",
        "stable_subtopic",
        "word_count_diff",
        "age_days_diff",
        "mtld_diff",
        "exact_subtopic_match",
    ]
    pairs[output_columns].to_csv(output_csv, index=False)

    print_topic_distribution(contested, stable, pairs)
    print_summary(pairs, diagnostics, unmatched)
    print(f"\nSaved matched pairs: {output_csv}")
    print(f"Output shape       : {pairs[output_columns].shape}")
    return pairs[output_columns]


def main() -> None:
    run_matching()


if __name__ == "__main__":
    main()
