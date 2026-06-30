# German F2 SentiWS Threshold Robustness

## Pipeline Location

German F2 is implemented in `wikipedia_analysis_v4_DE.ipynb` as **Affective Stance Marker Density**.

The relevant notebook cell defines:

- `HYLAND_ATTITUDE_DE`: 58 German functional equivalents of Hyland attitude markers.
- `parse_sentiws(filepath, threshold=0.5)`: parser for SentiWS v2.0 files.
- `SENTIWS_HIGH`: union of positive and negative SentiWS entries where `abs(weight) >= threshold`.
- `AFFECTIVE_MARKERS_DE`: deduplicated union of `HYLAND_ATTITUDE_DE + SENTIWS_HIGH`.
- `f2_affective_density(text)`: density of `AFFECTIVE_MARKERS_DE` per 1,000 words.

The SentiWS threshold is therefore applied at lexicon-construction time, before article scoring:

```python
if abs(weight) >= threshold:
    words.append(lemma)
```

The notebook's current threshold is `0.5`, documented as `|weight| >= 0.5`.

## Robustness Sweep Status

Requested thresholds:

- `0.0`
- `0.1`
- `0.2`
- `0.3`
- `0.5`

The threshold sweep is **not computable from the current repository state** because the required SentiWS v2.0 scored lexicon files are absent.

Expected paths:

- `data/external/sentiws/SentiWS_v2.0_Positive.txt`
- `data/external/sentiws/SentiWS_v2.0_Negative.txt`

Repository checks performed:

- `data/` contains only `processed/` and `raw/`.
- No local `SentiWS` or `sentiws` files were found.
- `Archive.zip` does not contain SentiWS files.
- Existing `outputs/f2_robustness_de/f2_lexicon_audit_de.json` reports `sentiws_files_available: false`.

Because the threshold sweep depends on the numeric SentiWS weights, the marker sets for thresholds `0.0`, `0.1`, `0.2`, `0.3`, and `0.5` cannot be reconstructed without those files. Reusing the already-combined F2 scores would not allow threshold variation, and reconstructing SentiWS terms from notebook output is impossible because only aggregate counts are available.

## Output Table

The file `outputs/f2_sentiment_threshold_robustness_de.csv` records one row per requested threshold with status `not_computable_missing_sentiws`.

For context only, the table also includes the previously computed Hyland-only fallback from `outputs/f2_robustness_de/f2_robustness_summary_de.csv`:

| Variant | Direction | Contested mean | Stable mean | Wilcoxon p | Effect r | Coverage |
|---|---|---:|---:|---:|---:|---:|
| Hyland-only | Stable > contested | 0.263307 | 0.318211 | 0.595863 | 0.532762 | 284/284 |

This fallback is **not** a SentiWS threshold result, because Hyland-only does not use SentiWS scores.

## Interpretation

The current repository is sufficient to identify exactly where F2 uses SentiWS scores, but it is not sufficient to test sensitivity to SentiWS threshold choice.

No valid conclusion can be drawn about whether German F2 changes across SentiWS thresholds from the current files alone.

The only available robustness evidence is:

- Original German notebook F2 is directionally correct: contested > stable.
- Original German notebook F2 is not Holm-significant.
- Hyland-only robustness is directionally wrong: stable > contested.
- Hyland-only has weak practical contribution, despite full coverage.

This suggests that the SentiWS component may be important for the original German F2 direction, but that claim cannot be confirmed until the scored SentiWS files are restored.

## Required Next Step

To complete the requested threshold robustness analysis, place the SentiWS v2.0 files at:

- `data/external/sentiws/SentiWS_v2.0_Positive.txt`
- `data/external/sentiws/SentiWS_v2.0_Negative.txt`

Then recompute F2 for each threshold by rebuilding `AFFECTIVE_MARKERS_DE = HYLAND_ATTITUDE_DE + SentiWS(|weight| >= threshold)`, rescoring all German articles, recomputing matched-pair differences, and applying Wilcoxon plus Holm correction across the tested thresholds.
