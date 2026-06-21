# Cross-Lingual Feature Comparison

Repository state audited on branch `pipeline-infrastructure`.

Primary sources inspected:

- `wikipedia_analysis_v4_EN.ipynb`
- `wikipedia_analysis_v4_DE.ipynb`
- `outputs/f1_robustness_de/`
- `outputs/f2_robustness_de/`
- `outputs/f3_robustness_de/`
- `outputs/f4_robustness_de/`
- `outputs/f5_robustness_de/`
- `outputs/de_original_vs_improved/`

## Final English Feature Set

The latest English V4 analysis uses the original seven linguistic features:

| Feature | English final implementation |
|---|---|
| F1 | Epistemic hedge density |
| F2 | Affective stance marker density |
| F3 | Weasel word density |
| F4 | Attribution bias ratio |
| F5 | Lexical presupposition density |
| F6 | Contrastive transition density |
| F7 | Lexical diversity, MTLD |

English V4 summary:

- Matched pairs: 143
- Direction correct: 6/7
- Holm-significant Wilcoxon features: 5/7
- Logistic regression, all seven features plus topic controls: macro F1 0.715 ± 0.046, accuracy 0.717, baseline 0.500

English feature behavior:

| Feature | Direction | Holm p | Effect r | Logistic coefficient | Abs. rank |
|---|---:|---:|---:|---:|---:|
| F1 | Wrong, stable > contested | 0.2086 | 0.105 | +0.020 | 7 |
| F2 | Correct, contested > stable | 0.0124 | 0.477 | -0.419 | 5 |
| F3 | Correct, contested > stable | 0.0000 | 0.394 | -0.866 | 2 |
| F4 | Correct, contested > stable | 0.0010 | 0.612 | -1.029 | 1 |
| F5 | Correct, contested > stable | 0.0721 | 0.186 | -0.316 | 6 |
| F6 | Correct, stable > contested | 0.0124 | 0.245 | +0.742 | 3 |
| F7 | Correct, contested > stable | 0.0022 | 0.294 | -0.486 | 4 |

The strongest English features are F4, F3, F6, and F7 by absolute logistic coefficient. F2 is also statistically significant. F1 is directionally wrong, and F5 is only marginal by Holm-corrected Wilcoxon and unstable in the bootstrap CI.

## Final German Feature Set

The original German V4 notebook used the direct seven-feature German adaptation:

| Feature | Original German V4 implementation |
|---|---|
| F1 | Original German epistemic hedge density |
| F2 | Original German affective stance marker density |
| F3 | Original German weasel word density |
| F4 | Original German attribution bias ratio |
| F5 | Original German lexical presupposition density |
| F6 | Original German contrastive transition density |
| F7 | Original German MTLD |

The latest improved German specification is:

| Feature | Improved German specification |
|---|---|
| F1 | Original F1 unchanged |
| F2 | Original F2 unchanged |
| F3 | No-generic-quantifiers robustness variant |
| F4 | Expanded smoothed attribution-bias ratio |
| F5 | Revised robustness version with German labels/hypothesis, threshold 0.75, and margin rule |
| F6 | Original F6 unchanged |
| F7 | Original F7 unchanged |

Important reproducibility note: `final_features_de.csv` does not contain the original SentiWS-based F2 column. The saved German original-vs-improved model in `outputs/de_original_vs_improved/` therefore omits F2 from both models. This preserves the F3/F4/F5 robustness comparison delta, because F2 is unchanged, but it is not an exact F2-inclusive reproduction of the original seven-feature German notebook model.

German original V4 notebook summary:

- Matched pairs: 142
- Direction correct: 5/7
- Holm-significant Wilcoxon features: 0/7
- Logistic regression, all seven features plus topic controls: macro F1 0.584 ± 0.059, accuracy 0.587, baseline 0.500

German available original-vs-improved comparison:

| Metric | Original available model | Improved available model | Delta |
|---|---:|---:|---:|
| Rows used | 150 | 282 | +132 |
| Accuracy | 0.547 | 0.624 | +0.077 |
| Macro F1 | 0.546 | 0.624 | +0.078 |
| ROC-AUC | 0.627 | 0.659 | +0.032 |

Common-sample diagnostic, same 150 rows:

| Metric | Original | Improved | Delta |
|---|---:|---:|---:|
| Accuracy | 0.547 | 0.607 | +0.060 |
| Macro F1 | 0.546 | 0.604 | +0.058 |
| ROC-AUC | 0.627 | 0.636 | +0.009 |

The improvement is therefore partly due to recovering rows lost to original F4 NaNs, but the common-sample result shows that the feature revisions also improve classification on the same article set.

## German Robustness Analyses F1-F5

### F1: Epistemic Hedge Density

F1 remains directionally wrong in German. Stable articles score higher than contested articles.

| Variant | Direction | Diff % | Wilcoxon p | Effect r | Macro F1 contribution |
|---|---|---:|---:|---:|---:|
| Original F1 recomputed | Stable > contested | -22.3% | 0.2876 | 0.187 | +0.0845 |
| Lexical-only | Stable > contested | -30.1% | 0.1791 | 0.259 | +0.0585 |
| Konjunktiv-II-only | Contested > stable | +13.5% | 0.3749 | 0.590 | +0.0103 |
| Remove generic verbs | Stable > contested | -26.4% | 0.1415 | 0.238 | +0.0693 |
| Strong uncertainty only | Stable > contested | -55.5% | 0.0010 | 0.618 | +0.1282 |

Interpretation: F1 appears to measure encyclopedic caution or cautious sourcing rather than contestation. The Konjunktiv-II-only variant recovers the expected direction but has weak model contribution. F1 should remain unchanged for comparability, but it should be interpreted as a negative finding.

### F2: Affective Stance Marker Density

The F2 robustness analysis was limited because SentiWS files are not vendored in the repository. Only Hyland-only could be computed.

| Variant | Direction | Diff % | Wilcoxon p | Effect r | Macro F1 contribution |
|---|---|---:|---:|---:|---:|
| Hyland-only | Stable > contested | -17.3% | 0.5959 | 0.533 | +0.0164 |

Interpretation: the original combined German F2 is directionally correct in the notebook, but not significant. Hyland-only is directionally wrong and weak. No final replacement is justified without the SentiWS files.

### F3: Weasel Word Density

F3 improves when generic quantifiers are removed.

| Variant | Direction | Diff % | Wilcoxon p | Effect r | Macro F1 contribution |
|---|---|---:|---:|---:|---:|
| Original F3 recomputed | Contested > stable | +18.9% | 0.0148 | 0.204 | +0.1077 |
| Remove generic quantifiers | Contested > stable | +31.9% | 0.0243 | 0.249 | +0.1068 |
| Attribution + evaluative + Konjunktiv-I | Contested > stable | +43.3% | 0.0689 | 0.293 | +0.1185 |
| Konjunktiv-I only | Contested > stable | +45.8% | 0.0596 | 0.396 | +0.1075 |

Interpretation: generic quantifiers add noise. The no-generic-quantifiers variant is a reasonable improved F3 because it increases the percent difference while preserving direction and comparable model contribution.

### F4: Attribution Bias

F4 is the most successful German robustness target. Original F4 has strong direction but severe missingness due to the minimum-count rule.

| Variant | Direction | Diff % | NaN rate | Wilcoxon p | Effect r | Macro F1 contribution |
|---|---|---:|---:|---:|---:|---:|
| Current ratio, min total >=3 | Contested > stable | +105.8% | 47.2% | 0.0487 | 0.683 | +0.0093 |
| Current ratio, min total >=2 | Contested > stable | +105.3% | 34.5% | 0.0075 | 0.713 | +0.0296 |
| Current biased attribution density | Contested > stable | +171.6% | 0.0% | 0.0006 | 0.800 | +0.1037 |
| Expanded smoothed ratio | Contested > stable | +23.1% | 0.0% | 0.0032 | 0.348 | +0.1393 |
| Expanded biased attribution density | Contested > stable | +136.2% | 0.0% | 0.0000 | 0.597 | +0.1711 |

Interpretation: expanded smoothed ratio is the best final robustness replacement because it preserves the original ratio concept, removes NaNs, and improves model contribution. Expanded biased attribution density is stronger predictively, but it changes the theoretical construct more substantially.

### F5: Lexical Presupposition Density

Original German F5 is saturated near 1.0 and directionally wrong.

| Variant | Direction | Diff % | Saturation >0.90 | Wilcoxon p | Effect r | Macro F1 contribution |
|---|---|---:|---:|---:|---:|---:|
| Original F5 reproduced | Stable > contested | -0.2% | 99.3% | 0.0214 | 0.618 | +0.0407 |
| Revised F5 | Stable > contested | -2.7% | 57.2% | 0.0107 | 0.225 | +0.1142 |

Interpretation: the revised German hypothesis/labels/threshold/margin reduce saturation and improve model contribution, but the feature remains directionally wrong. This is best treated as a robustness diagnostic rather than a confirmatory final result.

## Cross-Lingual Generalization

### Features That Generalize Best

F4 Attribution Bias is the strongest cross-lingual feature. It is English-significant with the largest English effect and largest English logistic coefficient. In German, the theoretical signal is present but the original implementation is sparse; the expanded smoothed ratio makes the feature usable and statistically stronger.

F3 Weasel Words generalizes directionally. English is strong and significant. German is weaker, but removing generic quantifiers improves the percent difference and maintains useful classification contribution.

F6 Contrastive Transitions generalizes directionally. Stable articles have higher transition density in both languages. English is significant; German is weaker in Wilcoxon tests but becomes the top coefficient in the improved available German model.

F7 Lexical Diversity generalizes directionally, with contested articles showing higher MTLD in both languages. English is significant; German is weak and non-significant.

### Features That Appear Language-Specific or Fragile

F1 Epistemic Hedges does not support the hypothesized contested > stable direction in either language. It likely captures cautious encyclopedic style rather than contestation.

F2 Affective Stance is strong in English but inconclusive in German. The German SentiWS dependency is not currently reproducible from the repository, and Hyland-only performs poorly.

F5 Lexical Presupposition is fragile. English is directionally correct but only marginal and unstable. German is directionally wrong, and the original German implementation is saturated. The revised German F5 improves saturation and model contribution but not theoretical direction.

## German Robustness Modifications That Improved Performance

The most successful German modification is F4 expanded smoothed ratio:

- Removes the original 47% NaN problem.
- Preserves the attribution-ratio concept.
- Produces Holm-significant Wilcoxon signal in the improved comparison.
- Has the strongest changed-feature logistic contribution.

F3 no-generic-quantifiers is also useful:

- Improves the contested/stable percent difference from +18.9% to +31.9% in the recomputed robustness audit.
- Retains the expected direction.
- Keeps model contribution close to the original F3.

F5 revised robustness version is useful diagnostically:

- Reduces saturation from 99.3% to 57.2%.
- Improves macro-F1 contribution from +0.0407 to +0.1142.
- Still remains directionally wrong, so it should not be treated as a clean theoretical improvement.

F1 variants do not justify replacement:

- Most variants remain directionally wrong.
- Konjunktiv-II-only recovers the expected direction but has weak contribution.

F2 variants do not justify replacement:

- Full SentiWS comparison cannot be reproduced from available files.
- Hyland-only is directionally wrong and weak.

## Results and Discussion Recommendations

Findings suitable for Results:

- English V4 provides strong support for the multi-layer feature framework: 6/7 features in the expected direction, 5/7 Holm-significant, macro F1 0.715.
- German original V4 is weaker: 5/7 expected directions, 0/7 Holm-significant, macro F1 0.584.
- Improved German robustness variants raise available-model macro F1 from 0.546 to 0.624 and accuracy from 0.547 to 0.624.
- F4 is the most robust cross-lingual feature after German smoothing and lexicon expansion.
- F3 and F6 show meaningful direction-level generalization.

Findings suitable for Discussion:

- German results are weaker partly because literal feature transfer from English introduces sparsity, saturation, and lexicon mismatch.
- F4 shows that theoretically constrained language-specific adaptation can improve German performance without abandoning the original construct.
- F3 shows that broad vague quantifiers are noisy in German and should be controlled.
- F5 demonstrates the risk of multilingual NLI features: changing labels/hypotheses and thresholds can reduce saturation but may not recover theoretical direction.
- F1 suggests hedging is not a universal marker of contestation in encyclopedic writing; it may instead mark careful, stable, high-quality exposition.
- F2 requires better reproducibility before a strong German conclusion can be drawn, because the SentiWS source files are absent.

Overall conclusion: the feature framework generalizes partially. Attribution bias, weasel-word density, contrastive transitions, and lexical diversity show the clearest cross-lingual continuity. Affective stance and lexical presupposition are more language- and resource-sensitive. Epistemic hedging is consistently problematic and should be discussed as an important null or reversal.
