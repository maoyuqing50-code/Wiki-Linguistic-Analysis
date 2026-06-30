# German Feature Extraction Validation Samples

Purpose: qualitative validation evidence for Methods > Validation. No significance tests, p-values, or regressions were computed.

Final implementation choices used for sampling:

- F1: original German hedge lexicon plus Konjunktiv-II modal morphology.
- F3: no-generic-quantifiers robustness variant.
- F4: expanded attribution lexicon used by the final improved German model.
- F5: revised German NLI robustness rule (`einseitig > 0.75` and margin `> 0.15`).
- F6: original German contrastive-transition lexicon.

## Detection Counts

| Feature | Total detected instances | Sampled instances | Plausibility note |
|---|---:|---:|---|
| F1 | 994 | 20 | Yes: sampled matches are lexical hedge terms or Konjunktiv-II modal forms. |
| F3 | 1244 | 20 | Mostly yes: sampled matches are vague/evaluative terms or Konjunktiv-I markers; generic quantifiers are excluded. |
| F4 | 1699 | 20 | Yes: sampled matches are attribution verbs/nominalisations from the expanded final lexicon. |
| F5 | 25640 | 20 | Requires human review: positives follow the final German NLI rule, but the matched phrase is a model score rather than a lexical span. |
| F6 | 1257 | 20 | Yes: sampled matches are German adversative/contrastive connectors. |

## Automatic Issue Checks

- No obvious automatic implementation issues were detected in the sampled extraction pass.
- F5 sentence-level validation processed 32 randomly ordered classifiable sentences to obtain sampled positives.
- F5 total detected instances are reconstructed from the saved full-dataset article scores as `round(f5_revised * n_classifiable_sentences)` because sentence-level decisions were not persisted.
- F5 validation does not identify a lexical span; the `matched_phrase` column records the model decision score and margin.

## Qualitative Assessment

The symbolic feature samples are linguistically plausible under their implemented definitions: F1 finds epistemic uncertainty markers, F3 finds vague/evaluative or indirect-speech markers, F4 finds reporting/attribution items, and F6 finds contrastive discourse connectors. F5 samples are plausible only as model-level sentence detections and should be reviewed manually because the NLI feature does not expose the exact lexical item responsible for the decision.
