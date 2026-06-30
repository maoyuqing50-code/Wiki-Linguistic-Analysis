# German Symbolic Feature False-Positive Review

This qualitative review uses `outputs/german_feature_validation_samples.csv` and covers only the symbolic German feature extractors: F1, F3, F4, and F6. F5 is excluded because it is an NLI-based sentence classifier rather than a lexical detector.

No statistical tests, precision/recall/F1 calculations, regressions, or significance tests were performed.

## Review Counts

| Feature | Reviewed samples | Correct detections | Ambiguous | Possible false positives |
|---|---:|---:|---:|---:|
| F1 | 20 | 16 | 2 | 2 |
| F3 | 20 | 18 | 1 | 1 |
| F4 | 20 | 14 | 3 | 3 |
| F6 | 20 | 13 | 4 | 3 |
| Total | 80 | 61 | 10 | 9 |

## Common Patterns

- F1 mostly behaves as intended: modal subjunctive forms and markers such as `womöglich`, `vielleicht`, `wohl`, and `unklar` generally correspond to epistemic hedging.
- The main F1 ambiguity comes from polysemous `glauben` and permissive `möglich`, where the local context can express religious belief or institutional possibility rather than authorial uncertainty.
- F3 is largely plausible in this sample, especially for Konjunktiv I reported-speech forms and broad quantifiers such as `zahlreiche`, `die meisten`, and `eine Reihe von`.
- The clearest F3 extraction issue is a morphological false positive: `fahrende` was labeled as `Konjunktiv-I:fahrende`, but the sentence uses an attributive participle.
- F4 captures many intended attribution frames, including neutral reporting verbs (`sagte`, `berichtete`, `beschrieb`) and biased frames (`beschuldigte`, `kritisierte`, `unterstellt`).
- F4 also shows context-sensitive false positives where lexicon terms have non-attribution meanings, especially legal `Forderung`, explanatory `erklären`, and theological `rechtfertigt`.
- F6 reliably captures many contrastive transitions (`aber`, `jedoch`, `hingegen`, `dennoch`, `dagegen`, `sondern`), but `während` is often temporal rather than contrastive.
- Some F6 `sondern` and `aber` cases are ambiguous in additive constructions such as `nicht nur ... sondern auch` or fragmentary `aber auch`.

## Systematic Extraction Issues Observed

- The symbolic extractors correctly identify the target lexical or morphological surface forms in most reviewed cases.
- The most systematic issue is semantic polysemy: some detected forms are valid German words from the lexicon but do not instantiate the intended discourse function in context.
- F6 would benefit most from context-aware handling of temporal `während`.
- F4 would benefit from sense restrictions or local syntactic checks to distinguish attribution/reporting uses from legal, explanatory, or theological uses.
- F3 has one apparent morphology-related false positive in the sample, suggesting that Konjunktiv-I detection should be checked for participial endings.
- F1 is broadly plausible but has predictable ambiguity around `glauben` and `möglich`.

## Qualitative Assessment

Overall, the sampled symbolic detections appear linguistically plausible enough to support a Methods/Validation discussion. The review also identifies concrete limitations that should be acknowledged: the framework is intentionally lexicon-based, so it is sensitive to polysemy, sentence-fragment extraction, and discourse markers whose function depends on local syntax.
