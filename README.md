# Wiki-Analysis

Computational Social Science project investigating linguistic and structural differences between contested and stable Wikipedia articles across English and German Wikipedia.

---

# Project Overview

This repository contains the full research pipeline for a multilingual Wikipedia analysis project developed for IS618 SMDA.

The project examines whether Wikipedia articles marked by neutrality or dispute-related maintenance signals systematically differ from stable, quality-rated articles in terms of:

- linguistic uncertainty
- definitional framing
- lexical diversity
- structural organization
- editorial behavior

The repository currently includes:

- English Wikipedia collection and analysis notebooks
- German Wikipedia collection and infrastructure scripts
- multilingual feature extraction pipelines
- matching and validation systems
- processed datasets and diagnostics

---

# Research Question

> Do articles carrying POV/neutrality dispute tags and articles with no dispute history show systematic differences in linguistic and structural features?

The project operationalizes article contestation through dispute-related maintenance templates and compares those articles against stable, quality-rated articles with no detected recent dispute signals.

---

# Methodology

The overall workflow consists of:

1. Collect contested and stable articles from Wikipedia
2. Apply quality and activity thresholds
3. Assign broad topical categories
4. Extract linguistic and structural features
5. Match contested articles with comparable stable articles
6. Validate dataset integrity and matching quality
7. Perform statistical analysis and modeling

---

# Article Definitions

## Contested Articles

Contested articles are collected from dispute-related maintenance templates.

### English
- POV templates
- Neutrality dispute templates

### German
- `Vorlage:Neutralität`

These articles are treated as operational indicators of editorial disagreement or neutrality concerns.

---

## Stable Articles

Stable articles are sampled from Wikipedia quality-rated article pools.

### English
- Good Articles

### German
- `Kategorie:Wikipedia:Exzellent`
- `Kategorie:Wikipedia:Lesenswert`

Stable articles are additionally checked to ensure that no current dispute templates are present.

---

# Topic Balancing

Articles are grouped into four broad topic categories:

- `politics_history`
- `culture`
- `geography`
- `science`

Topic balancing is enforced during collection and matching to reduce topic-driven confounds.

Specific Lift Wing subtopics are also retained for finer-grained matching diagnostics and regression controls.

---

# English Pipeline

The English pipeline is primarily notebook-based.

## Main Files

- `Wikipedia_EN_data_collection.ipynb`
- `Wikipedia_EN_Analysis_v5.ipynb`

## English Outputs

- `final_contested_en.json`
- `final_stable_en.json`
- `final_pairs_en.json`
- `final_features_en.csv`
-  final_pairs_en_strict

The English notebooks include:

- data collection
- feature engineering
- statistical testing
- logistic regression
- bootstrap confidence intervals
- coefficient analysis
- error analysis

---

# German Pipeline

The German pipeline is implemented as reusable Python modules.

## Main Files

- `src/wiki_trends/de_collection.py`
- `src/wiki_trends/features_de.py`
- `src/wiki_trends/matching.py`
- `src/wiki_trends/validate_de_dataset.py`

The German pipeline mirrors the English dataset structure while adapting:

- dispute templates
- category systems
- namespace handling
- German linguistic resources
- German NLP tooling

---

# Feature Extraction

The project extracts both linguistic and structural features.

## Linguistic Features
### Category 1 — Authorial Commitment
- F1: epistemic hedge density
- F2: affective stance marker density
- 
### Category 2 — Source Transparency
- F3: verification-evading expression density
- F4: attribution verb bias ratio
- F5: lexical presupposition density
- 
### Category 3 — Textual Organisation
- F6: contrastive transition density
- F7: lexical diversity (MTLD)


German feature extraction uses:

- German regex lexicons
- spaCy (`de_core_news_sm`)
- `lexicalrichness`

---

# Matching Strategy

The project uses nearest-neighbour matching to compare contested and stable articles with similar metadata characteristics.

## Matching Constraints

- same broad topic required
- same subtopic preferred
- one-to-one stable article usage

## Matching Variables

- word count
- article age
- lexical diversity (MTLD)

The matching pipeline outputs validated contested/stable article pairs for downstream statistical analysis.

---

# Validation and Sanity Checks

The German validation pipeline performs automated dataset diagnostics including:

- duplicate detection
- missing value checks
- empty-text detection
- article-length outlier detection
- dominant-language checks
- topic-balance diagnostics
- suspicious match inspection
- dispute-template leakage checks
- random matched-pair inspection

Validation outputs are exported to:

```text
data/processed/de_validation_report.txt
