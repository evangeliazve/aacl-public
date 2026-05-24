# Table Schemas

This document describes the tabular outputs released with the repository.

The main machine-learning result workbooks are:

```text
RESULTS/hydronewsfr/hydronewsfr_results.xlsx
RESULTS/climatenewsfr/climatenewsfr_results.xlsx
```

Each workbook contains three sheets:

```text
ml_metrics_with_ablation
fold_ablation
fold_ablation_paired_tests
```

The released workbooks report the diagonal agreement settings used in the paper, where:

```text
outlier_k = toa_k = agreement_k
toa_max_neg = 0
```

For the selected ablation-significance setting, the paper uses:

```text
HYDRONEWSFR:   k = 4 -> (4, 4, 0)
CLIMATENEWSFR: k = 6 -> (6, 6, 0)
```

---

## Workbook: `hydronewsfr_results.xlsx`

Path:

```text
RESULTS/hydronewsfr/hydronewsfr_results.xlsx
```

Corpus:

```text
HYDRONEWSFR
```

Main selected setting for the ablation table:

```text
outlier_maj = 4
toa_min_pos = 4
toa_max_neg = 0
```

---

## Workbook: `climatenewsfr_results.xlsx`

Path:

```text
RESULTS/climatenewsfr/climatenewsfr_results.xlsx
```

Corpus:

```text
CLIMATENEWSFR
```

Main selected setting for the ablation table:

```text
outlier_maj = 6
toa_min_pos = 6
toa_max_neg = 0
```

---

# Sheet: `ml_metrics_with_ablation`

## Purpose

This sheet reports cross-validated classifier results across the released diagonal agreement settings.

It includes both classifier comparisons and feature-family ablations. Rows are aggregated across cross-validation folds and report means and standard deviations for precision, F1, and recall.

## Columns

| Column | Type | Description |
|---|---|---|
| `outlier_k` | integer | Minimum number of embedding models that must classify an article as a publication-time outlier. |
| `toa_k` | integer | Minimum number of embedding models that must assign the article to an anticipatory trajectory for a positive label. |
| `toa_max_neg` | integer | Maximum number of anticipatory votes allowed for a negative label. In the released diagonal setting, this is `0`. |
| `cv_n_splits` | integer | Number of cross-validation folds. Usually `5`. |
| `clf_name` | string | Classifier abbreviation. |
| `ablation` | string | Feature-family setting used for the row. |
| `n_articles` | integer | Number of retained labeled articles under the consensus setting. |
| `n_pos_articles_est` | integer | Number of positive articles, i.e. anticipatory outliers. |
| `n_neg_articles_est` | integer | Number of negative articles, i.e. non-anticipatory publication-time outliers. |
| `F1_mean` | float | Mean F1 score across cross-validation folds. |
| `F1_std` | float | Standard deviation of F1 score across folds. |
| `Precision_mean` | float | Mean precision across cross-validation folds. |
| `Precision_std` | float | Standard deviation of precision across folds. |
| `Recall_mean` | float | Mean recall across cross-validation folds. |
| `Recall_std` | float | Standard deviation of recall across folds. |

## Classifier labels

| Value | Meaning |
|---|---|
| `xgb` | XGBoost classifier. |
| `rf` | Random Forest classifier. |
| `logreg` | Logistic Regression classifier. |
| `linear_svc` | Linear Support Vector Machine classifier. |
| `dt` | Decision Tree classifier. |
| `baseline_all_pos` | Constant-positive baseline that predicts every eligible article as anticipatory. |

## Ablation labels

| Value | Meaning |
|---|---|
| `all_features` | Uses the full feature set: geometric, textual, and social predictors. |
| `no_geom` | Removes geometric predictors from the full feature set. |
| `no_social` | Removes social predictors from the full feature set. |
| `no_text` | Removes textual predictors from the full feature set. |
| `only_geom` | Uses only geometric predictors. |
| `only_social` | Uses only social predictors. |
| `only_text` | Uses only textual predictors. |
| `baseline` | Used only for the `baseline_all_pos` classifier. |

## Notes

The baseline row is reported once per agreement threshold, not once per ablation setting.

The released workbooks keep only the paper metrics:

```text
Precision
F1
Recall
```

Average precision and ROC AUC are not included in the cleaned release workbooks.

---

# Sheet: `fold_ablation`

## Purpose

This sheet reports fold-level XGBoost ablation results for the selected paper setting.

It contains one row per cross-validation fold and ablation setting. These rows are used to compute the ablation means reported in the paper and to perform the paired fold-level significance tests in `fold_ablation_paired_tests`.

## Selected settings

```text
HYDRONEWSFR:   outlier_maj = 4, toa_min_pos = 4, toa_max_neg = 0
CLIMATENEWSFR: outlier_maj = 6, toa_min_pos = 6, toa_max_neg = 0
```

## Columns

| Column | Type | Description |
|---|---|---|
| `outlier_maj` | integer | Minimum number of embedding models that must classify the article as a publication-time outlier. |
| `toa_min_pos` | integer | Minimum number of embedding models that must assign the article to an anticipatory trajectory for a positive label. |
| `toa_max_neg` | integer | Maximum number of anticipatory votes allowed for a negative label. |
| `clf_name` | string | Classifier name. In this sheet, this is `xgb`. |
| `ablation` | string | Feature-family ablation setting. |
| `fold` | integer | Cross-validation fold identifier. |
| `n_train` | integer | Number of training articles in the fold. |
| `n_test` | integer | Number of test articles in the fold. |
| `n_train_pos` | integer | Number of positive training articles in the fold. |
| `n_train_neg` | integer | Number of negative training articles in the fold. |
| `n_test_pos` | integer | Number of positive test articles in the fold. |
| `n_test_neg` | integer | Number of negative test articles in the fold. |
| `n_features` | integer | Number of article-level features used for the ablation setting. |
| `F1` | float | Fold-level F1 score. |
| `Precision` | float | Fold-level precision. |
| `Recall` | float | Fold-level recall. |

## Ablation labels

| Value | Meaning |
|---|---|
| `all_features` | Uses the full feature set: geometric, textual, and social predictors. |
| `only_geom` | Uses only geometric predictors. |
| `only_text` | Uses only textual predictors. |
| `only_social` | Uses only social predictors. |
| `no_geom` | Removes geometric predictors from the full feature set. |
| `no_social` | Removes social predictors from the full feature set. |
| `no_text` | Removes textual predictors from the full feature set. |

---

# Sheet: `fold_ablation_paired_tests`

## Purpose

This sheet reports paired fold-level significance tests for the selected XGBoost ablation setting.

The tests compare paired fold-level scores across ablation settings. The same folds are used for the reference and comparison settings, so the paired test is applied to the fold-level differences.

The sheet includes tests for:

```text
Precision
F1
Recall
```

The paper table uses significance symbols only for F1, because F1 is the primary evaluation metric. Precision and recall tests are retained in the supplementary workbook as diagnostics.

## Columns

| Column | Type | Description |
|---|---|---|
| `k` | integer | Selected consensus threshold. This corresponds to `(k, k, 0)`. |
| `clf_name` | string | Classifier name. In this sheet, this is `xgb`. |
| `metric` | string | Metric tested. One of `Precision`, `F1`, or `Recall`. |
| `reference` | string | Reference ablation setting in the paired comparison. |
| `comparison` | string | Comparison ablation setting in the paired comparison. |
| `n_folds` | integer | Number of paired folds used in the test. |
| `reference_mean` | float | Mean score of the reference setting across folds. |
| `comparison_mean` | float | Mean score of the comparison setting across folds. |
| `mean_diff_ref_minus_comp` | float | Mean paired difference, computed as `reference - comparison`. Positive values mean the reference setting performs better. |
| `std_diff` | float | Standard deviation of paired fold-level differences. |
| `cohens_dz` | float | Paired-sample effect size, computed as the mean paired difference divided by the standard deviation of paired differences. |
| `paired_t_stat` | float | Paired t-test statistic. |
| `paired_t_p` | float | Raw paired t-test p-value. |
| `diffs_by_fold` | string | Comma-separated fold-level paired differences. |
| `paired_t_q_fdr` | float | Benjamini-Hochberg corrected q-value. |
| `paired_t_sig` | string | Significance label derived from the raw p-value or corrected value, depending on the export configuration. Typical values are `***`, `**`, `*`, `ns`, or `n/a`. |

## Comparisons

The main comparisons are:

| Reference | Comparison | Purpose |
|---|---|---|
| `all_features` | `no_geom` | Tests whether removing geometry hurts performance. |
| `all_features` | `no_social` | Tests whether removing social features hurts performance. |
| `all_features` | `no_text` | Tests whether removing textual features hurts performance. |
| `all_features` | `only_geom` | Tests whether geometry alone differs from the full model. |
| `only_geom` | `only_text` | Tests whether text-only performance is lower than geometry-only performance. |
| `only_geom` | `only_social` | Tests whether social-only performance is lower than geometry-only performance. |

## Paper-symbol convention

In the paper ablation table:

| Symbol | Meaning |
|---|---|
| `†` | Significant F1 drop relative to `all_features` after Benjamini-Hochberg correction of paired fold-level t-test p-values. |
| `‡` | Significant F1 drop relative to `only_geom` after Benjamini-Hochberg correction of paired fold-level t-test p-values. |

Precision and recall values are reported in the paper table as complementary diagnostics. Their paired tests are included in this supplementary sheet but are not encoded by the paper-table symbols.

## Recommended caption wording

```latex
Ablations for XGBoost at \(\Ta\). Results are 5-fold CV means.
Symbols mark significant \(F_1\) drops after Benjamini--Hochberg correction
of paired fold-level \(t\)-test \(p\)-values at \(\alpha=0.05\):
\(^{\dagger}\) indicates a significant drop relative to all features, and
\(^{\ddagger}\) indicates a significant drop relative to geometry only.
Precision and recall are reported as complementary diagnostics.
```

---

# Metric definitions

| Metric | Meaning |
|---|---|
| `Precision` | Among articles predicted as anticipatory, the fraction that are truly anticipatory. |
| `Recall` | Among truly anticipatory articles, the fraction predicted as anticipatory. |
| `F1` | Harmonic mean of precision and recall. This is the primary metric used for ablation significance in the paper table. |

---

# Consensus-threshold terminology

| Term | Meaning |
|---|---|
| `outlier_k` / `outlier_maj` | Minimum number of embedding models that must identify the article as a publication-time outlier. |
| `toa_k` / `toa_min_pos` | Minimum number of embedding models that must assign the article to an anticipatory trajectory for a positive label. |
| `toa_max_neg` | Maximum number of anticipatory votes allowed for a negative label. |
| `(k, k, 0)` | Conservative diagonal rule where outlier eligibility and positive labels both require at least `k` votes, while negatives require zero anticipatory votes. |

---

# Recommended repository consistency checks

Before release, check that:

1. Both result workbooks contain the same three sheets:

```text
ml_metrics_with_ablation
fold_ablation
fold_ablation_paired_tests
```

2. The released metric columns are limited to:

```text
Precision
F1
Recall
```

3. `fold_ablation_paired_tests` includes rows for `Precision`, `F1`, and `Recall`.

4. Paper-table symbols are based only on F1 rows from `fold_ablation_paired_tests`.

5. Precision and recall tests remain available in the supplementary workbook as diagnostics.

6. The README and paper caption both state that the table symbols indicate significant F1 drops, not precision or recall drops.
