# Appendix files

This folder contains supporting tables used during analysis.

- `ml_feature_glossary.xlsx` documents the released feature columns.
- `additional_experiments/*_toa_correlations_by_model.xlsx` contains feature/TOA correlation checks by embedding model.
- `additional_experiments/hydronewsfr_extended_results.xlsx` contains the extended HydroNewsFr robustness check across diagonal agreement thresholds 1–8.
- `selected_setting_ablation_significance_HYDRO*.xlsx` and `selected_setting_ablation_significance_CLIMATE*.xlsx` contain the paired fold-level ablation comparisons used to assign the significance markers in the ablation table.

The main reproducible supervised rerun path is described in the root README and uses `agreement_k = outlier_k = toa_k`, with `toa_max_neg = 0`.
