# Appendix files

This folder contains supporting tables used during analysis.

- `ml_feature_glossary.xlsx` documents the released feature columns.
- `additional_experiments/*_toa_correlations_by_model.xlsx` contains feature/TOA correlation checks by embedding model.
- `additional_experiments/hydronewsfr_extended_results.xlsx` contains additional diagonal agreement thresholds beyond the main released 1–8 rerun.
- `additional_experiments/hydronewsfr_extended_chronological_eval_k440.xlsx` contains the chronological evaluation check for the selected HYDRONEWSFR setting.

The main reproducible supervised rerun path is described in the root README and uses `agreement_k = outlier_k = toa_k`, with `toa_max_neg = 0`.
