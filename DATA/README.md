# DATA

This folder contains shareable article-level workbooks used for review.

The raw article text, raw collection files, and raw social-media traces are not redistributed here because they may be subject to publisher, API, or platform restrictions.

## Files

- `hydronewsfr_article-url_target_features_all_k.xlsx`: article URL, agreement threshold, target label, and final model features for HYDRONEWSFR.
- `climatenewsfr_article-url_target_features_all_k.xlsx`: article URL, agreement threshold, target label, and final model features for CLIMATENEWSFR.

For full reproduction on a new corpus, provide local article and optional social-sharing files following the schemas in `DOCS/TABLE_SCHEMAS.md`. The default example configuration expects private local files under `DATA/private/`.
