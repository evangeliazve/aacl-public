# Table schemas

This document defines the input, intermediate, and exported tables needed to reproduce the results on a new corpus.

## 1. Raw article table

Required columns are configurable in `CONFIG/*.yaml`.

| Column role | Default column | Type | Required | Description |
|---|---:|---|---|---|
| Article identifier | `media_url` | string | yes | Stable article key. A canonical URL is recommended. |
| Publication date | `publication_date_cleaned` | datetime | yes | Article publication time. Parsed with `pandas.to_datetime`. |
| Title | `title` | string | yes | Article title. Used in the embedding text and text features. |
| Lead/body text | `description` | string | yes | Lead paragraph, description, or body text. Used in the embedding text and text features. |
| URL | `media_url` | string | recommended | Used to match social-sharing data. Can be identical to the article identifier. |

Additional columns are preserved only if a downstream custom analysis needs them.

## 2. Optional social-sharing table

| Column | Type | Required | Description |
|---|---|---:|---|
| `media_url` or configured URL column | string | yes | Article URL shared on X. |
| `user_id` or `author_id` | string | yes | User/account identifier. |
| `created_at` | datetime | recommended | Share timestamp. If present, only shares at or before publication time are used. |
| `followers_count` or `user_public_metrics_followers_count` | numeric | optional | User follower count. |
| `tweet_count` or `user_public_metrics_tweet_count` | numeric | optional | User tweet/status count. |
| `listed_count` or `user_public_metrics_listed_count` | numeric | optional | User listed count. |

If no social table is supplied, all social variables are set to zero.

## 3. Model-specific dynamic topic results (`models/<model>/results.csv`)

Produced by `01_dynamic_topic_reconstruction.py` and consumed by scripts 02 and 04.

| Column | Type | Description |
|---|---|---|
| `media_url` or configured article ID | string | Article identifier. |
| publication date column | datetime | Article publication date. |
| `snapshot_date` | datetime | Cumulative snapshot date. |
| `model` | string | Embedding model short name. |
| `cluster_id` | integer | HDBSCAN cluster label for the snapshot; `-1` is noise/outlier. |
| `topic_id` | integer | Temporally aligned topic ID; `-1` for outliers. |
| `is_outlier` | boolean | Whether the article is an outlier in that snapshot. |
| `outlier_score` | numeric | HDBSCAN GLOSH-style outlierness score. |
| `umap_0` ... `umap_19` | numeric | Reduced embedding coordinates. Number depends on `clustering.n_components`. |

## 4. Trajectory matrix (`trajectory_matrix.xlsx`)

Produced by `02_trajectory_annotation_matrix.py`.

| Column pattern | Type | Description |
|---|---|---|
| `media_url` | string | Article identifier. |
| `publication_date_cleaned` | datetime | Article publication date. |
| `outlier_<model>` | 0/1 | Whether the article was an outlier at publication time for the model. |
| `toa_<model>` | 0/1 | Whether the model assigns the article to an anticipatory outlier trajectory. |
| `trajectory_<model>` | string | Trajectory category: `TOAfirst`, `TOAlate`, `TODlate`, `Oold`, `other`, or `not_publication_outlier`. |
| `first_topic_id_<model>` | integer/string | First aligned topic joined after publication-time outlier status. |
| `first_topic_time_<model>` | datetime | Creation time of the joined topic. |
| `integration_time_<model>` | datetime | First time the article joins the topic. |
| `n_outlier_votes` | integer | Number of models identifying the article as a publication-time outlier. |
| `n_TOA_votes` | integer | Number of models assigning the article to a TOA trajectory. |
| `n_models` | integer | Number of embedding models in the matrix. |

## 5. Agreement labels (`agreement_labels.xlsx`)

Produced by `03_calculate_agreement.py`.

### Sheet: `article_labels`

Contains the trajectory matrix plus one label column per threshold.

| Column | Type | Description |
|---|---|---|
| `label_k1`, `label_k2`, ... | 0/1/blank | Consensus label at threshold `k`. `1` means TOA, `0` means confident non-TOA, blank means uncertain/excluded. |

Consensus rule:

- Eligible population: `n_outlier_votes >= k`.
- Positive: `n_TOA_votes >= k`.
- Negative: `n_TOA_votes <= toa_max_neg`; the paper setting uses `toa_max_neg = 0`.
- Otherwise: uncertain.

### Sheet: `agreement_summary`

| Column | Type | Description |
|---|---|---|
| `agreement_k` | integer | Consensus threshold. |
| `retained` | integer | Number of articles with defined labels. |
| `positive` | integer | Number of positive TOA articles. |
| `negative` | integer | Number of negative articles. |
| `coverage` | numeric | Retained share of all articles. |
| `fleiss_kappa_TOA_votes` | numeric | Fleiss' kappa over model TOA votes. |

## 6. Model-level feature table (`feature_long_model_level.csv`)

Produced by `04_create_features_long_tables.py`.

| Feature | Type | Description |
|---|---|---|
| `d1_nearest_centroid_pct` | numeric | Within-model percentile rank of distance to nearest existing topic centroid. |
| `d2_second_centroid_pct` | numeric | Within-model percentile rank of distance to second-nearest existing topic centroid. |
| `margin_d2_minus_d1_pct` | numeric | Percentile-ranked difference between second-nearest and nearest centroid distances. |
| `mahal_nearest_pct` | numeric | Percentile-ranked diagonal Mahalanobis distance to nearest plausible topic cluster. |
| `knn_mean_k20_pct` | numeric | Percentile-ranked mean distance to the 20 nearest neighbors. |
| `knn_std_k20_pct` | numeric | Percentile-ranked standard deviation of 20-nearest-neighbor distances. |
| `outlier_proto_mean_dist_pct` | numeric | Percentile-ranked mean distance to nearest recent outlier articles. |
| `outlier_score` | numeric | HDBSCAN outlierness score. |
| `has_recent_outliers` | 0/1 | Whether publication-time outlier pool is non-empty. |
| `n_recent_outliers` | integer | Number of publication-time outliers in the snapshot. |
| text/social feature columns | numeric | Article-level values copied to each model row before aggregation. |

## 7. Article-level feature table (`feature_article_level.csv`)

Produced by `04_create_features_long_tables.py`.

Geometric model-level features are aggregated by article using `_mean`, `_median`, and `_std`. Text and social features are already article-level and are carried as single values. `n_models_present` records the number of model rows available for the article.

## 8. Exported review matrix (`article-url_target_features_all_k.xlsx`)

Produced by `05_export_feature_matrix.py`. This is the main shareable supervised-learning table.

| Column | Type | Description |
|---|---|---|
| `article_url` | string | Article identifier / URL. |
| `agreement_k` | integer | Consensus threshold used for the row. |
| `label_TOA` | 0/1 | Supervised target for that threshold. |
| `n_models_present` | integer | Number of available embedding-model rows. |
| `d1_nearest_centroid_pct_mean` | numeric | Mean across models. |
| `d1_nearest_centroid_pct_median` | numeric | Median across models. |
| `d1_nearest_centroid_pct_std` | numeric | Standard deviation across models. |
| `d2_second_centroid_pct_mean` | numeric | Mean across models. |
| `d2_second_centroid_pct_median` | numeric | Median across models. |
| `d2_second_centroid_pct_std` | numeric | Standard deviation across models. |
| `margin_d2_minus_d1_pct_mean` | numeric | Mean across models. |
| `margin_d2_minus_d1_pct_median` | numeric | Median across models. |
| `margin_d2_minus_d1_pct_std` | numeric | Standard deviation across models. |
| `mahal_nearest_pct_mean` | numeric | Mean across models. |
| `mahal_nearest_pct_median` | numeric | Median across models. |
| `mahal_nearest_pct_std` | numeric | Standard deviation across models. |
| `knn_mean_k20_pct_mean` | numeric | Mean across models. |
| `knn_mean_k20_pct_median` | numeric | Median across models. |
| `knn_mean_k20_pct_std` | numeric | Standard deviation across models. |
| `knn_std_k20_pct_mean` | numeric | Mean across models. |
| `knn_std_k20_pct_median` | numeric | Median across models. |
| `knn_std_k20_pct_std` | numeric | Standard deviation across models. |
| `outlier_proto_mean_dist_pct_mean` | numeric | Mean across models. |
| `outlier_proto_mean_dist_pct_median` | numeric | Median across models. |
| `outlier_proto_mean_dist_pct_std` | numeric | Standard deviation across models. |
| `outlier_score_mean` | numeric | Mean across models. |
| `outlier_score_median` | numeric | Median across models. |
| `outlier_score_std` | numeric | Standard deviation across models. |
| `has_recent_outliers_mean` | numeric | Mean across models. |
| `has_recent_outliers_median` | numeric | Median across models. |
| `has_recent_outliers_std` | numeric | Standard deviation across models. |
| `n_recent_outliers_mean` | numeric | Mean across models. |
| `n_recent_outliers_median` | numeric | Median across models. |
| `n_recent_outliers_std` | numeric | Standard deviation across models. |
| `soc_unique_users` | numeric | Number of distinct users sharing the article URL by publication time. |
| `soc_median_user_public_metrics_followers_count` | numeric | Median follower count of sharing users. |
| `soc_median_user_public_metrics_tweet_count` | numeric | Median tweet count of sharing users. |
| `soc_median_user_public_metrics_listed_count` | numeric | Median listed count of sharing users. |
| `media_weighted_clustering` | numeric | Weighted clustering coefficient in the URL co-sharing graph. |
| `media_bridge_ratio` | numeric | Share of co-sharing edge weight connecting to other communities. |
| `media_community_size` | numeric | Size of Louvain community containing the URL node. |
| `text_subjectivity` | numeric | French TextBlob subjectivity score, or zero fallback. |
| `text_neutrality` | numeric | `1 - abs(VADER compound)`. |
| `avg_sentence_len_words` | numeric | Mean sentence length in words. |
| `avg_word_len_chars` | numeric | Mean word length in characters. |
| `total_syllables` | numeric | Approximate syllable count. |
| `avg_syllables_per_word` | numeric | Approximate syllables per word. |
| `len_chars` | numeric | Character count. |
| `len_words` | numeric | Word count. |
| `ner_total_ents` | numeric | Total named entities. |
| `ner_distinct_ents` | numeric | Distinct named entities. |
| `ner_person` | numeric | Person entity count. |
| `ner_org` | numeric | Organization entity count. |
| `ner_loc` | numeric | Location/GPE entity count. |
| `ner_misc` | numeric | Other entity count. |

## 9. ML results (`results.xlsx`)

Produced by `06_run_ml_experiments.py`. The released matrices support the diagonal agreement settings `outlier_k = toa_k = agreement_k`, with `toa_max_neg = 0`.

### Sheet: `ml_metrics_with_ablation`

| Column | Type | Description |
|---|---|---|
| `horizon` | string | Prediction horizon; default `TA`. |
| `outlier_k` | integer | Publication-time outlier consensus threshold. In the released reruns, this equals `agreement_k`. |
| `toa_k` | integer | TOA positive-vote threshold. In the released reruns, this equals `agreement_k`. |
| `toa_max_neg` | integer | Maximum TOA votes for negatives. |
| `cv_n_splits` | integer | Number of CV folds. |
| `clf_name` | string | Classifier or baseline: `xgb`, `rf`, `logreg`, `linear_svc`, `dt`, or `baseline_all_pos`. |
| `ablation` | string | Feature subset: `all_features`, `no_geom`, `no_social`, `no_text`, `only_geom`, `only_social`, or `only_text`. `baseline` for the baseline. |
| `n_articles` | integer | Number of retained labeled articles. |
| `n_pos_articles_est` | integer | Number of positives. |
| `n_neg_articles_est` | integer | Number of negatives. |
| `F1_mean`, `F1_std` | numeric | Fold mean and standard deviation. |
| `Precision_mean`, `Precision_std` | numeric | Fold mean and standard deviation. |
| `Recall_mean`, `Recall_std` | numeric | Fold mean and standard deviation. |
| `AP_mean`, `AP_std` | numeric | Average precision fold mean and standard deviation. |
| `ROC_AUC_mean`, `ROC_AUC_std` | numeric | ROC AUC fold mean and standard deviation. |

## 10. SHAP interpretation workbook

Produced by `07_shap_oof_interpretation.py`.

| Sheet | Description |
|---|---|
| `xgb_global_shap` | Global feature ranking by mean absolute SHAP value, with Spearman direction diagnostics. |
| `xgb_local_predictions` | Out-of-fold article predictions. |
| `xgb_local_shap_long` | Long table of local SHAP values for every article-feature pair. |

## 11. Recommended embedding ensemble

The paper ensemble used the following embedding models:

| Short name | Model |
|---|---|
| `camembert` | `dangvantuan/sentence-camembert-base` |
| `solon` | `OrdalieTech/Solon-embeddings-large-0.1` |
| `miniLM` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `mpnet` | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` |
| `LaBSE` | `sentence-transformers/LaBSE` |
| `e5-large` | `intfloat/multilingual-e5-large` |
| `snowflake` | `Snowflake/snowflake-arctic-embed-l-v2.0` |
| `bge-m3` | `BAAI/bge-m3` |
| `openai-small` | `openai/text-embedding-3-small` as precomputed embeddings |
| `gemini` | `gemini/embedding-001` as precomputed embeddings |
| `mistral` | `mistral/mistral-embed` as precomputed embeddings |
