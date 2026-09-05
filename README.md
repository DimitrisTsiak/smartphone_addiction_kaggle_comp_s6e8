# Smartphone Addiction Prediction (Playground Series s6e8)

Predicting smartphone addiction (`addicted_label`) using tabular behavioral and telemetry data from Kaggle Playground Series s6e8.

## Feature Engineering

The feature engineering pipeline (`engineer_features`) expands the raw dataset to 58 features:

- **Missingness Signals**: Total missing value counts per row and individual binary missingness flags (`isna_*`).
- **Activity Breakdown & Ratios**: Combined entertainment hours, ratios of social/gaming/work to daily screen time, productivity-to-entertainment balance, and unaccounted screen time.
- **Engagement Dynamics**: Notifications and app opens per screen hour, notifications per open, and average screen minutes per open.
- **Weekend vs. Weekday Patterns**: Weekend-to-daily ratios, screen time differences, and weighted weekly estimates.
- **Sleep & Schedule Metrics**: Sleep-to-screen ratio, estimated free time proxy, and extreme threshold flags (`is_sleep_deprived`, `high_screen_low_sleep`).
- **Constrained Imputation**: Bounding missing component values using the domain invariant `daily_screen_time_hours >= social_media + gaming + work_study`.
- **Demographics & Aggregations**: Demographic interaction encodings and group-level daily screen time statistics (mean, standard deviation, and user deviation from group mean).

## Model & Ensemble

- **Cross-Validation**: 10-Fold Stratified K-Fold cross-validation preserving target distribution.
- **XGBoost Classifier**: Hyperparameters optimized via Optuna.
- **Fold Ensemble**: The final test predictions are produced by averaging the out-of-fold models across all 10 folds:

$$\hat{y} = \frac{1}{10} \sum_{k=1}^{10} P_k(X_{\text{test}})$$
