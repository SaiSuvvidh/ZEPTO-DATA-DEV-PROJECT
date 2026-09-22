# Analytics Module

Full EDA + predictive modeling pipeline on the Titanic dataset: profiling, cleaning, univariate/bivariate/multivariate analysis, a stratified classification pipeline (3 models), imbalance handling comparison, hyperparameter tuning, and a regression side-task.

## Run
```bash
python _01_eda.py        # profiling, cleaning, EDA story -> titanic.csv + chart PNGs
python _02_modeling.py   # classification + regression modeling -> best_pipeline.joblib
```
`sns.load_dataset('titanic')` is called exactly once, in `_01_eda.py`; `_02_modeling.py` reads the committed `titanic.csv` offline fallback, never re-loading from the network.

## Part A — EDA

**Missing values (threshold rule applied):**
| Column | % missing | Action |
|---|---|---|
| `embarked` | 0.22% | Dropped (<5%) |
| `embark_town` | 0.22% | Dropped (<5%) |
| `age` | 19.87% | Median-imputed, grouped by pclass+sex (5–30%) |
| `deck` | 77.22% | Encoded as "Unknown" category rather than dropped (>30%) — whether a cabin was recorded likely correlates with pclass/survival, so that signal was kept |

**Univariate:** `age` has 32 IQR outliers, `fare` has 114. Fare: mean=32.10, median=14.45, mode=8.05 — **right-skewed** (mean > median > mode), driven by a small number of high-value tickets (up to ~512, the historical Cardeza family fare).

**Bivariate:** Survival rate — female 0.740 vs male 0.189; class 1: 0.626, class 2: 0.473, class 3: 0.242; female 1st-class 0.967 vs male 3rd-class 0.135. Top 2 correlations (6×6 matrix, `survived`/`pclass`/`age`/`sibsp`/`parch`/`fare`): **pclass↔fare (-0.548)** — higher-numbered (cheaper) classes correlate with lower fares, confirming the fare/class relationship as expected; **sibsp↔parch (0.415)** — siblings/spouses and parents/children counts are correlated, suggesting both largely capture the same "traveling with family" signal.

**Multivariate story (4 charts, `story_1`–`story_4` PNGs):** Survival is driven primarily by sex, secondarily by class; age doesn't cleanly separate survival once class is controlled for, except children skew toward survival in every class; fare (as a class/cabin proxy) shows a denser high-fare band among survivors; survival rises for small families (1–3) vs solo travelers, then drops for large families (4+), suggesting a coordination "sweet spot."

**Z-score check:** `age`/`fare` standardized to confirm mean≈0, std≈1 (`zscore_before_after.png`) — exploratory only, not fed into modeling.

## Part B — Modeling

**Split:** Stratified 80/20 on `survived`, justified by the dataset's class imbalance (61.6%/38.4%) — an unstratified split risks a test set with a meaningfully different survival ratio purely by chance.

**Preprocessing:** `ColumnTransformer` (median-impute+scale numeric; most-frequent-impute+one-hot categorical) wrapped in a `Pipeline`, fit on training data only.

**Classifier comparison:**
| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.804 | 0.793 | 0.667 | 0.724 | 0.844 |
| Decision Tree | 0.765 | 0.755 | 0.580 | 0.656 | 0.797 |
| Random Forest | 0.816 | 0.800 | 0.696 | 0.744 | 0.827 |

**Imbalance handling (Random Forest):**
| Variant | Precision | Recall | F1 |
|---|---|---|---|
| Baseline | 0.800 | 0.696 | 0.744 |
| class_weight='balanced' | 0.754 | 0.710 | 0.731 |
| SMOTE (train-fold only) | 0.754 | 0.710 | 0.731 |

Baseline had the best F1; with only mild imbalance here, rebalancing traded precision for recall without a net F1 gain — the baseline was the better choice for this dataset.

**Hyperparameter tuning (GridSearchCV, Random Forest):** Best params `{max_depth: 10, max_features: 'sqrt', n_estimators: 300}`, CV F1 = 0.737, **OOB score = 0.823**.

**Regression side-task (predicting `fare`):** MAE=20.809, RMSE=30.473, R²=0.400, Adjusted R²=0.375. Residual plot shows a clear funnel shape — **heteroscedasticity confirmed** — error grows with predicted fare, consistent with fare's right-skewed distribution from Part A.

**Final recommendation:** Random Forest is the recommended deployment model — best accuracy (0.816), precision (0.800), and F1 (0.744), with tuned OOB score 0.823. Logistic Regression has a marginally higher AUC (0.844 vs 0.827) but a weaker precision/recall balance. The regression model is exploratory (R²=0.40, heteroscedastic errors), not a deployment candidate.

**Saved artifact:** `best_pipeline.joblib` — the complete tuned Random Forest pipeline (preprocessing + classifier), verified reloadable and correct on raw input via `joblib.load`.