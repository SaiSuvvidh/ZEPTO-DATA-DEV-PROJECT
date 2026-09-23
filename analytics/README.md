# Analytics Module

Full EDA + predictive modeling pipeline on the Titanic dataset — profiling, cleaning, univariate/bivariate/multivariate analysis, a stratified classification pipeline across 3 models, an imbalance-handling comparison, hyperparameter tuning, and a regression side-task.

## Run
```bash
python _01_eda.py        # profiling, cleaning, EDA story -> titanic.csv + chart PNGs
python _02_modeling.py   # classification + regression modeling -> best_pipeline.joblib
```
`sns.load_dataset('titanic')` only gets called once, inside `_01_eda.py`, and I saved the result to `titanic.csv` immediately. `_02_modeling.py` reads that same CSV rather than touching the network again.

## Part A — EDA

This module leaned less on debugging and more on the string of written justifications the task kept asking for — missing-value strategy, skew direction, correlation reads, imbalance strategy, deployment pick — each needing its own reasoning rather than just a number.

**Missing values.** I applied the threshold rule as given, column by column:

| Column | % missing | Action taken |
|---|---|---|
| `embarked` | 0.22% | Dropped those rows (<5%) |
| `embark_town` | 0.22% | Dropped (same rows, <5%) |
| `age` | 19.87% | Median-imputed, grouped by pclass+sex (5–30% band) |
| `deck` | 77.22% | Encoded as its own "Unknown" category rather than dropped (>30%) |

The threshold rule covered everything under 30% cleanly, but `deck`'s 77% needed its own separate call — past that point, the task asked me to either drop the column or encode the gaps as their own category, and justify the pick. I kept it as "Unknown" rather than dropping it, since whether a passenger's cabin got logged at all is plausibly tied to class and fare — wealthier passengers in better cabins were probably more likely to have that recorded — so dropping the column felt like it would throw away a real signal along with the noise.

**Univariate.** `age` came out with 32 IQR outliers, `fare` with 114. For `fare`: mean=32.10, median=14.45, mode=8.05 — that ordering (mean > median > mode) told me it's right-skewed, which made sense once I looked at the actual histogram: a handful of very expensive tickets (up to ~512, which I now know is the historical Cardeza family fare) were dragging the mean upward while most tickets clustered cheap.

**Bivariate.** Breaking survival down by sex gave me 0.740 for women vs 0.189 for men — a huge gap. By class: 0.626 / 0.473 / 0.242 for 1st/2nd/3rd. Combining both, the extremes were striking — 0.967 survival for 1st-class women vs 0.135 for 3rd-class men. For the 6×6 correlation matrix, the two strongest pairs I found were `pclass`↔`fare` at -0.548 (makes sense — pclass counts down from 1 as the best class, so a negative correlation with fare is exactly what you'd expect: better class, higher fare) and `sibsp`↔`parch` at 0.415, which reads to me like both are largely picking up the same underlying "traveling with family" signal rather than being independent measurements.

**Multivariate — the data story.** I built four charts to walk through what was actually driving survival:

*Chart 1 — Survival rate by class and sex (`story_1_survival_by_class_sex.png`).* This one made it clear that sex is the dominant factor, by a wide margin — women survived at far higher rates than men in every single class. Class still mattered as a secondary factor, since survival drops from 1st to 3rd within each sex, but it's clearly second in line behind sex. This lines up with the "women and children first" evacuation policy, with class likely acting through practical things like cabin location and proximity to lifeboats.

*Chart 2 — Age distribution by class and survival (`story_2_age_by_class_survival.png`).* Once I controlled for class, age on its own didn't separate survivors from non-survivors very cleanly — the medians look pretty similar. The one place age did visibly matter was at the young end: children showed a higher survival skew in every class, which fits the "children first" half of the evacuation policy.

*Chart 3 — Fare vs age, colored by survival (`story_3_fare_vs_age_survival.png`).* Survivors were noticeably denser at higher fare values. There's no clean age band separating survivors from non-survivors, but there is a fairly clear fare threshold above which survival becomes more common — reinforcing that fare (as a stand-in for class and cabin location) mattered more than age.

*Chart 4 — Survival rate by family size (`story_4_survival_by_family_size.png`).* This was the one I found most interesting. Survival rises from traveling completely alone up through small families (1–3 people), then drops off sharply for large families (4+). My read on this is a coordination "sweet spot": solo travelers may have gotten less help getting to a lifeboat, while very large families likely struggled to stay together and evacuate as a unit in the chaos — small families could coordinate quickly without being unwieldy.

**Z-score check.** I standardized `age` and `fare` and confirmed the result lands at mean≈0, std≈1 (see `zscore_before_after.png`) — this was purely an exploratory sanity check, it doesn't feed into the modeling pipeline below, which does its own separate train-only scaling.

## Part B — Modeling

**Split.** Stratified 80/20 on `survived`. I stratified deliberately because the class balance is 61.6%/38.4%, not 50/50 — an unstratified split risks landing on a test set with a meaningfully different survival ratio just by chance, which would make my evaluation metrics noisier than they need to be.

**Preprocessing.** I built this as a `ColumnTransformer` (median-impute + scale for numeric columns, most-frequent-impute + one-hot for categorical) wrapped in a `Pipeline`, fit only on the training split. Doing it this way, rather than by hand, meant one couldn't accidentally leak test-set information into training even if one wasn't paying close attention.

**Classifier comparison:**

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.804 | 0.793 | 0.667 | 0.724 | 0.844 |
| Decision Tree | 0.765 | 0.755 | 0.580 | 0.656 | 0.797 |
| Random Forest | 0.816 | 0.800 | 0.696 | 0.744 | 0.827 |

Random Forest came out ahead on accuracy, precision, recall, and F1; Logistic Regression edged it slightly on AUC despite otherwise trailing.

**Imbalance handling (tested on Random Forest):**

| Variant | Precision | Recall | F1 |
|---|---|---|---|
| Baseline | 0.800 | 0.696 | 0.744 |
| class_weight='balanced' | 0.754 | 0.710 | 0.731 |
| SMOTE (train-fold only) | 0.754 | 0.710 | 0.731 |

Both rebalancing techniques traded some precision for a bit more recall, but neither beat the baseline on F1. Given the imbalance here is mild (61.6/38.4, not severe), I wasn't expecting a big swing either way — this confirmed that for this particular dataset, the baseline was the better call.

**Hyperparameter tuning.** GridSearchCV over Random Forest's `n_estimators`, `max_depth`, and `max_features` landed on `{max_depth: 10, max_features: 'sqrt', n_estimators: 300}`, with a CV F1 of 0.737 and an OOB score of 0.823.

**Regression side-task — predicting fare.** MAE=20.809, RMSE=30.473, R²=0.400, Adjusted R²=0.375. When I looked at the residual plot, there's a clear funnel shape — residuals stay tight at low predicted fares and spread out a lot as predicted fare increases. That's heteroscedasticity, and it makes sense given how right-skewed fare is to begin with: the model's errors are naturally larger on the handful of expensive tickets it's least equipped to predict precisely.

**My final recommendation.** I'd deploy Random Forest. It had the best accuracy (0.816), precision (0.800), and F1 (0.744) of the three, and the tuned version reached an OOB score of 0.823. Logistic Regression's slightly higher AUC (0.844 vs 0.827) wasn't enough to outweigh its weaker precision/recall balance for me. The fare-regression model is a separate, lower-confidence exercise — an R² of 0.40 and confirmed heteroscedasticity mean I'd treat it as exploratory rather than something to actually deploy.

**Saved artifact.** I put the complete tuned Random Forest pipeline (preprocessing + classifier together) to `best_pipeline.joblib`, reloaded it, and confirmed it produces identical predictions on raw, unpreprocessed input.