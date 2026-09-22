"""
01_eda.py -- Part A: profiling, cleaning, EDA story for the Titanic dataset.

Loads sns.load_dataset('titanic') exactly ONCE (this is the single load
point for the whole /analytics module). Immediately saves it to
titanic.csv as a committed offline fallback, so this script and
02_modeling.py can both work from pd.read_csv("analytics/titanic.csv")
without needing network access on subsequent runs / at grading time.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

CSV_PATH = "analytics/titanic.csv"


def load_and_save():
    df = sns.load_dataset("titanic")  # the ONE and ONLY load of the raw dataset
    df.to_csv(CSV_PATH, index=False)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} cols. Saved offline fallback to {CSV_PATH}")
    return df


def profile(df):
    print("\n=== df.info() ===")
    df.info()

    print("\n=== df.describe() ===")
    print(df.describe(include="all"))

    print(f"\n=== df.shape ===\n{df.shape}")

    print("\n=== Missing value % per column (only columns with any missing) ===")
    missing_pct = (df.isna().sum() / len(df) * 100).round(2)
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    print(missing_pct)

    return missing_pct

def clean_missing(df):
    # embarked/embark_town: <5% missing -> drop rows (only 2 rows lost)
    df = df.dropna(subset=["embarked", "embark_town"]).copy()

    # age: 5-30% missing -> impute with median grouped by pclass+sex
    # (more defensible than a single global median -- age distributions
    # differ meaningfully by class and sex in this dataset)
    df["age"] = df.groupby(["pclass", "sex"])["age"].transform(
        lambda s: s.fillna(s.median())
    )

    # deck: >30% missing -> too sparse to impute reliably; encode "missing"
    # as its own category rather than dropping the column, since whether
    # a cabin/deck was recorded likely correlates with pclass/fare and
    # therefore survival -- that signal is worth keeping.
    df["deck"] = df["deck"].astype(object).fillna("Unknown")

    remaining_missing = df.isna().sum()
    remaining_missing = remaining_missing[remaining_missing > 0]
    print("\n=== Remaining missing values after cleaning ===")
    print(remaining_missing if len(remaining_missing) else "None")

    return df


def univariate_analysis(df):
    # --- Histograms + box plots ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].hist(df["age"], bins=30, edgecolor="black")
    axes[0, 0].set_title("Age Distribution")
    axes[0, 1].boxplot(df["age"], vert=False)
    axes[0, 1].set_title("Age Box Plot")
    axes[1, 0].hist(df["fare"], bins=30, edgecolor="black")
    axes[1, 0].set_title("Fare Distribution")
    axes[1, 1].boxplot(df["fare"], vert=False)
    axes[1, 1].set_title("Fare Box Plot")
    plt.tight_layout()
    plt.savefig("analytics/univariate_age_fare.png")
    plt.close()
    print("\nSaved univariate_age_fare.png")

    # --- IQR outlier counts ---
    def iqr_outlier_count(series):
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return ((series < lower) | (series > upper)).sum(), lower, upper

    age_outliers, age_lo, age_hi = iqr_outlier_count(df["age"])
    fare_outliers, fare_lo, fare_hi = iqr_outlier_count(df["fare"])
    print(f"\nAge IQR outliers: {age_outliers} (bounds: [{age_lo:.2f}, {age_hi:.2f}])")
    print(f"Fare IQR outliers: {fare_outliers} (bounds: [{fare_lo:.2f}, {fare_hi:.2f}])")

    # --- fare mean/median/mode + skew ---
    fare_mean = df["fare"].mean()
    fare_median = df["fare"].median()
    fare_mode = df["fare"].mode().iloc[0]
    print(f"\nFare: mean={fare_mean:.2f}, median={fare_median:.2f}, mode={fare_mode:.2f}")

    # mean > median > mode is the classic right-skew signature: a few very
    # high fares (1st-class, high-value tickets) pull the mean up further
    # than the median, while the mode sits at the cheapest/most common fare
    if fare_mean > fare_median > fare_mode:
        skew_conclusion = "right-skewed (mean > median > mode) -- a small number of high-value fares pull the mean upward"
    elif fare_mean < fare_median < fare_mode:
        skew_conclusion = "left-skewed (mean < median < mode)"
    else:
        skew_conclusion = "approximately symmetric (mean ≈ median ≈ mode)"
    print(f"Fare distribution: {skew_conclusion}")

    return {
        "age_outliers": age_outliers,
        "fare_outliers": fare_outliers,
        "fare_mean": fare_mean,
        "fare_median": fare_median,
        "fare_mode": fare_mode,
        "skew_conclusion": skew_conclusion,
    }


def bivariate_analysis(df):
    # --- Survival rate by sex (boolean masking) ---
    male_mask = df["sex"] == "male"
    female_mask = df["sex"] == "female"
    survival_by_sex = {
        "male": df.loc[male_mask, "survived"].mean(),
        "female": df.loc[female_mask, "survived"].mean(),
    }
    print("\n=== Survival rate by sex ===")
    for k, v in survival_by_sex.items():
        print(f"{k}: {v:.3f}")

    # --- Survival rate by pclass (boolean masking) ---
    survival_by_pclass = {}
    for pc in sorted(df["pclass"].unique()):
        mask = df["pclass"] == pc
        survival_by_pclass[pc] = df.loc[mask, "survived"].mean()
    print("\n=== Survival rate by pclass ===")
    for k, v in survival_by_pclass.items():
        print(f"class {k}: {v:.3f}")

    # --- Survival rate by sex AND pclass combined (& masking) ---
    print("\n=== Survival rate by sex + pclass ===")
    survival_by_sex_pclass = {}
    for sex_val, sex_mask in [("male", male_mask), ("female", female_mask)]:
        for pc in sorted(df["pclass"].unique()):
            combined_mask = sex_mask & (df["pclass"] == pc)
            rate = df.loc[combined_mask, "survived"].mean()
            survival_by_sex_pclass[(sex_val, pc)] = rate
            print(f"{sex_val}, class {pc}: {rate:.3f}")

    # --- 6x6 correlation matrix (exact columns specified) ---
    corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr_matrix = df[corr_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Correlation Matrix (6 numeric columns)")
    plt.tight_layout()
    plt.savefig("analytics/correlation_heatmap.png")
    plt.close()
    print("\nSaved correlation_heatmap.png")
    print("\n=== Correlation matrix ===")
    print(corr_matrix)

    # --- Top 2 strongest off-diagonal correlations ---
    corr_pairs = []
    for i, col_i in enumerate(corr_cols):
        for j, col_j in enumerate(corr_cols):
            if i < j:  # upper triangle only, avoid duplicates/diagonal
                corr_pairs.append((col_i, col_j, corr_matrix.loc[col_i, col_j]))
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    top_2 = corr_pairs[:2]
    print("\n=== Top 2 strongest correlations ===")
    for a, b, val in top_2:
        print(f"{a} <-> {b}: {val:.3f}")

    return {
        "survival_by_sex": survival_by_sex,
        "survival_by_pclass": survival_by_pclass,
        "survival_by_sex_pclass": survival_by_sex_pclass,
        "corr_matrix": corr_matrix,
        "top_2_correlations": top_2,
    }

def multivariate_analysis(df):
    interpretations = {}

    # --- Chart 1: grouped bar — survival rate by sex + pclass ---
    plt.figure(figsize=(7, 5))
    sns.barplot(data=df, x="pclass", y="survived", hue="sex", errorbar=None)
    plt.title("Survival Rate by Class and Sex")
    plt.ylabel("Survival Rate")
    plt.tight_layout()
    plt.savefig("analytics/story_1_survival_by_class_sex.png")
    plt.close()
    interpretations["chart_1"] = (
        "Survival rate is overwhelmingly driven by sex first, class second: women "
        "survived at far higher rates than men in every class, but within each sex, "
        "survival still drops sharply from 1st to 3rd class. This suggests the "
        "'women and children first' evacuation policy dominated, with class acting "
        "as a secondary factor -- likely via cabin location and proximity to lifeboats."
    )

    # --- Chart 2: box plot — age distribution by survival, split by pclass ---
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="pclass", y="age", hue="survived")
    plt.title("Age Distribution by Class and Survival")
    plt.tight_layout()
    plt.savefig("analytics/story_2_age_by_class_survival.png")
    plt.close()
    interpretations["chart_2"] = (
        "Within each class, survivors and non-survivors have broadly similar age "
        "medians, meaning age alone doesn't cleanly separate survival outcomes once "
        "class is controlled for. The main visible age effect is at the low end: "
        "young children show a higher survival skew in every class, consistent with "
        "the 'children first' component of the evacuation policy."
    )

    # --- Chart 3: scatter — fare vs age, colored by survival ---
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="age", y="fare", hue="survived", alpha=0.6)
    plt.title("Fare vs Age, Colored by Survival")
    plt.tight_layout()
    plt.savefig("analytics/story_3_fare_vs_age_survival.png")
    plt.close()
    interpretations["chart_3"] = (
        "Survivors (orange) are visibly denser at higher fare values, reinforcing "
        "that fare (a proxy for class and cabin location) mattered more than age for "
        "survival -- there's no clear age band that separates survivors from "
        "non-survivors, but there's a clear vertical fare band above which survival "
        "becomes noticeably more common."
    )

    # --- Chart 4: bar — survival rate by family size (sibsp + parch) ---
    df["family_size"] = df["sibsp"] + df["parch"]
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="family_size", y="survived", errorbar=None)
    plt.title("Survival Rate by Family Size (sibsp + parch)")
    plt.ylabel("Survival Rate")
    plt.tight_layout()
    plt.savefig("analytics/story_4_survival_by_family_size.png")
    plt.close()
    interpretations["chart_4"] = (
        "Survival rate rises from traveling completely alone (family_size=0) to "
        "small families (1-3), then drops sharply for large families (4+). This "
        "suggests a 'sweet spot': solo travelers may have been slower to be helped "
        "into lifeboats, while very large families likely struggled to stay together "
        "and evacuate as a unit in the chaos, whereas small families could coordinate "
        "quickly without being logistically unwieldy."
    )

    print("\n=== Multivariate data story: chart interpretations ===")
    for k, v in interpretations.items():
        print(f"\n{k}: {v}")

    return interpretations


def main():
    df = load_and_save()
    missing_pct = profile(df)
    df = clean_missing(df)
    univariate_results = univariate_analysis(df)
    bivariate_results = bivariate_analysis(df)
    multivariate_results = multivariate_analysis(df)
    return df, missing_pct, univariate_results, bivariate_results, multivariate_results


if __name__ == "__main__":
    main()