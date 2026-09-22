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


def main():
    df = load_and_save()
    missing_pct = profile(df)
    df = clean_missing(df)
    univariate_results = univariate_analysis(df)
    return df, missing_pct, univariate_results


if __name__ == "__main__":
    main()