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


def main():
    df = load_and_save()
    missing_pct = profile(df)
    return df, missing_pct


if __name__ == "__main__":
    main()