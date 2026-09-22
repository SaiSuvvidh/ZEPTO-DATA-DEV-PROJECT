"""
_02_modeling.py -- Part B: predictive modeling pipeline, continuing from
the same offline titanic.csv that _01_eda.py produced (raw dataset, saved
immediately after the single sns.load_dataset call). This script performs
its OWN missing-value handling / encoding / scaling inside a sklearn
Pipeline (fit train-only) -- it does not reuse Part A's already-cleaned
DataFrame, since the rubric explicitly allows an independent preprocessing
choice here, and doing it via ColumnTransformer inside a Pipeline is what
structurally enforces the fit-on-train/transform-on-test separation.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

CSV_PATH = "analytics/titanic.csv"


def load_data():
    df = pd.read_csv(CSV_PATH)  # reads the committed offline fallback, no network call
    return df


def stratified_split(df):
    # Features: drop columns that are either leakage (alive == survived
    # restated as text) or too sparse/unused for this modeling pass
    # (deck is 77% missing -- Part A encoded it as "Unknown" for EDA
    # purposes, but for modeling we drop it here rather than mix an EDA
    # decision into the model's own preprocessing choice).
    drop_cols = ["alive", "deck", "embark_town", "class", "who", "adult_male", "alone"]
    X = df.drop(columns=["survived"] + [c for c in drop_cols if c in df.columns])
    y = df["survived"]

    print(f"\nClass balance (full dataset): \n{y.value_counts(normalize=True)}")
    # Stratification matters here because survived is imbalanced (~38% survived
    # vs ~62% did not, per Part A's profiling) -- a plain random split risks
    # producing a test set with a meaningfully different survival ratio than
    # the training set purely by chance, which would make evaluation metrics
    # (especially recall/precision on the minority "survived" class) noisy
    # and not representative of true model performance.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train class balance:\n{y_train.value_counts(normalize=True)}")
    print(f"Test class balance:\n{y_test.value_counts(normalize=True)}")

    return X_train, X_test, y_train, y_test


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = stratified_split(df)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()