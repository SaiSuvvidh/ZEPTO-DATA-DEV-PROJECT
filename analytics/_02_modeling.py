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
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, roc_auc_score
)

CSV_PATH = "analytics/titanic.csv"
NUMERIC_FEATURES = ["age", "fare", "sibsp", "parch", "pclass"]
CATEGORICAL_FEATURES = ["sex", "embarked"]


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


def build_preprocessor():
    # Numeric: median-impute (robust to fare's right-skew/outliers from
    # Part A) then scale -- both steps fit on train only when used inside
    # a Pipeline's .fit(X_train, y_train) call.
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical: most-frequent-impute (handles embarked's 2 missing
    # values) then one-hot encode. handle_unknown="ignore" guards against
    # a category appearing in test but not train (won't happen here given
    # dataset size, but is good practice).
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    return preprocessor


def build_models(preprocessor):
    models = {
        "Logistic Regression": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Decision Tree": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(random_state=42, max_depth=5)),
        ]),
        "Random Forest": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]),
    }
    return models


def train_models(models, X_train, y_train):
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        print(f"Trained: {name}")
    return models


def plot_decision_tree(models):
    dt_pipeline = models["Decision Tree"]
    dt_model = dt_pipeline.named_steps["classifier"]
    
    # Extract the fitted preprocessor directly from the pipeline to avoid NotFittedError
    fitted_preprocessor = dt_pipeline.named_steps["preprocessor"]
    feature_names = fitted_preprocessor.get_feature_names_out()

    plt.figure(figsize=(20, 10))
    plot_tree(
        dt_model,
        feature_names=feature_names,
        class_names=["Did not survive", "Survived"],
        filled=True,
        max_depth=3,  # cap displayed depth for readability, tree itself still trained at max_depth=5
        fontsize=8,
    )
    plt.title("Decision Tree (top 3 levels shown)")
    plt.tight_layout()
    plt.savefig("analytics/decision_tree.png")
    plt.close()
    print("\nSaved decision_tree.png")


def evaluate_models(models, X_test, y_test):
    results = {}

    plt.figure(figsize=(8, 6))

    for name, pipeline in models.items():
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]  # probability of class 1 (survived)

        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

        results[name] = {
            "confusion_matrix": cm,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
        }

        print(f"\n=== {name} ===")
        print(f"Confusion matrix:\n{cm}")
        print(f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")

    plt.plot([0, 1], [0, 1], "k--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — All Three Classifiers")
    plt.legend()
    plt.tight_layout()
    plt.savefig("analytics/roc_curves.png")
    plt.close()
    print("\nSaved roc_curves.png")

    # Comparison table
    comparison_df = pd.DataFrame({
        name: {
            "Accuracy": r["accuracy"], "Precision": r["precision"],
            "Recall": r["recall"], "F1": r["f1"], "AUC": r["auc"],
        }
        for name, r in results.items()
    }).T
    print("\n=== Comparison table ===")
    print(comparison_df.round(3))

    return results, comparison_df


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = stratified_split(df)

    preprocessor = build_preprocessor()
    models = build_models(preprocessor)
    models = train_models(models, X_train, y_train)
    
    # Passing only 'models' as the preprocessor is accessed from inside the pipeline 
    # to ensure feature names are successfully extracted from the fitted state.
    plot_decision_tree(models)

    eval_results, comparison_df = evaluate_models(models, X_test, y_test)

    return X_train, X_test, y_train, y_test, models, eval_results, comparison_df


if __name__ == "__main__":
    main()