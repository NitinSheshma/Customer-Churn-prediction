"""
Customer Churn Prediction - Telco Customer Churn dataset (7,043 customers)

Pipeline:
  1. Load + clean data
  2. Feature engineering (avg monthly charge, tenure group)
  3. Stratified train/test split
  4. Compare Logistic Regression, Random Forest, Gradient Boosting
     (class weights to handle imbalance; metrics: recall + ROC-AUC)
  5. Tune Random Forest with GridSearchCV
  6. Feature importance + plots
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             classification_report, precision_score,
                             recall_score, roc_auc_score, f1_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")


# ----------------------------------------------------------------------------
# 1. Load & clean
# ----------------------------------------------------------------------------
def load_and_clean(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place the Telco Customer Churn CSV in data/ with the filename expected by --data."
        )

    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")

    required = {"TotalCharges", "Churn", "tenure", "MonthlyCharges"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    df = df.drop(columns=["customerID"])

    # TotalCharges has blank strings for brand-new customers (tenure == 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = df["TotalCharges"].isna().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    print(f"Fixed {n_missing} blank TotalCharges values")

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    if df["Churn"].isna().any():
        raise ValueError("Churn column must contain only 'Yes' or 'No' values")
    df["Churn"] = df["Churn"].astype(int)
    df = df.drop_duplicates()
    return df


# ----------------------------------------------------------------------------
# 2. Feature engineering
# ----------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Average amount paid per month of tenure (fall back to MonthlyCharges for new customers)
    df["AvgMonthlyCharge"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"].replace(0, np.nan), df["MonthlyCharges"]
    )
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, np.inf],
        labels=["0-12m", "13-24m", "25-48m", "49-60m", "60m+"],
    ).astype(str)
    return df


# ----------------------------------------------------------------------------
# 3. EDA plots
# ----------------------------------------------------------------------------
def eda_plots(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    df["Churn"].map({0: "Stayed", 1: "Churned"}).value_counts().plot.bar(
        ax=ax, color=["#4C72B0", "#C44E52"])
    ax.set_title(f"Class balance (churn rate = {df['Churn'].mean():.1%})")
    ax.set_ylabel("Customers")
    plt.xticks(rotation=0)
    fig.tight_layout(); fig.savefig(out / "01_class_balance.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, col in zip(axes, ["Contract", "TenureGroup", "InternetService"]):
        order = None
        if col == "TenureGroup":
            order = ["0-12m", "13-24m", "25-48m", "49-60m", "60m+"]
        rate = df.groupby(col)["Churn"].mean()
        if order:
            rate = rate.reindex(order)
        rate.plot.bar(ax=ax, color="#C44E52")
        ax.set_title(f"Churn rate by {col}"); ax.set_ylabel("Churn rate")
        ax.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(out / "02_churn_rate_by_segment.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(data=df, x="MonthlyCharges", hue=df["Churn"].map({0: "Stayed", 1: "Churned"}),
                fill=True, common_norm=False, ax=ax)
    ax.set_title("Monthly charges: churned vs stayed")
    fig.tight_layout(); fig.savefig(out / "03_monthly_charges_dist.png", dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------
# 4. Modelling
# ----------------------------------------------------------------------------
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])


def evaluate(name, model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "recall": recall_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def main(data_path: Path, out: Path, quick: bool = False) -> None:
    out.mkdir(exist_ok=True, parents=True)

    df = add_features(load_and_clean(data_path))
    print(f"Churn rate: {df['Churn'].mean():.1%}")
    eda_plots(df, out)

    X, y = df.drop(columns="Churn"), df["Churn"]
    # Stratified split keeps the ~26% churn ratio identical in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    prep = build_preprocessor(X)
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        # GradientBoosting has no class_weight arg -> balanced sample weights at fit time
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    # ---- compare baseline models ----
    results, fitted = [], {}
    sw = compute_sample_weight("balanced", y_train)
    for name, clf in models.items():
        pipe = Pipeline([("prep", prep), ("clf", clf)])
        fit_kwargs = {"clf__sample_weight": sw} if name == "Gradient Boosting" else {}
        pipe.fit(X_train, y_train, **fit_kwargs)
        fitted[name] = pipe
        results.append(evaluate(name, pipe, X_test, y_test))
        print(f"  trained {name}")

    # ---- tune Random Forest ----
    rf_pipe = Pipeline([
        ("prep", prep),
        ("clf", RandomForestClassifier(class_weight="balanced",
                                       random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    param_grid = {
        "clf__n_estimators": [200] if quick else [200, 400],
        "clf__max_depth": [8, 12] if quick else [6, 10, 14, None],
        "clf__min_samples_leaf": [5] if quick else [1, 5, 10],
        "clf__max_features": ["sqrt"] if quick else ["sqrt", 0.5],
    }
    grid = GridSearchCV(
        rf_pipe, param_grid, cv=5 if not quick else 3,
        scoring={"roc_auc": "roc_auc", "recall": "recall"}, refit="roc_auc", n_jobs=-1)
    print("Running GridSearchCV on Random Forest ...")
    grid.fit(X_train, y_train)
    print("Best params:", grid.best_params_)
    best_rf = grid.best_estimator_
    fitted["Tuned Random Forest"] = best_rf
    results.append(evaluate("Tuned Random Forest", best_rf, X_test, y_test))

    # ---- report ----
    res_df = pd.DataFrame(results).set_index("model").round(3)
    print("\n=== Test-set results ===")
    print(res_df.to_string())
    res_df.to_csv(out / "model_comparison.csv")
    with open(out / "best_params.json", "w") as f:
        json.dump(grid.best_params_, f, indent=2)

    print("\n=== Tuned Random Forest - classification report ===")
    print(classification_report(y_test, best_rf.predict(X_test),
                                target_names=["Stayed", "Churned"]))

    # ---- plots: ROC curves, confusion matrix ----
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, m in fitted.items():
        RocCurveDisplay.from_estimator(m, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=.4)
    ax.set_title("ROC curves")
    fig.tight_layout(); fig.savefig(out / "04_roc_curves.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        best_rf, X_test, y_test, display_labels=["Stayed", "Churned"], cmap="Blues", ax=ax)
    ax.set_title("Tuned Random Forest - confusion matrix")
    fig.tight_layout(); fig.savefig(out / "05_confusion_matrix.png", dpi=150); plt.close(fig)

    # ---- feature importance ----
    names = best_rf.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    imp = pd.Series(best_rf.named_steps["clf"].feature_importances_, index=names)
    top = imp.sort_values(ascending=False).head(15)
    top.to_csv(out / "feature_importance_top15.csv", header=["importance"])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=top.values, y=top.index, color="#4C72B0", ax=ax)
    ax.set_title("Top 15 features driving churn (Random Forest)")
    ax.set_xlabel("Importance")
    fig.tight_layout(); fig.savefig(out / "06_feature_importance.png", dpi=150); plt.close(fig)
    print("\nTop 10 features:\n", top.head(10).round(3).to_string())

    joblib.dump(best_rf, out / "churn_model.joblib")
    print(f"\nAll outputs saved to: {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--quick", action="store_true", help="Smaller grid for a fast test run")
    args = ap.parse_args()
    main(args.data, args.out, args.quick)
