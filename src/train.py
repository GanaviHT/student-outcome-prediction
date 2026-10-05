"""
train.py - model comparison, evaluation and selection.

THE PROTOCOL (this is what keeps the test set honest)
-----------------------------------------------------
1. The test set was split off BEFORE anything else and is not touched here
   until step 3.
2. MODEL SELECTION uses 5-fold stratified cross-validation on the TRAINING
   set only. The model with the best mean Macro F1 wins.
3. After selection, every model is fitted on the full training set and
   evaluated ONCE on the test set for the final report. The test results are
   reported, never used to choose or tune anything.
"""
import json
import re
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_validate

from src import config, evaluate
from src.preprocessing import build_pipeline


def slugify(name: str) -> str:
    """'Random Forest (balanced)' -> 'random_forest_balanced' (safe file name)."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# ---------------------------------------------------------------------------
# Step 2: model selection with cross-validation (training data only)
# ---------------------------------------------------------------------------
def cross_validate_models(models: dict, X_train, y_train, exclude=None) -> pd.DataFrame:
    """
    Stratified K-fold CV for each model. Because we pass the whole PIPELINE
    (preprocessing + model), sklearn re-learns the preprocessing inside every
    fold from that fold's training part only.
    """
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_SEED)
    scoring = {"macro_f1": "f1_macro", "accuracy": "accuracy",
               "balanced_accuracy": "balanced_accuracy"}
    rows = []
    for name, model in models.items():
        start = time.time()
        scores = cross_validate(build_pipeline(model, exclude=exclude), X_train, y_train,
                                cv=cv, scoring=scoring, n_jobs=1)
        rows.append({
            "model": name,
            "cv_macro_f1_mean": scores["test_macro_f1"].mean(),
            "cv_macro_f1_std": scores["test_macro_f1"].std(),
            "cv_accuracy_mean": scores["test_accuracy"].mean(),
            "cv_balanced_accuracy_mean": scores["test_balanced_accuracy"].mean(),
        })
        print(f"  CV  {name:<34s} macro-F1 = {rows[-1]['cv_macro_f1_mean']:.4f} "
              f"(+/- {rows[-1]['cv_macro_f1_std']:.4f})   [{time.time() - start:.0f}s]")
    return pd.DataFrame(rows).set_index("model")


def select_best_model(cv_results: pd.DataFrame) -> str:
    """Pick the model with the highest mean cross-validated Macro F1."""
    return cv_results["cv_macro_f1_mean"].idxmax()


# ---------------------------------------------------------------------------
# Step 3: fit on all training data, evaluate once on the test set
# ---------------------------------------------------------------------------
def fit_and_evaluate(models: dict, X_train, y_train, X_test, y_test, exclude=None,
                     save_figures=True):
    """
    Returns
    -------
    fitted      : {name: fitted Pipeline}
    test_table  : DataFrame, one row per model, overall test metrics
    per_class   : {name: DataFrame of per-class precision/recall/F1}
    predictions : {name: array of predicted labels for X_test}
    """
    fitted, per_class, predictions, rows = {}, {}, {}, []
    for name, model in models.items():
        pipe = build_pipeline(model, exclude=exclude)
        pipe.fit(X_train, y_train)                 # learns from TRAIN only
        y_pred = pipe.predict(X_test)              # only APPLIES what was learned

        fitted[name], predictions[name] = pipe, y_pred
        per_class[name] = evaluate.per_class_table(y_test, y_pred)
        rows.append({"model": name, **evaluate.compute_metrics(y_test, y_pred)})

        if save_figures:
            config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
            evaluate.plot_confusion_matrix(
                y_test, y_pred, f"Confusion matrix (test set) - {name}",
                config.FIGURE_DIR / f"cm_{slugify(name)}.png")
    return fitted, pd.DataFrame(rows).set_index("model"), per_class, predictions


def per_class_long_table(per_class: dict) -> pd.DataFrame:
    """Stack the per-class tables of all models into one tidy table."""
    frames = []
    for name, table in per_class.items():
        t = table.copy()
        t.insert(0, "class", t.index)
        t.insert(0, "model", name)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def save_results(cv_results, test_table, per_class, y_test, predictions, best_name):
    """Write all result tables to outputs/ (these are the numbers quoted in the README)."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv_results.round(4).to_csv(config.OUTPUT_DIR / "cv_results.csv")
    test_table.round(4).to_csv(config.OUTPUT_DIR / "test_metrics.csv")
    per_class_long_table(per_class).round(4).to_csv(
        config.OUTPUT_DIR / "per_class_metrics_test.csv", index=False)

    with open(config.OUTPUT_DIR / "classification_reports.txt", "w") as f:
        for name in predictions:
            f.write(f"===== {name} (test set) =====\n")
            f.write(per_class[name].round(4).to_string() + "\n\n")
            f.write(evaluate.get_confusion_matrix(y_test, predictions[name]).to_string() + "\n\n")

    summary = {
        "selection_rule": f"highest mean {config.CV_FOLDS}-fold CV Macro F1 on the training set",
        "selected_model": best_name,
        "cv_macro_f1_mean": round(float(cv_results.loc[best_name, "cv_macro_f1_mean"]), 4),
        "test_metrics": {k: round(float(v), 4) for k, v in test_table.loc[best_name].items()},
    }
    (config.OUTPUT_DIR / "final_model_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def save_model(pipeline, model_name, X_train):
    """Save the fitted pipeline (preprocessing + model) as ONE file, plus metadata."""
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.MODEL_PATH)
    metadata = {
        "model_name": model_name,
        "random_seed": config.RANDOM_SEED,
        "trained_on_rows": int(len(X_train)),
        "feature_columns": list(X_train.columns),
        "classes": [str(c) for c in pipeline.classes_],
        "prohibited_columns_not_used": config.PROHIBITED_COLUMNS,
        "sklearn_version": sklearn.__version__,
        "note": "Fitted on the TRAINING split only (the same model that was evaluated on the test split).",
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))
