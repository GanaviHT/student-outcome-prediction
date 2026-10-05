"""
interpret.py - which features does the final model rely on?

METHOD: permutation importance
------------------------------
Take the held-out test students, shuffle ONE column at a time (this breaks its
link to the outcome) and measure how much the model's Macro F1 drops. A big
drop = the model relied heavily on that column. It works for ANY model and is
measured on the ORIGINAL columns (e.g. "Course", not "Course_9500"), which is
easier to explain than one-hot columns.

IMPORTANT: importance shows what the MODEL uses to predict. It does NOT prove
that a feature CAUSES dropout or graduation.
(We use the test set here only to describe the already-chosen model; no
decision about the model is made from it.)
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from src import config


def compute_permutation_importance(pipeline, X_test, y_test, n_repeats=10) -> pd.DataFrame:
    result = permutation_importance(
        pipeline, X_test, y_test,
        scoring="f1_macro", n_repeats=n_repeats,
        random_state=config.RANDOM_SEED, n_jobs=1,
    )
    table = pd.DataFrame({
        "feature": X_test.columns,
        "importance_macro_f1_drop": result.importances_mean,
        "std": result.importances_std,
    })
    return table.sort_values("importance_macro_f1_drop", ascending=False).reset_index(drop=True)


def plot_importance(table: pd.DataFrame, model_name: str, top_n=15, filename="feature_importance.png"):
    top = table.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance_macro_f1_drop"], xerr=top["std"], color="#2c7fb8")
    ax.set_xlabel("Drop in Macro F1 when the feature is shuffled (test set)")
    ax.set_title(f"Permutation importance - {model_name}\n(association with prediction, NOT causation)")
    fig.tight_layout()
    if filename:
        config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.FIGURE_DIR / filename, dpi=130, bbox_inches="tight")
    return fig
