"""
sensitivity.py - how much does the chosen model depend on the two fee columns?

'Debtor' and 'Tuition fees up to date' are NOT on the assignment's prohibited
list, so the PRIMARY model keeps them. But a student's fee status can change
after enrollment (e.g. they stop paying because they are about to drop out),
so these columns may be partly "future" information. This check re-runs the
chosen model WITHOUT them so the reader can see the honest enrollment-only
performance range.

This is a side analysis. It does not change the primary model or its selection.
"""
import pandas as pd
from sklearn.base import clone

from src import config, evaluate
from src.models import get_models
from src.preprocessing import build_pipeline
from src.train import cross_validate_models


def run_sensitivity(model_name, X_train, y_train, X_test, y_test, cv_primary, test_primary):
    drop = config.POSSIBLY_LATE_FEATURES
    model = clone(get_models()[model_name])

    cv_alt = cross_validate_models({model_name: model}, X_train, y_train, exclude=drop)
    pipe = build_pipeline(clone(model), exclude=drop).fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    m = evaluate.compute_metrics(y_test, y_pred)
    pc = evaluate.per_class_table(y_test, y_pred)

    table = pd.DataFrame([
        {"setting": "Primary (all 24 features)",
         "cv_macro_f1": cv_primary.loc[model_name, "cv_macro_f1_mean"],
         "test_macro_f1": test_primary.loc[model_name, "macro_f1"],
         "test_accuracy": test_primary.loc[model_name, "accuracy"]},
        {"setting": f"Without {', '.join(drop)} (22 features)",
         "cv_macro_f1": cv_alt.loc[model_name, "cv_macro_f1_mean"],
         "test_macro_f1": m["macro_f1"],
         "test_accuracy": m["accuracy"]},
    ]).set_index("setting").round(4)
    table.to_csv(config.OUTPUT_DIR / "sensitivity_without_fee_features.csv")
    pc.round(4).to_csv(config.OUTPUT_DIR / "sensitivity_per_class_without_fee_features.csv")
    return table, pc
