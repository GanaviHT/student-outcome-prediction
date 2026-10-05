"""
preprocessing.py - everything that turns the raw table into model-ready numbers.

KEY IDEA (leakage safety)
-------------------------
We build a scikit-learn ColumnTransformer and put it INSIDE a Pipeline together
with the model. A Pipeline's .fit() learns things (medians, scaling factors,
the list of one-hot categories) from the data it is given ONLY. So:

  * pipeline.fit(X_train, y_train)  -> learns from training students only
  * pipeline.predict(X_test)        -> just APPLIES what was learned

During cross-validation sklearn re-fits the whole pipeline inside every fold,
so even the validation fold never influences the preprocessing.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def get_feature_lists(exclude=None):
    """Return (numeric, binary, categorical) lists, optionally dropping some columns."""
    exclude = set(exclude or [])
    return (
        [c for c in config.NUMERIC_FEATURES if c not in exclude],
        [c for c in config.BINARY_FEATURES if c not in exclude],
        [c for c in config.CATEGORICAL_FEATURES if c not in exclude],
    )


def split_features_target(df: pd.DataFrame, exclude=None):
    """
    Separate predictors (X) from the target (y).

    X contains ONLY the permitted enrollment-time columns. The 12 prohibited
    semester-performance columns are never selected, so they cannot leak in.
    """
    numeric, binary, categorical = get_feature_lists(exclude)
    columns = numeric + binary + categorical
    banned_present = set(columns) & set(config.PROHIBITED_COLUMNS)
    assert not banned_present, f"Prohibited columns selected: {banned_present}"
    return df[columns].copy(), df[config.TARGET].copy()


def make_train_test_split(X, y):
    """
    80/20 split, stratified on the target so that every class has the same
    proportion in train and test. The seed makes the split reproducible.
    """
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_SEED,
    )


def build_preprocessor(exclude=None, scale_numeric=True) -> ColumnTransformer:
    """
    Build the (unfitted) ColumnTransformer.

    numeric     -> fill missing with the median, then standardise (mean 0, std 1)
    binary 0/1  -> fill missing with the most common value, otherwise unchanged
    categorical -> fill missing with the most common value, then one-hot encode.
                   handle_unknown="ignore": a code never seen in training becomes
                   an all-zero row instead of crashing at prediction time.

    (The dataset has no missing values, but the imputers make the pipeline
    safe for new data and are still fitted on training data only.)
    """
    numeric, binary, categorical = get_feature_lists(exclude)

    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric),
            ("bin", SimpleImputer(strategy="most_frequent"), binary),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical),
        ],
        remainder="drop",   # anything not listed above is dropped, never silently used
    )


def build_pipeline(model, exclude=None) -> Pipeline:
    """preprocessor + model glued together into one object."""
    return Pipeline([
        ("preprocess", build_preprocessor(exclude=exclude)),
        ("model", model),
    ])
