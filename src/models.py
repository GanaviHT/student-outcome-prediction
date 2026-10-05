"""
models.py - the candidate models we compare.

All use the SAME preprocessing (see preprocessing.py). The imbalance-aware
models use class_weight="balanced": mistakes on the small "Enrolled" class are
penalised more heavily DURING TRAINING. Nothing is resampled and no test data
is touched.
"""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src import config


def get_models() -> dict:
    seed = config.RANDOM_SEED
    return {
        # 0. Reference only: always predicts the most common class. Any real
        #    model must beat this, otherwise it has learned nothing.
        "Dummy (most frequent)": DummyClassifier(strategy="most_frequent"),

        # 1. Simple, interpretable baseline - NO class weighting.
        "Logistic Regression (no weights)": LogisticRegression(
            max_iter=2000, random_state=seed),

        # 2. Same baseline WITH class weighting -> shows the effect of imbalance handling.
        "Logistic Regression (balanced)": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed),

        # 3. Tree ensemble: many decision trees voting. Handles non-linear
        #    effects and interactions. min_samples_leaf limits overfitting.
        "Random Forest (balanced)": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed),

        # 4. Gradient boosting: trees built one after another, each fixing the
        #    errors of the previous ones. Strong on tabular data.
        "Gradient Boosting (balanced)": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=200, max_depth=4, min_samples_leaf=20,
            class_weight="balanced", random_state=seed),
    }
