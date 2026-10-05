"""
predict.py - use the SAVED pipeline to predict for new students.

Because the saved object is the whole Pipeline (preprocessing + model), new
data automatically gets EXACTLY the same preprocessing that was used in
training - no manual steps to forget.

Usage from the project root:
    python -m src.predict                          # demo on sample students
    python -m src.predict --csv my_students.csv    # your own file (';' or ',' delimited)
"""
import argparse

import joblib
import pandas as pd

from src import config, data_loader
from src.preprocessing import make_train_test_split, split_features_target


def load_model(path=config.MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python run_all.py` first to train and save the model.")
    return joblib.load(path)


def predict_students(pipeline, students: pd.DataFrame) -> pd.DataFrame:
    """
    Predict outcome + class probabilities. `students` must contain the 24
    enrollment-time columns; any other columns (e.g. semester columns or
    Target) are ignored, never used.
    """
    missing = [c for c in config.FEATURE_COLUMNS if c not in students.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    X = students[config.FEATURE_COLUMNS]
    result = pd.DataFrame(index=students.index)
    result["predicted_outcome"] = pipeline.predict(X)
    proba = pd.DataFrame(pipeline.predict_proba(X), columns=pipeline.classes_, index=students.index)
    for cls in config.CLASS_ORDER:
        result[f"P({cls})"] = proba[cls].round(3)
    return result


def sample_test_students(per_class=2) -> pd.DataFrame:
    """
    Rebuild the SAME train/test split (same seed) and pick a few unseen test
    students per true class, so we can compare prediction vs. reality.
    """
    df = data_loader.load_data()
    X, y = split_features_target(df)
    _, X_test, _, y_test = make_train_test_split(X, y)
    chosen_index = []
    for cls in config.CLASS_ORDER:
        of_this_class = y_test[y_test == cls]
        chosen_index += list(of_this_class.sample(per_class, random_state=config.RANDOM_SEED).index)
    chosen = y_test.loc[chosen_index]
    sample = X_test.loc[chosen_index].copy()
    sample.insert(0, "actual_outcome", chosen)
    return sample


def hypothetical_students() -> pd.DataFrame:
    """
    Two made-up students built from the TRAINING data's typical profile
    (median for numbers, most common value for flags/codes).
    Illustration only - they show the model accepts brand-new records.
    """
    df = data_loader.load_data()
    X, y = split_features_target(df)
    X_train, _, _, _ = make_train_test_split(X, y)

    typical = {}
    for col in X_train.columns:
        if col in config.NUMERIC_FEATURES:
            typical[col] = X_train[col].median()
        else:
            typical[col] = X_train[col].mode().iloc[0]

    a = dict(typical)                       # typical student, fees paid, no debt
    b = dict(typical)                       # same, but older, in debt, fees overdue
    b.update({"Age at enrollment": 30, "Debtor": 1, "Tuition fees up to date": 0,
              "Scholarship holder": 0})
    out = pd.DataFrame([a, b], index=["Hypothetical A (typical)", "Hypothetical B (debt, fees overdue, age 30)"])
    out.insert(0, "actual_outcome", "(unknown - made up)")
    return out


def run_demo(pipeline=None, save=True) -> pd.DataFrame:
    pipeline = pipeline or load_model()
    sample = sample_test_students()
    made_up = hypothetical_students()
    combined = pd.concat([sample, made_up])

    preds = predict_students(pipeline, combined.drop(columns="actual_outcome"))
    preds.insert(0, "actual_outcome", combined["actual_outcome"])
    preds["correct"] = [("n/a" if a.startswith("(") else ("yes" if a == p else "NO"))
                        for a, p in zip(preds["actual_outcome"], preds["predicted_outcome"])]
    if save:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        preds.to_csv(config.OUTPUT_DIR / "sample_predictions.csv")
    return preds


def main():
    parser = argparse.ArgumentParser(description="Predict student outcomes with the saved model.")
    parser.add_argument("--csv", help="CSV file with the 24 enrollment-time columns")
    args = parser.parse_args()
    pipeline = load_model()
    pd.set_option("display.width", 200)
    if args.csv:
        students = pd.read_csv(args.csv, sep=None, engine="python")   # auto-detect ; or ,
        students.columns = students.columns.str.strip()
        print(predict_students(pipeline, students).to_string())
    else:
        print(run_demo(pipeline).to_string())


if __name__ == "__main__":
    main()
