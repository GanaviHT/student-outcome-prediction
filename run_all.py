"""
run_all.py - run the COMPLETE workflow with one command:

    python run_all.py

Steps: load -> inspect -> split -> EDA -> cross-validate -> fit & evaluate ->
select -> save model -> interpret -> sample predictions -> sensitivity check.
Everything printed is also written to outputs/run_log.txt.
All numbers come from the data; nothing is hard-coded.
"""
import sys

import matplotlib
matplotlib.use("Agg")                      # draw charts to files, no pop-up windows
import matplotlib.pyplot as plt
import pandas as pd

from src import config, data_loader, eda, evaluate, interpret, predict, train
from src.models import get_models
from src.preprocessing import make_train_test_split, split_features_target
from src.sensitivity import run_sensitivity


class Tee:
    """Send print() output to the screen AND to a log file."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, text):
        for s in self.streams:
            s.write(text)
    def flush(self):
        for s in self.streams:
            s.flush()


def banner(text):
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def main():
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    log = open(config.OUTPUT_DIR / "run_log.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    banner("1. LOAD AND INSPECT")
    df = data_loader.load_data()
    data_loader.check_columns(df)
    data_loader.inspect_dataset(df)
    data_loader.column_overview(df).to_csv(config.OUTPUT_DIR / "column_overview.csv")
    print("\nNumeric ranges:\n", data_loader.data_quality_notes(df))
    print("\nCategorical-code columns:\n", data_loader.rare_category_table(df))
    eda.plot_target_distribution(df[config.TARGET], "01_target_distribution.png")

    banner("2. SEPARATE TARGET, DROP PROHIBITED COLUMNS, TRAIN/TEST SPLIT")
    X, y = split_features_target(df)
    print(f"Predictors used: {X.shape[1]} (the {len(config.PROHIBITED_COLUMNS)} semester-performance columns are excluded)")
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)
    print(f"Train: {len(X_train)} students | Test: {len(X_test)} students (seed={config.RANDOM_SEED}, stratified)")
    print("Train class %:", (y_train.value_counts(normalize=True) * 100).round(1).to_dict())
    print("Test  class %:", (y_test.value_counts(normalize=True) * 100).round(1).to_dict())

    banner("3. EDA (training data only)")
    train_df = X_train.assign(**{config.TARGET: y_train})
    eda.run_all_eda(train_df)
    eda.binary_outcome_table(train_df).to_csv(config.OUTPUT_DIR / "binary_feature_outcome_table.csv", index=False)
    print(f"EDA figures saved to {config.FIGURE_DIR}")

    banner("4. MODEL SELECTION: 5-fold CV on TRAINING data")
    models = get_models()
    cv_results = train.cross_validate_models(models, X_train, y_train)
    best_name = train.select_best_model(cv_results)
    print(f"\nSelected by highest CV Macro F1: {best_name}")

    banner("5. FIT ON TRAIN, EVALUATE ONCE ON TEST SET")
    fitted, test_table, per_class, predictions = train.fit_and_evaluate(
        models, X_train, y_train, X_test, y_test)
    evaluate.plot_per_class_f1({k: v for k, v in per_class.items()},
                               config.FIGURE_DIR / "per_class_f1_by_model.png")
    summary = train.save_results(cv_results, test_table, per_class, y_test, predictions, best_name)
    print("CV results:\n", cv_results.round(4))
    print("\nTest-set metrics:\n", test_table.round(4))
    for name in models:
        print(f"\n--- {name} ---")
        print(per_class[name].round(3))
        print(evaluate.get_confusion_matrix(y_test, predictions[name]))

    banner("6. SAVE FINAL MODEL")
    train.save_model(fitted[best_name], best_name, X_train)
    print(f"Saved {config.MODEL_PATH.name} ({config.MODEL_PATH.stat().st_size / 1e6:.1f} MB) + metadata")
    print("Final model summary:", summary)

    banner("7. INTERPRETATION (permutation importance, final model)")
    importance = interpret.compute_permutation_importance(fitted[best_name], X_test, y_test)
    importance.round(4).to_csv(config.OUTPUT_DIR / "feature_importance.csv", index=False)
    interpret.plot_importance(importance, best_name)
    print(importance.head(10).round(4))

    banner("8. SAMPLE PREDICTIONS (loaded from the saved model file)")
    print(predict.run_demo().to_string())

    banner("9. SENSITIVITY CHECK: without the two fee-status columns")
    table, pc = run_sensitivity(best_name, X_train, y_train, X_test, y_test, cv_results, test_table)
    print(table)
    print(pc.round(3))

    plt.close("all")
    banner("DONE - see the outputs/ folder")
    sys.stdout = sys.__stdout__
    log.close()


if __name__ == "__main__":
    main()
