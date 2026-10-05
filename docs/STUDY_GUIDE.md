# Study guide - how to explain this project in the verification round

Read this once, then open each file next to its section below. You should be able to say *what* each part does and *why*.

## 1. The 60-second story

> "I predict whether a student will Drop out, stay Enrolled or Graduate, using only information known at enrollment. I removed the 12 semester-performance columns because they are not known at enrollment - using them would be leakage. I split off a stratified 20% test set first. All preprocessing sits inside a scikit-learn Pipeline, so it is learned from training data only. I compared five models with 5-fold cross-validation on the training set and chose the one with the best Macro F1, Gradient Boosting. On the untouched test set it scores a Macro F1 of about 0.55. It works reasonably for Graduate and Dropout but poorly for Enrolled. Feature importance shows fee status, course and age matter most, but that is association, not causation."

## 2. File-by-file walkthrough

| File | What it does | One-line explanation |
|---|---|---|
| `src/config.py` | Seed, paths, the lists of numeric / binary / categorical / prohibited columns | "One place for every setting." |
| `src/data_loader.py` | `pd.read_csv(sep=";")`, checks the columns, prints rows/missing/duplicates/classes | "The file is semicolon-delimited." |
| `src/preprocessing.py` | `split_features_target` (picks the 24 allowed columns), `make_train_test_split`, `build_preprocessor` (ColumnTransformer), `build_pipeline` | "Preprocessing + model are one Pipeline object." |
| `src/models.py` | The five candidate models | "Dummy floor, two Logistic Regressions, Random Forest, Gradient Boosting." |
| `src/eda.py` | All EDA charts (from the training split) | "I never looked at test data." |
| `src/evaluate.py` | Accuracy, Macro P/R/F1, per-class table, confusion-matrix plots | "Macro F1 weights all 3 classes equally." |
| `src/train.py` | Cross-validation, fit + test evaluation, model selection, saving | "Select on CV, report on test once." |
| `src/interpret.py` | Permutation importance | "Shuffle a column, see how much Macro F1 drops." |
| `src/predict.py` | Load saved model, predict, demo | "The saved Pipeline repeats the training preprocessing automatically." |
| `src/sensitivity.py` | Re-run the chosen model without the 2 fee columns | "Checks a possible timing issue." |
| `run_all.py` | Calls all of the above in order, logs to `outputs/run_log.txt` | "One command reproduces everything." |
| `tests/test_pipeline.py` | 10 tests of the leakage rules | "Proves the rules in code." |

## 3. Key ideas in plain English

- **Data leakage:** using information the model would not have at prediction time (or using the test set while building the model), which makes scores look better than reality. Here: (a) semester columns, (b) fitting the scaler/encoder on test data, (c) choosing models by test score.
- **Pipeline + ColumnTransformer:** `pipeline.fit(X_train)` learns medians, means, std and the one-hot category lists from training rows only; `pipeline.predict(X_test)` only *applies* them. Inside cross-validation sklearn refits the whole pipeline in every fold.
- **`stratify=y`:** each class keeps the same percentage in train and test.
- **One-hot encoding of codes:** `Course = 9500` is not "bigger" than `171`; one-hot gives each code its own 0/1 column. `handle_unknown="ignore"` stops a crash on an unseen code.
- **StandardScaler:** puts numeric columns on a common scale; needed for Logistic Regression, harmless for trees.
- **Precision / Recall / F1:** precision = of those predicted "Enrolled", how many really are; recall = of the true Enrolled, how many were found; F1 = their harmonic mean. **Macro F1** = plain average of the three class F1s.
- **Why not accuracy:** ~50% of students are Graduates, so "always Graduate" gets ~50% accuracy. Plain Logistic Regression had the best accuracy but found only 13% of Enrolled students.
- **`class_weight="balanced"`:** mistakes on rarer classes cost more *during training*. No rows are copied or deleted and the test set is untouched.
- **Cross-validation:** split the training set into 5 parts; train on 4, validate on 1, rotate; average. It gives a more stable score than one split and keeps the test set clean.
- **Permutation importance:** model-agnostic; measures reliance on a feature, **not** causation.

## 4. Likely questions and good answers

1. **Why did you exclude the semester columns?** They describe performance *after* enrollment. The task is enrollment-time prediction, so they would leak the future (they are also very strong predictors, which would inflate the score).
2. **How do you know there is no leakage in preprocessing?** Everything learned is inside the Pipeline and fitted on `X_train` only. `tests/test_pipeline.py` checks that the scaler's mean equals the *training* mean, not the test or full-data mean, and is unchanged by `transform`.
3. **Why is the model choice based on cross-validation rather than test scores?** Choosing by test score would make the test set part of training decisions and the final score optimistic. The test set is used once, for reporting.
4. **On the test set, Logistic Regression has a slightly higher Macro F1 than your final model. Why not switch?** The gap is ~0.003, which is noise for 885 students; switching after seeing test results would break the protocol. The rule (best CV Macro F1) was fixed in advance. (Be ready to say Logistic Regression is a perfectly reasonable, simpler alternative.)
5. **Why is Enrolled so hard?** It is the smallest class and "still enrolled" lies between the other two outcomes; enrollment-time information says little about it. (Say this is a plausible explanation, not a proven one.)
6. **What does the confusion matrix tell you?** Rows are true classes. For the final model, 114 Graduates were predicted Enrolled and 70 Dropouts were predicted Enrolled: the model over-predicts Enrolled because of the class weights, which is the trade-off for catching more true Enrolled students.
7. **What would you change if recall on Dropout mattered most?** Choose by Dropout recall or adjust class weights/thresholds - but that must be decided on training/validation data, never the test set.
8. **Are the fee columns a problem?** Possibly: fee status can change after enrollment. They are allowed by the assignment, so they are in the primary model; the sensitivity check shows CV Macro F1 falls from 0.582 to 0.529 without them.
9. **Does "Tuition fees up to date" cause dropout?** Not shown. It is an association; unpaid fees may be a symptom of disengagement.
10. **Why is `Unemployment rate` important?** It only has 10 distinct values, so it probably identifies the enrollment cohort. That limits generalisation to other years.
11. **What would you do next?** Tune hyperparameters with cross-validation, try calibration, add SHAP, audit fairness (age, gender, nationality are inputs), and check whether fee columns are truly available at enrollment.
12. **Is 0.55 Macro F1 good?** It is far above the dummy (0.22) but only moderate; the model should support, not replace, human judgement.

## 5. "Change it live" cheat-sheet

| Request | Where |
|---|---|
| Different random seed / test size / CV folds | `src/config.py` (`RANDOM_SEED`, `TEST_SIZE`, `CV_FOLDS`) |
| Add another model | add an entry in `src/models.py -> get_models()` (e.g. `KNeighborsClassifier()`); re-run `python run_all.py` |
| Remove a feature | delete its name from the matching list in `src/config.py` (`FEATURE_COLUMNS` is built from those lists automatically) |
| Change a Random Forest setting | `src/models.py`, e.g. `n_estimators` or `min_samples_leaf` |
| Select by a different metric | `src/train.py -> select_best_model` and the `scoring` dict in `cross_validate_models` |
| Show another EDA chart | call `eda.plot_categorical_vs_target(train_df, "Mother's occupation")` in the notebook |
| Predict for a new student | build a one-row DataFrame with the 24 columns and call `predict.predict_students(predict.load_model(), df)` |

After any change, **re-run `python run_all.py`** and **update the numbers in README.md** (they are copied from `outputs/`).

## 6. Publishing to GitHub (from the project folder)

```bash
git init
git add .
git commit -m "Student outcome prediction - Gupio assignment (Option 1)"
git branch -M main
# create an EMPTY repository on github.com first, then:
git remote add origin https://github.com/<your-username>/student-outcome-prediction.git
git push -u origin main
```

Then make sure the repository is public (or shared with the Gupio panel), open it in a private window to check README images load, and paste the URL in the Google Form.
