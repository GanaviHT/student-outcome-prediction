"""
config.py - ONE place for every setting used in the project.

If you want to change the random seed, the test size, or the list of
features, change it here and every other file picks it up automatically.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths (all relative to the project root, so it works on any computer)
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

# ----------------------------------------------------------------------------
# Reproducibility / experiment settings
# ----------------------------------------------------------------------------
RANDOM_SEED = 42      # used everywhere a random number is needed
TEST_SIZE = 0.20      # 20% of students are locked away as the final test set
CV_FOLDS = 5          # stratified K-fold on the TRAINING data for model selection

# ----------------------------------------------------------------------------
# Target
# ----------------------------------------------------------------------------
TARGET = "Target"
CLASS_ORDER = ["Dropout", "Enrolled", "Graduate"]   # fixed order for reports/plots

# ----------------------------------------------------------------------------
# PROHIBITED columns (assignment rule): semester-performance information is NOT
# available at enrollment time, so using it would be data leakage for an
# "enrollment-time" prediction model.
# ----------------------------------------------------------------------------
PROHIBITED_COLUMNS = [
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
]

# ----------------------------------------------------------------------------
# The 24 PERMITTED enrollment-time predictors, grouped by how we treat them.
# (36 UCI features - 12 prohibited = 24)
# ----------------------------------------------------------------------------

# Real numbers -> median-impute + StandardScaler
NUMERIC_FEATURES = [
    "Application order",              # 0-9, a preference rank (ordered, so numeric)
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

# Already 0/1 flags -> passed through (no scaling or encoding needed)
BINARY_FEATURES = [
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

# Integer CODES that stand for categories (e.g. a course code, a marital-status
# code). The numbers have NO order/size meaning, so they are one-hot encoded.
CATEGORICAL_FEATURES = [
    "Marital status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",              # (sic) - spelling is from the original UCI data
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Used ONLY by the sensitivity analysis (src/sensitivity.py). These two columns
# are allowed by the assignment, but they may be recorded AFTER enrollment
# (a student can fall behind on fees later), so we check how much the model
# depends on them.
POSSIBLY_LATE_FEATURES = ["Debtor", "Tuition fees up to date"]
