"""
data_loader.py - load the CSV and describe it.

Note: the file is SEMICOLON-delimited, so we must pass sep=";" to pandas.
"""
import pandas as pd

from src import config


def load_data(path=config.DATA_PATH) -> pd.DataFrame:
    """Read data.csv (semicolon-delimited) and strip stray spaces from headers."""
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()
    return df


def check_columns(df: pd.DataFrame) -> None:
    """Fail early with a clear message if the file is not what we expect."""
    expected = set(config.FEATURE_COLUMNS) | set(config.PROHIBITED_COLUMNS) | {config.TARGET}
    missing = expected - set(df.columns)
    extra = set(df.columns) - expected
    if missing or extra:
        raise ValueError(f"Unexpected columns. Missing: {sorted(missing)} | Extra: {sorted(extra)}")


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Print and return the basic facts the assignment asks for."""
    info = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_counts": df[config.TARGET].value_counts().to_dict(),
        "target_percent": (df[config.TARGET].value_counts(normalize=True) * 100).round(2).to_dict(),
        "dtype_counts": df.dtypes.astype(str).value_counts().to_dict(),
    }
    print(f"Rows: {info['rows']}   Columns: {info['columns']}")
    print(f"Missing values (total): {info['missing_values_total']}")
    print(f"Duplicate rows: {info['duplicate_rows']}")
    print(f"Data types: {info['dtype_counts']}")
    print("Target classes (count):", info["target_counts"])
    print("Target classes (%):    ", info["target_percent"])
    return info


def column_overview(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, role in our project, #unique values, #missing."""
    def role(col):
        if col == config.TARGET:
            return "target"
        if col in config.PROHIBITED_COLUMNS:
            return "PROHIBITED (semester performance)"
        if col in config.NUMERIC_FEATURES:
            return "numeric"
        if col in config.BINARY_FEATURES:
            return "binary (0/1)"
        if col in config.CATEGORICAL_FEATURES:
            return "categorical (integer code)"
        return "unknown"

    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "role": [role(c) for c in df.columns],
        "n_unique": df.nunique(),
        "n_missing": df.isna().sum(),
    })


def data_quality_notes(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric facts behind the data-quality observations written in the README."""
    num = df[config.NUMERIC_FEATURES].describe().T[["min", "50%", "max"]]
    num.columns = ["min", "median", "max"]
    return num.round(2)


def rare_category_table(df: pd.DataFrame, min_count: int = 10) -> pd.DataFrame:
    """For each categorical-code column: how many codes and how many are rare?"""
    rows = []
    for col in config.CATEGORICAL_FEATURES:
        counts = df[col].value_counts()
        rows.append({
            "column": col,
            "n_codes": int(counts.size),
            f"codes_with_<{min_count}_students": int((counts < min_count).sum()),
            "largest_code_share_%": round(100 * counts.iloc[0] / len(df), 1),
        })
    return pd.DataFrame(rows).set_index("column")
