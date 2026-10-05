"""
eda.py - exploratory data analysis plots.

Every function:
  * draws a REAL chart from the data it is given (nothing is hard-coded),
  * saves a PNG into outputs/figures/ when a filename is given,
  * returns the matplotlib figure (so a notebook can show it).

In this project the feature-vs-target plots are drawn from the TRAINING split
only, so nothing we "see" in the test set can influence our decisions.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

CLASS_COLORS = {"Dropout": "#c0392b", "Enrolled": "#f39c12", "Graduate": "#27ae60"}
ORDER = config.CLASS_ORDER


def _save(fig, filename):
    if filename:
        config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.FIGURE_DIR / filename, dpi=130, bbox_inches="tight")


def _outcome_percent(df, by):
    """% of each outcome (Dropout/Enrolled/Graduate) within every value of `by`."""
    pct = pd.crosstab(df[by], df[config.TARGET], normalize="index") * 100
    return pct.reindex(columns=ORDER, fill_value=0)


# ---------------------------------------------------------------------------
def plot_target_distribution(y: pd.Series, filename="01_target_distribution.png"):
    counts = y.value_counts().reindex(ORDER)
    pct = counts / counts.sum() * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(ORDER, counts.values, color=[CLASS_COLORS[c] for c in ORDER])
    for bar, n, p in zip(bars, counts.values, pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, n + 25, f"{n}\n({p:.1f}%)",
                ha="center", va="bottom")
    ax.set_ylabel("Number of students")
    ax.set_title("Target distribution (class imbalance check)")
    ax.set_ylim(0, counts.max() * 1.18)
    fig.tight_layout()
    _save(fig, filename)
    return fig


def plot_numeric_by_target(train_df, filename="02_numeric_features_by_target.png"):
    """Distribution of the main numeric predictors, one line per outcome class."""
    cols = ["Admission grade", "Previous qualification (grade)", "Age at enrollment"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, cols):
        upper = train_df[col].quantile(0.995)           # clip display range only
        bins = np.linspace(train_df[col].min(), upper, 30)
        for cls in ORDER:
            vals = train_df.loc[train_df[config.TARGET] == cls, col]
            ax.hist(vals.clip(upper=upper), bins=bins, density=True, histtype="step",
                    linewidth=2, color=CLASS_COLORS[cls], label=cls)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
    axes[0].legend()
    fig.suptitle("Numeric predictors vs outcome (training data)")
    fig.tight_layout()
    _save(fig, filename)
    return fig


def plot_age_group_outcomes(train_df, filename="03_age_group_vs_outcome.png"):
    """Outcome mix by age band (fixed, human-chosen bands - nothing learned)."""
    df = train_df.copy()
    df["Age group"] = pd.cut(df["Age at enrollment"], [0, 19, 24, 29, 100],
                             labels=["<=19", "20-24", "25-29", "30+"])
    pct = _outcome_percent(df, "Age group")
    n = df["Age group"].value_counts().reindex(pct.index)
    fig, ax = plt.subplots(figsize=(7, 4))
    pct.plot(kind="bar", stacked=True, ax=ax, color=[CLASS_COLORS[c] for c in ORDER], width=0.7)
    ax.set_xticklabels([f"{g}\n(n={n[g]})" for g in pct.index], rotation=0)
    ax.set_ylabel("% of students")
    ax.set_xlabel("Age at enrollment")
    ax.set_title("Outcome by age group (training data)")
    ax.legend(title="Outcome", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    _save(fig, filename)
    return fig


def plot_binary_vs_target(train_df, filename="04_binary_features_vs_target.png"):
    """For each 0/1 feature: stacked 100% bars of the outcome for value 0 and value 1."""
    feats = config.BINARY_FEATURES
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax, col in zip(axes.ravel(), feats):
        pct = _outcome_percent(train_df, col)
        n = train_df[col].value_counts()
        pct.plot(kind="bar", stacked=True, ax=ax, legend=False, width=0.7,
                 color=[CLASS_COLORS[c] for c in ORDER])
        ax.set_xticklabels([f"{v}\n(n={n[v]})" for v in pct.index], rotation=0)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("% of students")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Binary (0/1) predictors vs outcome (training data)", fontsize=14, y=0.995)
    fig.legend(handles, labels, loc="upper center", ncol=3, title="Outcome",
               bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    _save(fig, filename)
    return fig


def binary_outcome_table(train_df) -> pd.DataFrame:
    """Dropout rate (%) for value 0 vs value 1 of each binary feature."""
    rows = []
    for col in config.BINARY_FEATURES:
        for value in (0, 1):
            sub = train_df[train_df[col] == value]
            rows.append({"feature": col, "value": value, "n": len(sub),
                         "dropout_%": round(100 * (sub[config.TARGET] == "Dropout").mean(), 1),
                         "graduate_%": round(100 * (sub[config.TARGET] == "Graduate").mean(), 1)})
    return pd.DataFrame(rows)


def plot_categorical_vs_target(train_df, col, min_count=30, filename=None):
    """
    Horizontal stacked bars: outcome mix for every code of a categorical column.
    Codes with fewer than `min_count` students are hidden (too few to be reliable).
    """
    counts = train_df[col].value_counts()
    keep = counts[counts >= min_count].index
    sub = train_df[train_df[col].isin(keep)]
    pct = _outcome_percent(sub, col).sort_values("Dropout")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.38 * len(pct) + 1.5)))
    pct.plot(kind="barh", stacked=True, ax=ax, color=[CLASS_COLORS[c] for c in ORDER], width=0.75)
    ax.set_yticklabels([f"{c} (n={counts[c]})" for c in pct.index])
    ax.set_xlabel("% of students")
    ax.set_ylabel(f"{col} (code)")
    hidden = len(counts) - len(keep)
    ax.set_title(f"Outcome by '{col}' (training data)" +
                 (f"\n{hidden} rare code(s) with <{min_count} students not shown" if hidden else ""))
    ax.legend(title="Outcome", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    _save(fig, filename)
    return fig


def plot_macro_economic_by_target(train_df, filename="08_macro_economic_by_target.png"):
    cols = ["Unemployment rate", "Inflation rate", "GDP"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, cols):
        data = [train_df.loc[train_df[config.TARGET] == c, col] for c in ORDER]
        bp = ax.boxplot(data, patch_artist=True, tick_labels=ORDER)
        for patch, c in zip(bp["boxes"], ORDER):
            patch.set_facecolor(CLASS_COLORS[c])
            patch.set_alpha(0.7)
        ax.set_title(col)
    fig.suptitle("Macro-economic context at enrollment vs outcome (training data)")
    fig.tight_layout()
    _save(fig, filename)
    return fig


def plot_correlation_heatmap(train_df, filename="09_correlation_heatmap.png"):
    """Pearson correlation between numeric + binary predictors (NOT the target)."""
    cols = config.NUMERIC_FEATURES + config.BINARY_FEATURES
    corr = train_df[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=75, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.1f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Correlation between numeric/binary predictors (training data)")
    fig.tight_layout()
    _save(fig, filename)
    return fig


def run_all_eda(train_df):
    """Draw and save every EDA figure. Returns a list of the saved filenames."""
    plot_target_distribution(train_df[config.TARGET], "01_target_distribution_train.png")
    plot_numeric_by_target(train_df)
    plot_age_group_outcomes(train_df)
    plot_binary_vs_target(train_df)
    plot_categorical_vs_target(train_df, "Course", filename="05_course_vs_target.png")
    plot_categorical_vs_target(train_df, "Application mode", filename="06_application_mode_vs_target.png")
    plot_categorical_vs_target(train_df, "Marital status", filename="07_marital_status_vs_target.png")
    plot_macro_economic_by_target(train_df)
    plot_correlation_heatmap(train_df)
    plt.close("all")
