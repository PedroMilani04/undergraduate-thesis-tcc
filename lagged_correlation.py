"""
Lagged Autocorrelation Analysis
================================
Plots the lagged autocorrelation (ACF) for lags 5, 21, and 63 trading days
for each feature selected by RFE in the 3-class generalista model.

Features are read from:
    4-modelos-generalistas/xgboost/rfe/RFE-features_treino_teste_3_classes.csv
Only the training set rows are used (Split == 'Treino').
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.graphics.tsaplots import plot_acf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(
    BASE_DIR,
    "4-modelos-generalistas", "xgboost", "rfe",
    "RFE-features_treino_teste_3_classes.csv"
)
OUTPUT_DIR = os.path.join(BASE_DIR, "lagged_correlation_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

# Keep only the training split to avoid look-ahead bias
df_train = df[df["Split"] == "Treino"].copy().reset_index(drop=True)

# Columns that are NOT features
NON_FEATURES = {"Target_Real", "Target_Previsto", "Split"}
feature_cols = [c for c in df.columns if c not in NON_FEATURES]

LAGS = [5, 21, 63]
MAX_LAG = max(LAGS)

print(f"Features to analyse: {feature_cols}")
print(f"Training rows: {len(df_train)}")

# ---------------------------------------------------------------------------
# Helper – compute autocorrelation at a specific lag
# ---------------------------------------------------------------------------

def acf_at_lags(series: pd.Series, lags: list[int]) -> dict[int, float]:
    """Return the sample autocorrelation at each requested lag."""
    s = series.dropna().values.astype(float)
    n = len(s)
    mean = s.mean()
    var = ((s - mean) ** 2).sum()
    if var == 0:
        return {lag: np.nan for lag in lags}
    result = {}
    for lag in lags:
        if lag >= n:
            result[lag] = np.nan
        else:
            cov = ((s[lag:] - mean) * (s[:-lag] - mean)).sum()
            result[lag] = cov / var
    return result

# ---------------------------------------------------------------------------
# Plot 1 – Full ACF plot for each feature (up to MAX_LAG lags)
#           with vertical lines at lags 5, 21, 63
# ---------------------------------------------------------------------------

n_features = len(feature_cols)
COLS = 3
ROWS = int(np.ceil(n_features / COLS))

fig, axes = plt.subplots(
    ROWS, COLS,
    figsize=(6 * COLS, 4 * ROWS),
    constrained_layout=True,
)
fig.suptitle(
    "Lagged Autocorrelation – RFE Features (training set)\n"
    "Highlighted lags: 5, 21, 63 (trading days)",
    fontsize=14, fontweight="bold"
)

axes_flat = axes.flatten()

HIGHLIGHT_COLORS = {5: "#e63946", 21: "#2a9d8f", 63: "#e9c46a"}

for ax_idx, col in enumerate(feature_cols):
    ax = axes_flat[ax_idx]
    series = df_train[col]

    # Draw the ACF up to MAX_LAG lags (statsmodels)
    plot_acf(
        series.dropna(),
        lags=MAX_LAG,
        alpha=0.05,
        ax=ax,
        title="",
        zero=False,
        auto_ylims=True,
        color="#457b9d",
    )

    # Overlay vertical lines at requested lags
    acf_vals = acf_at_lags(series, LAGS)
    for lag, color in HIGHLIGHT_COLORS.items():
        acf_val = acf_vals[lag]
        ax.axvline(x=lag, color=color, linewidth=1.8,
                   linestyle="--", alpha=0.85,
                   label=f"lag={lag}: r={acf_val:.3f}")

    ax.set_title(col, fontsize=10, fontweight="bold", pad=4)
    ax.set_xlabel("Lag (trading days)", fontsize=8)
    ax.set_ylabel("ACF", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.axhline(0, color="black", linewidth=0.8)

# Hide unused axes
for ax_idx in range(n_features, len(axes_flat)):
    axes_flat[ax_idx].set_visible(False)

out_path_full = os.path.join(OUTPUT_DIR, "acf_full_all_features.png")
fig.savefig(out_path_full, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path_full}")

# ---------------------------------------------------------------------------
# Plot 2 – Bar chart of ACF values at lags 5, 21, 63 for every feature
# ---------------------------------------------------------------------------

acf_records = []
for col in feature_cols:
    vals = acf_at_lags(df_train[col], LAGS)
    for lag, val in vals.items():
        acf_records.append({"Feature": col, "Lag": f"lag={lag}", "ACF": val})

acf_df = pd.DataFrame(acf_records)

fig2, ax2 = plt.subplots(figsize=(max(14, n_features * 1.4), 6), constrained_layout=True)

bar_colors = {f"lag={l}": c for l, c in HIGHLIGHT_COLORS.items()}
lag_labels = [f"lag={l}" for l in LAGS]

x = np.arange(n_features)
width = 0.25

for i, lag_label in enumerate(lag_labels):
    subset = acf_df[acf_df["Lag"] == lag_label]
    vals = subset.set_index("Feature").reindex(feature_cols)["ACF"].values
    bars = ax2.bar(
        x + (i - 1) * width,
        vals,
        width=width,
        label=lag_label,
        color=bar_colors[lag_label],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )

ax2.set_xticks(x)
ax2.set_xticklabels(feature_cols, rotation=40, ha="right", fontsize=9)
ax2.axhline(0, color="black", linewidth=0.9)
ax2.set_ylabel("Autocorrelation", fontsize=11)
ax2.set_title(
    "Autocorrelation at Lags 5, 21, 63 – RFE Features (training set)",
    fontsize=13, fontweight="bold"
)
ax2.legend(fontsize=10)
ax2.grid(axis="y", linestyle="--", alpha=0.5)
ax2.set_ylim(-1.05, 1.05)

out_path_bar = os.path.join(OUTPUT_DIR, "acf_bar_lags_5_21_63.png")
fig2.savefig(out_path_bar, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {out_path_bar}")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
pivot = acf_df.pivot(index="Feature", columns="Lag", values="ACF").reindex(feature_cols)
pivot.columns.name = None
print("\n=== ACF Values ===")
print(pivot.to_string(float_format="{:.4f}".format))
