"""
research_plots.py
=================
Publication-quality research visualisations for the FWI Wildfire Prediction paper.
Generates all figures and saves them to research_plots/ as high-resolution PNGs.

Figures produced
----------------
1. feature_importance.png   — Top-20 feature importances (Random Forest proxy)
2. correlation_heatmap.png  — Pearson correlation matrix of all 35 features
3. spatial_fire_density.png — 2-D spatial heatmap of fire event density
4. fwi_distribution.png     — FWI distribution: Fire vs No-Fire (violin + KDE)
5. temporal_fire_trend.png  — Monthly fire event counts, 2015–2020

Usage
-----
    cd Novel_Model
    python research_plots.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        120,
    "savefig.dpi":       300,
    "font.size":         11,
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = r"c:\Users\Admin\Desktop\Projects\FWI_Index_India"
DATA_PATH   = os.path.join(BASE_DIR, "Main_Data", "FLAGED_data.csv")
OUT_DIR     = os.path.join(BASE_DIR, "Novel_Model", "research_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# Colour palette (consistent with model evaluation plots)
C_FIRE     = "#E63946"   # crimson  — fire events
C_NOFIRE   = "#457B9D"   # steel    — no-fire events
C_ACCENT   = "#2A9D8F"   # teal     — accent
CMAP_FIRE  = LinearSegmentedColormap.from_list("fire_map",
             ["#0d0d0d", "#3a0000", "#E63946", "#FF9800", "#FFEB3B"])

# ---------------------------------------------------------------------------
# Load and engineer data
# ---------------------------------------------------------------------------
print("[Data] Loading FLAGED_data.csv …")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

print("[Data] Engineering features …")
df["temp_humidity_interaction"]      = df["temp"] * df["humidity"] / 100.0
df["wind_temp_interaction"]          = df["wind_speed"] * (df["temp"] + 273.15)
df["fwi_temp_ratio"]                 = df["FWI"] / (df["temp"] + 50.0)
df["humidity_rainfall_interaction"]  = df["humidity"] * (df["rainfall"] + 0.001)
df["temp_squared"]                   = df["temp"] ** 2
df["fwi_squared"]                    = df["FWI"] ** 2
df["wind_speed_squared"]             = df["wind_speed"] ** 2
df["humidity_squared"]               = df["humidity"] ** 2
df["dryness_index"]                  = (100 - df["humidity"]) * (df["temp"] + 10) / 100.0
df["fire_danger_index"]              = (
    df["FWI"] * (100 - df["humidity"]) * (df["temp"] + 10) / (df["rainfall"] + 1)
)
df["wind_dryness"]                   = df["wind_speed"] * (100 - df["humidity"])
df["temp_deviation"]                 = np.abs(df["temp"] - df["temp"].mean())
df["fwi_humidity_ratio"]             = df["FWI"] / (df["humidity"] + 1)
df["temp_rainfall_ratio"]            = df["temp"] / (df["rainfall"] + 0.1)
df["wind_humidity_ratio"]            = df["wind_speed"] / (df["humidity"] + 1)
df["log_fwi"]                        = np.log1p(df["FWI"])

FEATURE_COLS = [
    "temp", "humidity", "wind_speed", "rainfall",
    "cvh", "lai_hv", "lai_lv", "cvl", "tvh", "tvl",
    "FFMC", "DMC", "DC", "ISI", "BUI", "FWI",
    "temp_humidity_interaction", "wind_temp_interaction", "fwi_temp_ratio",
    "humidity_rainfall_interaction", "temp_squared", "fwi_squared",
    "wind_speed_squared", "humidity_squared", "dryness_index",
    "fire_danger_index", "wind_dryness", "temp_deviation",
    "fwi_humidity_ratio", "temp_rainfall_ratio", "wind_humidity_ratio",
    "log_fwi",
]
# Keep only columns that exist
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
TARGET_COL = "fire_flag"

print(f"[Data] Loaded {len(df):,} rows | {df[TARGET_COL].mean()*100:.2f}% fire rate")


# ---------------------------------------------------------------------------
# Figure 1 — Feature Importance (Random Forest)
# ---------------------------------------------------------------------------
def plot_feature_importance():
    print("[Fig 1] Computing feature importances via Random Forest …")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    sample = df.sample(n=min(80_000, len(df)), random_state=42)
    X = sample[FEATURE_COLS].fillna(0).values
    y = sample[TARGET_COL].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X_sc, y)
    importances = rf.feature_importances_

    feat_df = pd.DataFrame({
        "Feature":    FEATURE_COLS,
        "Importance": importances,
    }).sort_values("Importance", ascending=True).tail(20)

    bar_colors = [C_FIRE if imp > feat_df["Importance"].median() else C_NOFIRE
                  for imp in feat_df["Importance"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(feat_df["Feature"], feat_df["Importance"],
                   color=bar_colors, edgecolor="white", height=0.65, zorder=3)

    for bar, val in zip(bars, feat_df["Importance"]):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=9, fontweight="bold")

    ax.set_xlabel("Mean Decrease in Impurity (Feature Importance)", fontsize=12)
    ax.set_title(
        "Feature Importance — Random Forest\n"
        "Wildfire Prediction | Indian Subcontinent (2015–2020)",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.set_xlim(0, feat_df["Importance"].max() * 1.18)

    legend_elements = [
        Patch(facecolor=C_FIRE,   label="High importance (above median)"),
        Patch(facecolor=C_NOFIRE, label="Moderate importance"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="lower right")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Fig 1] Saved → {path}")


# ---------------------------------------------------------------------------
# Figure 2 — Correlation Heatmap (FWI system + key meteorological)
# ---------------------------------------------------------------------------
def plot_correlation_heatmap():
    print("[Fig 2] Computing correlation matrix …")
    # Focus on the most interpretable subset for the paper
    HMAP_COLS = [
        "temp", "humidity", "wind_speed", "rainfall",
        "FFMC", "DMC", "DC", "ISI", "BUI", "FWI",
        "dryness_index", "fire_danger_index", "log_fwi",
        "temp_humidity_interaction", "fwi_temp_ratio",
        TARGET_COL,
    ]
    hmap_cols = [c for c in HMAP_COLS if c in df.columns]
    corr = df[hmap_cols].corr()

    nice_labels = {
        "temp":                       "Temperature",
        "humidity":                   "Humidity",
        "wind_speed":                 "Wind Speed",
        "rainfall":                   "Rainfall",
        "FFMC":                       "FFMC",
        "DMC":                        "DMC",
        "DC":                         "Drought Code",
        "ISI":                        "ISI",
        "BUI":                        "Buildup Index",
        "FWI":                        "FWI",
        "dryness_index":              "Dryness Index",
        "fire_danger_index":          "Fire Danger Index",
        "log_fwi":                    "log(FWI)",
        "temp_humidity_interaction":  "Temp × Humidity",
        "fwi_temp_ratio":             "FWI / Temp",
        TARGET_COL:                   "Fire Flag ★",
    }
    corr.index   = [nice_labels.get(c, c) for c in corr.index]
    corr.columns = [nice_labels.get(c, c) for c in corr.columns]

    fig, ax = plt.subplots(figsize=(13, 11))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        corr, ax=ax, cmap=cmap, center=0,
        vmin=-1, vmax=1, annot=True, fmt=".2f",
        annot_kws={"size": 8}, linewidths=0.5,
        linecolor="#222", square=True, cbar_kws={"shrink": 0.8},
    )
    ax.set_title(
        "Pearson Correlation Matrix — FWI System & Meteorological Features\n"
        "Wildfire Prediction | Indian Subcontinent",
        fontsize=13, fontweight="bold", pad=16,
    )
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "correlation_heatmap.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Fig 2] Saved → {path}")


# ---------------------------------------------------------------------------
# Figure 3 — Spatial Fire Density Map
# ---------------------------------------------------------------------------
def plot_spatial_fire_density():
    print("[Fig 3] Building spatial fire density map …")
    fire_df = df[df[TARGET_COL] == 1]
    density = (
        fire_df.groupby(["latitude", "longitude"])
        .size()
        .reset_index(name="fire_count")
    )

    fig, ax = plt.subplots(figsize=(11, 9))

    # Background grid — all cells
    all_grid = df.groupby(["latitude", "longitude"]).size().reset_index(name="n")
    ax.scatter(
        all_grid["longitude"], all_grid["latitude"],
        s=60, color="#1a1a1a", alpha=0.4, zorder=1, label="Grid cell (no fire)",
    )

    # Fire density scatter
    sc = ax.scatter(
        density["longitude"], density["latitude"],
        c=density["fire_count"],
        s=density["fire_count"] / density["fire_count"].max() * 600 + 20,
        cmap=CMAP_FIRE, alpha=0.88, edgecolors="white", linewidths=0.4,
        zorder=3,
    )

    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Total Fire Days (2015–2020)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)
    ax.set_title(
        "Spatial Distribution of Wildfire Events\n"
        "Indian Subcontinent Grid (2015–2020) | MODIS FIRMS",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.15)

    legend_elements = [
        plt.scatter([], [], s=60,  color="#1a1a1a", alpha=0.5, label="No-fire grid cell"),
        plt.scatter([], [], s=200, color=C_FIRE,    alpha=0.85, label="High fire density"),
        plt.scatter([], [], s=60,  color="#FF9800", alpha=0.85, label="Moderate fire density"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper left",
              framealpha=0.85, edgecolor="#444")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "spatial_fire_density.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Fig 3] Saved → {path}")


# ---------------------------------------------------------------------------
# Figure 4 — FWI Distribution: Fire vs No-Fire (Violin + Strip)
# ---------------------------------------------------------------------------
def plot_fwi_distribution():
    print("[Fig 4] Building FWI distribution comparison …")

    fwi_cols  = ["FWI", "FFMC", "DMC", "DC", "ISI", "BUI"]
    fwi_exist = [c for c in fwi_cols if c in df.columns]

    # Sample for speed
    fire_s   = df[df[TARGET_COL] == 1].sample(n=min(5000, df[TARGET_COL].sum()), random_state=42)
    nofire_s = df[df[TARGET_COL] == 0].sample(n=min(5000, (df[TARGET_COL]==0).sum()), random_state=42)
    plot_df  = pd.concat([
        fire_s[fwi_exist + [TARGET_COL]].assign(Class="Fire"),
        nofire_s[fwi_exist + [TARGET_COL]].assign(Class="No Fire"),
    ])

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    palette = {"Fire": C_FIRE, "No Fire": C_NOFIRE}

    for ax, col in zip(axes, fwi_exist):
        sns.violinplot(
            data=plot_df, x="Class", y=col, palette=palette,
            inner="quartile", linewidth=1.2, ax=ax,
        )
        ax.set_title(col, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(col, fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(axis="y", alpha=0.2)

        # Annotate medians
        for i, cls in enumerate(["No Fire", "Fire"]):
            med = plot_df[plot_df["Class"] == cls][col].median()
            ax.text(i, med, f"  {med:.1f}", va="center", fontsize=9,
                    color="white", fontweight="bold")

    # Hide unused subplots
    for ax in axes[len(fwi_exist):]:
        ax.set_visible(False)

    fig.suptitle(
        "FWI System Component Distributions — Fire vs No-Fire Days\n"
        "Indian Subcontinent (2015–2020) | Median annotated",
        fontsize=13, fontweight="bold", y=1.01,
    )
    legend_elements = [
        Patch(facecolor=C_FIRE,   label="Fire days (fire_flag = 1)"),
        Patch(facecolor=C_NOFIRE, label="No-Fire days (fire_flag = 0)"),
    ]
    fig.legend(handles=legend_elements, fontsize=11, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.03), framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fwi_distribution.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Fig 4] Saved → {path}")


# ---------------------------------------------------------------------------
# Figure 5 — Temporal Fire Trend (Monthly, 2015–2020)
# ---------------------------------------------------------------------------
def plot_temporal_trend():
    print("[Fig 5] Building temporal fire trend …")
    df["year_month"] = df["date"].dt.to_period("M")
    monthly = (
        df.groupby("year_month")[TARGET_COL]
        .agg(["sum", "count"])
        .rename(columns={"sum": "fire_events", "count": "total_obs"})
        .reset_index()
    )
    monthly["fire_rate"] = monthly["fire_events"] / monthly["total_obs"] * 100
    monthly["year_month_dt"] = monthly["year_month"].dt.to_timestamp()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(
        "Temporal Fire Activity — Indian Subcontinent (2015–2020)\n"
        "MODIS FIRMS Active Fire Detections",
        fontsize=13, fontweight="bold",
    )

    # Panel 1: Fire event count
    ax1.fill_between(monthly["year_month_dt"], monthly["fire_events"],
                     alpha=0.35, color=C_FIRE)
    ax1.plot(monthly["year_month_dt"], monthly["fire_events"],
             lw=2, color=C_FIRE, label="Monthly fire events")
    ax1.set_ylabel("Fire Events (grid-cell days)", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.2)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Annotate year boundaries
    for year in range(2015, 2021):
        ax1.axvline(pd.Timestamp(f"{year}-01-01"), color="#555",
                    lw=0.8, linestyle="--", alpha=0.6)
        ax1.text(pd.Timestamp(f"{year}-06-15"), ax1.get_ylim()[1] * 0.92,
                 str(year), ha="center", fontsize=9, color="#aaa")

    # Panel 2: Fire rate %
    ax2.fill_between(monthly["year_month_dt"], monthly["fire_rate"],
                     alpha=0.35, color=C_ACCENT)
    ax2.plot(monthly["year_month_dt"], monthly["fire_rate"],
             lw=2, color=C_ACCENT, label="Monthly fire rate (%)")
    ax2.axhline(df[TARGET_COL].mean() * 100, color="#888",
                linestyle=":", lw=1.5, label=f"Overall mean ({df[TARGET_COL].mean()*100:.2f}%)")
    ax2.set_ylabel("Fire Rate (%)", fontsize=11)
    ax2.set_xlabel("Month", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.2)

    for year in range(2015, 2021):
        ax2.axvline(pd.Timestamp(f"{year}-01-01"), color="#555",
                    lw=0.8, linestyle="--", alpha=0.6)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "temporal_fire_trend.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Fig 5] Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Research Plots — FWI Wildfire Prediction")
    print("  Indian Subcontinent | Journal Publication")
    print("="*60)

    plot_correlation_heatmap()   # Fast (no model training)
    plot_spatial_fire_density()  # Fast
    plot_fwi_distribution()      # Fast
    plot_temporal_trend()        # Fast
    plot_feature_importance()    # Slower (trains RF on 80k samples)

    print("\n" + "="*60)
    print(f"  All figures saved to: {OUT_DIR}")
    print("="*60)
