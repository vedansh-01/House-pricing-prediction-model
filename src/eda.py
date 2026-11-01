"""
eda.py — Exploratory Data Analysis script (Phase 1).
Outputs charts to data/processed/eda/ and prints a structured summary.
Replaces notebook EDA with a reproducible script.

Usage:
    python -m src.eda
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import skew

from src.config import TARGET_COL, TRAIN_CSV

warnings.filterwarnings("ignore")

EDA_OUT = Path(__file__).resolve().parent.parent / "data" / "processed" / "eda"


def run_eda() -> None:
    EDA_OUT.mkdir(parents=True, exist_ok=True)

    # ── Load ────────────────────────────────────────────────────────────────────
    df = pd.read_csv(TRAIN_CSV, index_col="Id")
    print(f"\n{'='*60}")
    print(f"  EDA — House Price Dataset")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"Target range: ${df[TARGET_COL].min():,.0f} – ${df[TARGET_COL].max():,.0f}")
    print(f"Target median: ${df[TARGET_COL].median():,.0f}")

    # ── Missing values ──────────────────────────────────────────────────────────
    missing = df.isnull().sum().sort_values(ascending=False)
    missing_pct = (missing / len(df) * 100).round(1)
    miss_df = pd.DataFrame({"missing": missing, "pct": missing_pct})
    miss_df = miss_df[miss_df["missing"] > 0]
    print(f"\nMissing values ({len(miss_df)} columns with missings):")
    print(miss_df.head(20).to_string())

    # ── Target distribution ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[TARGET_COL], bins=50, ax=axes[0], color="#4C72B0")
    axes[0].set_title("SalePrice Distribution (raw)")
    axes[0].set_xlabel("Sale Price ($)")
    sns.histplot(np.log1p(df[TARGET_COL]), bins=50, ax=axes[1], color="#55A868")
    axes[1].set_title("log1p(SalePrice) Distribution")
    axes[1].set_xlabel("log(Sale Price + 1)")
    plt.tight_layout()
    fig.savefig(EDA_OUT / "target_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    skewness = skew(df[TARGET_COL])
    log_skewness = skew(np.log1p(df[TARGET_COL]))
    print(f"\nTarget skewness:        {skewness:.3f}")
    print(f"log1p(target) skewness: {log_skewness:.3f}  ← near-normal ✓")

    # ── Outlier detection ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    scatter_cols = ["GrLivArea", "TotalBsmtSF", "LotArea"]
    for ax, col in zip(axes, scatter_cols):
        if col in df.columns:
            ax.scatter(df[col], df[TARGET_COL], alpha=0.4, s=15, color="#4C72B0", edgecolors="none")
            ax.set_xlabel(col)
            ax.set_ylabel("SalePrice")
            ax.set_title(f"{col} vs SalePrice")
    plt.tight_layout()
    fig.savefig(EDA_OUT / "outlier_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Identify outlier candidates (GrLivArea > 4000, SalePrice < 300k)
    outliers = df[(df["GrLivArea"] > 4000) & (df[TARGET_COL] < 300_000)]
    print(f"\nOutlier candidates (GrLivArea>4000, Price<300k): {len(outliers)} rows")
    if len(outliers):
        print(outliers[[TARGET_COL, "GrLivArea"]].to_string())

    # ── Correlation heatmap (top 15 features) ──────────────────────────────────
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()[[TARGET_COL]].drop(TARGET_COL).abs().sort_values(TARGET_COL, ascending=False)
    top15 = corr.head(15).index.tolist()

    fig, ax = plt.subplots(figsize=(12, 10))
    corr_matrix = num_df[top15 + [TARGET_COL]].corr()
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, ax=ax, annot_kws={"size": 9}
    )
    ax.set_title("Top 15 Features — Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(EDA_OUT / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nTop 15 correlated features with SalePrice:")
    print(corr.head(15).to_string())

    # ── Neighborhood boxplot ────────────────────────────────────────────────────
    if "Neighborhood" in df.columns:
        order = df.groupby("Neighborhood")[TARGET_COL].median().sort_values().index
        fig, ax = plt.subplots(figsize=(16, 6))
        sns.boxplot(
            data=df, x="Neighborhood", y=TARGET_COL,
            order=order, ax=ax, palette="viridis", width=0.6
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        ax.set_title("SalePrice by Neighborhood", fontsize=13, fontweight="bold")
        ax.set_ylabel("Sale Price ($)")
        plt.tight_layout()
        fig.savefig(EDA_OUT / "neighborhood_boxplot.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Cardinality report ──────────────────────────────────────────────────────
    cat_cols = df.select_dtypes(include="object").columns
    card_df = pd.DataFrame({
        "unique_values": df[cat_cols].nunique(),
        "missing_pct": (df[cat_cols].isnull().sum() / len(df) * 100).round(1),
    }).sort_values("unique_values", ascending=False)
    print(f"\nCategorical column cardinality:")
    print(card_df.to_string())

    # ── Multicollinear pairs ────────────────────────────────────────────────────
    corr_full = num_df.corr().abs()
    upper = corr_full.where(np.triu(np.ones(corr_full.shape), k=1).astype(bool))
    high_corr_pairs = [
        (col, row, round(upper.loc[row, col], 3))
        for col in upper.columns
        for row in upper.index
        if upper.loc[row, col] > 0.85
    ]
    print(f"\nHighly correlated feature pairs (|r| > 0.85):")
    for a, b, r in sorted(high_corr_pairs, key=lambda x: -x[2]):
        print(f"  {a} ↔ {b}: {r}")

    print(f"\n[eda] Charts saved to {EDA_OUT}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_eda()
