"""
plot_lasso_feature_impact.py
============================
Generates a horizontal bar chart showing the standardized LassoCV coefficients
for the routing model that predicts ΔCER (CER_raw - CER_corrected).

This is the actual model used for regression-guided routing in the paper (RQ1).
Positive coefficients indicate features that predict LARGER CER improvement from
LLM correction (i.e., the LLM will help more). Negative coefficients indicate
features that predict the LLM will help less (or may even harm).
"""

import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

# Paths
BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "code"))

from experiments.experiment_gbt_classifier import (
    load_tesseract_records,
    load_llm_corrections,
    build_features,
    feature_names,
)

FIG_OUT_DIR = BASE / "paper" / "figures"
FIG_OUT_PATH = FIG_OUT_DIR / "lasso_feature_impact.png"


def compute_delta_cer_coefficients():
    """Train LassoCV on ΔCER (the actual routing target) and return coefficients."""
    print("Loading tesseract records...")
    records = load_tesseract_records()
    target_file = "corrections/tesseract/tesseract_Full_Expert_Robuste_8_google__gemini-3-flash-preview.json"
    corrections = load_llm_corrections(target_file)
    X = build_features(records)
    names = feature_names()

    # Compute ΔCER = CER_raw - CER_corrected (positive means LLM helps)
    base_cer = np.array([float(r["cer"]) for r in records], dtype=np.float32)
    corr_cer = np.array([
        corrections.get(r["filename"], {}).get("cer", float(r["cer"]))
        for r in records
    ], dtype=np.float32)
    delta_cer = base_cer - corr_cer

    print(f"  N = {len(records)}, Features = {X.shape[1]}")
    print(f"  ΔCER range: [{delta_cer.min():.4f}, {delta_cer.max():.4f}]")
    print(f"  Mean ΔCER: {delta_cer.mean():.4f}")

    # Standardize features and fit LassoCV
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    print("Training LassoCV (predicting ΔCER)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lasso = LassoCV(cv=5, max_iter=5000, random_state=42)
        lasso.fit(X_s, delta_cer)

    print(f"  Selected alpha: {lasso.alpha_:.6f}")
    n_nonzero = np.sum(lasso.coef_ != 0)
    print(f"  Non-zero coefficients: {n_nonzero}/{len(lasso.coef_)}")

    # Get top 10 by absolute magnitude
    abs_coefs = np.abs(lasso.coef_)
    top_idx = np.argsort(abs_coefs)[::-1][:10]
    top_features = [(names[i], float(lasso.coef_[i])) for i in top_idx]

    print("\n  Top 10 features (by |coefficient|):")
    for rank, (name, val) in enumerate(top_features, 1):
        print(f"    {rank:2d}. {name:30s} β = {val:+.6f}")

    return top_features


def plot_feature_impact(top_features):
    """Plot horizontal bar chart of LassoCV ΔCER coefficients."""
    # Sort by absolute magnitude (smallest first → largest at top of plot)
    top_features = sorted(top_features, key=lambda x: abs(x[1]))
    features = [x[0] for x in top_features]
    coefficients = [x[1] for x in top_features]

    # Clean feature names for display
    clean_names = {
        "newline_density": "Newline Density",
        "space_ratio": "Space Ratio",
        "avg_confidence": "Avg. OCR Confidence",
        "ortho_integrity_word": "Spelling Integrity (word)",
        "ortho_integrity_char": "Spelling Integrity (char)",
        "dict_hit_rate": "Dictionary Hit Rate",
        "freq_y": "Freq. letter 'y'",
        "freq_j": "Freq. letter 'j'",
        "freq_g": "Freq. letter 'g'",
        "freq_c": "Freq. letter 'c'",
        "freq_l": "Freq. letter 'l'",
        "freq_e": "Freq. letter 'e'",
        "freq_d": "Freq. letter 'd'",
        "freq_s": "Freq. letter 's'",
        "freq_n": "Freq. letter 'n'",
        "newspaper_TL": "Tribune de Lausanne",
        "publication_year": "Publication Year",
        "word_count": "Word Count",
        "text_length": "Text Length",
        "avg_word_length": "Avg. Word Length",
        "upper_ratio": "Uppercase Ratio",
        "punct_ratio": "Punctuation Ratio",
        "digit_ratio": "Digit Ratio",
        "unique_char_ratio": "Unique Char. Ratio",
        "avg_chars_per_line": "Avg. Chars/Line",
        "num_lines": "Number of Lines",
        "max_run_length": "Max Run Length",
        "avg_run_length": "Avg. Run Length",
        "spell_length_ratio": "Spell Length Ratio",
    }
    display_names = [clean_names.get(f, f) for f in features]

    # Styling
    C_POS = "#2e86de"   # blue (LLM correction helps more)
    C_NEG = "#e74c3c"   # red (LLM correction helps less / may harm)
    C_PANEL = "#f8f9fa"
    C_GRID = "#dee2e6"

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": C_PANEL,
        "axes.edgecolor": "#adb5bd"
    })

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    colors = [C_POS if val >= 0 else C_NEG for val in coefficients]
    bars = ax.barh(display_names, coefficients, color=colors,
                   edgecolor="none", height=0.6, alpha=0.9, zorder=3)

    # Value labels
    for bar, val in zip(bars, coefficients):
        width = bar.get_width()
        offset = 0.002 if val >= 0 else -0.002
        align = "left" if val >= 0 else "right"
        ax.text(width + offset, bar.get_y() + bar.get_height()/2,
                f"{val:+.4f}", ha=align, va="center", fontsize=9,
                fontweight="bold", color="#2d3436")

    # Format axes
    ax.axvline(0, color="black", linestyle="-", linewidth=1.0, zorder=4)
    ax.set_xlabel(r"Standardized LassoCV Coefficient ($\beta$)", fontweight="semibold", labelpad=8)
    ax.set_title(r"LassoCV Feature Impact on Predicted $\Delta$CER", fontweight="bold", pad=15)
    ax.grid(True, axis="x", color=C_GRID, linestyle="--", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_POS, alpha=0.9, label=r"Predicts larger $\Delta$CER (LLM helps more)"),
        Patch(facecolor=C_NEG, alpha=0.9, label=r"Predicts smaller $\Delta$CER (LLM helps less)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", framealpha=0.9, fontsize=9.5)

    # Adjust limits
    max_val = max(abs(x) for x in coefficients)
    ax.set_xlim(-max_val - 0.02, max_val + 0.02)

    plt.tight_layout()
    FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    # Copy to results and artifacts
    for dest_dir in [BASE / "results" / "figures", BASE / "artifacts"]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(FIG_OUT_PATH), str(dest_dir / "lasso_feature_impact.png"))

    print(f"\n✓ Saved plot to {FIG_OUT_PATH}")


if __name__ == "__main__":
    top_features = compute_delta_cer_coefficients()
    plot_feature_impact(top_features)
