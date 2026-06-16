"""
plot_safeguard_routing.py
=========================
Generates a two-panel comparison figure matching the paper's original styling:
- Left Panel: Standard routing curves for Character Error Rate (CER) only.
- Right Panel: Routing curves for CER with the overcorrection guard (safeguard) active.

Uses the CIKM paper's optimal configuration:
`corrections/tesseract/tesseract_Full_Expert_Robuste_8_google__gemini-3-flash-preview.json`
with LassoCV regression for routing.
"""

import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.experiment_gbt_classifier import (
    load_tesseract_records,
    load_llm_corrections,
    build_features,
)

# ── Orthographic baseline (Tesseract, Full_Orthographic) ──
ORTHO_BASELINES = {"wer": 0.2284, "cer": 0.0675}

BASE = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE / "results"
IMAGES = BASE / "data" / "evaluation_dataset" / "images"
FIG_OUT_DIR = BASE / "paper" / "figures"
FIG_OUT_PATH = FIG_OUT_DIR / "safeguard_routing_plot.png"


def train_lassocv_delta(X, records, corrections):
    """Train LassoCV to predict Δ-WER and Δ-CER (CIKM config)."""
    delta_wer = np.array([
        float(r["wer"]) - corrections.get(r["filename"], {}).get("wer", float(r["wer"]))
        for r in records
    ], dtype=np.float32)
    delta_cer = np.array([
        float(r["cer"]) - corrections.get(r["filename"], {}).get("cer", float(r["cer"]))
        for r in records
    ], dtype=np.float32)

    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reg', LassoCV(cv=5, max_iter=5000, random_state=42))
    ])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred_dw = cross_val_predict(pipe, X, delta_wer, cv=cv)
        pred_dc = cross_val_predict(pipe, X, delta_cer, cv=cv)

    return pred_dw, pred_dc


def train_classifier(X_stacked, delta_target, min_delta):
    """Train LogReg routing classifier. Returns P(Δ > min_delta)."""
    y = (delta_target > min_delta).astype(int)
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0, class_weight='balanced',
                                   max_iter=1000, random_state=42))
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probas = cross_val_predict(clf, X_stacked, y, cv=cv, n_jobs=-1,
                                   method='predict_proba')[:, 1]
    return probas


def train_harm_guard(X_stacked, delta_target):
    """
    Train a binary overcorrection guard.
    Returns P(Δ >= 0) — probability that correction does NOT harm.
    """
    y = (delta_target >= 0).astype(int)  # 1 = safe to correct, 0 = correction harms
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0, class_weight='balanced',
                                   max_iter=1000, random_state=42))
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probas = cross_val_predict(clf, X_stacked, y, cv=cv, n_jobs=-1,
                                   method='predict_proba')[:, 1]
    return probas


def compute_routing_curves(sort_idx, base_vals, corr_vals, N, scores, threshold):
    """Compute the full and capped frontiers for a given document ordering."""
    avg_base = np.mean(base_vals)
    pct_list = [0.0]
    val_list_full = [avg_base]
    val_list_capped = [avg_base]
    current_sum_full = np.sum(base_vals)
    current_sum_capped = np.sum(base_vals)
    
    last_capped_k = -1
    for k in range(N):
        idx = sort_idx[k]
        
        current_sum_full -= base_vals[idx]
        current_sum_full += corr_vals[idx]
        
        if scores[idx] >= threshold:
            current_sum_capped -= base_vals[idx]
            current_sum_capped += corr_vals[idx]
            last_capped_k = k
            
        pct_list.append((k + 1) / N * 100)
        val_list_full.append(current_sum_full / N)
        val_list_capped.append(current_sum_capped / N)
        
    pct_list_capped = list(pct_list)
    if last_capped_k != -1:
        pct_list_capped = pct_list_capped[:last_capped_k + 2]
        val_list_capped = val_list_capped[:last_capped_k + 2]
        
    return pct_list, val_list_full, pct_list_capped, val_list_capped


def compute_routing_curve(sort_idx, base_vals, corr_vals, N):
    """Standard routing curve: correct documents in sort_idx order."""
    avg_base = np.mean(base_vals)
    pct_list, val_list = [0.0], [avg_base]
    current_sum = np.sum(base_vals)
    for k in range(N):
        idx = sort_idx[k]
        current_sum -= base_vals[idx]
        current_sum += corr_vals[idx]
        pct_list.append((k + 1) / N * 100)
        val_list.append(current_sum / N)
    return pct_list, val_list


def compute_guarded_routing_curve(sort_idx, base_vals, corr_vals,
                                  guard_safe, N, max_human=None):
    """
    Full routing curve (to 100%) with overcorrection guard.
    The safeguard IS the human routing mechanism:
    - guard_safe >= 0.5 → accept LLM correction
    - guard_safe <  0.5 → route to human corrector (0% error)
      but only up to max_human docs; after that, fall back to LLM correction.
    """
    avg_base = np.mean(base_vals)
    pct_list, val_list = [0.0], [avg_base]
    current_sum = np.sum(base_vals)
    n_human = 0
    for k in range(N):
        idx = sort_idx[k]
        if guard_safe[idx] >= 0.5:
            # Safe → LLM correction
            current_sum -= base_vals[idx]
            current_sum += corr_vals[idx]
        elif max_human is None or n_human < max_human:
            # Guard blocked → route to human (perfect correction, 0% error)
            current_sum -= base_vals[idx]
            n_human += 1
        else:
            # Budget exhausted → fall back to LLM correction (better than nothing)
            current_sum -= base_vals[idx]
            current_sum += corr_vals[idx]
        pct_list.append((k + 1) / N * 100)
        val_list.append(current_sum / N)
    return pct_list, val_list, n_human


def compute_token_axis(sort_idx, token_counts, N):
    cum_tokens = [0]
    running = 0
    for k in range(N):
        running += token_counts[sort_idx[k]]
        cum_tokens.append(running)
    return cum_tokens


def main():
    print("Loading data...")
    records = load_tesseract_records()
    target_file = "corrections/tesseract/tesseract_Full_Expert_Robuste_8_google__gemini-3-flash-preview.json"
    corrections = load_llm_corrections(target_file)
    X = build_features(records)

    # Token counts per document
    token_counts = np.array([
        len(r.get("raw_ocr", "").split()) for r in records
    ], dtype=np.int32)
    print(f"  Total tokens: {np.sum(token_counts):,}")

    print("Training LassoCV delta regressions (10-fold CV)...")
    pred_dw, pred_dc = train_lassocv_delta(X, records, corrections)
    print(f"  pred_ΔWER range: [{pred_dw.min():.4f}, {pred_dw.max():.4f}]")
    print(f"  pred_ΔCER range: [{pred_dc.min():.4f}, {pred_dc.max():.4f}]")

    # Our approach: LassoCV regression-guided routing (CIKM config)
    our_pred_delta = pred_dc
    print("Using LassoCV regression-guided routing (pred Δ-CER)...")

    # Precompute ground-truth deltas for CER
    base_cer = np.array([float(r["cer"]) for r in records])
    corr_cer = np.array([
        corrections.get(r["filename"], {}).get("cer", float(r["cer"]))
        for r in records
    ])
    delta_cer_gt = base_cer - corr_cer

    # Feature matrix for guard (base features only, no stacking needed for LassoCV)
    X_guard = X

    print("Training overcorrection guard (P(Δ ≥ 0))...")
    guard_safe = train_harm_guard(X_guard, delta_cer_gt)

    # ── ConfBERT ──
    print("Loading/Training ConfBERT...")
    confbert_cache = RESULTS / "ml_models/confbert_probas.npy"
    if confbert_cache.exists():
        confbert_probas = np.load(confbert_cache)
        print(f"  Loaded cached ConfBERT probas: {len(confbert_probas)}")
        if len(confbert_probas) != len(records):
            print(f"  Cache mismatch ({len(confbert_probas)} vs {len(records)}), retraining...")
            from experiments.confbert_router import train_confbert_router
            confbert_probas = train_confbert_router(
                records, corrections, str(RESULTS), str(IMAGES),
                metric="cer", min_delta=0
            )
            np.save(confbert_cache, confbert_probas)
    else:
        from experiments.confbert_router import train_confbert_router
        confbert_probas = train_confbert_router(
            records, corrections, str(RESULTS), str(IMAGES),
            metric="cer", min_delta=0
        )
        np.save(confbert_cache, confbert_probas)
        print(f"  Cached ConfBERT probas to {confbert_cache}")

    # ── Plotting ──
    print("Generating single-panel plot...")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9
    })

    fig, ax = plt.subplots(1, 1, figsize=(8, 6.5))

    display_metric = "cer"
    avg_base = np.mean(base_cer)
    N = len(records)

    # Precompute orderings
    oracle_idx = np.argsort(delta_cer_gt)[::-1]
    oracle_pct, oracle_vals = compute_routing_curve(oracle_idx, base_cer, corr_cer, N)

    confbert_idx = np.argsort(confbert_probas)[::-1]
    confbert_pct, confbert_vals = compute_routing_curve(confbert_idx, base_cer, corr_cer, N)

    our_idx = np.argsort(our_pred_delta)[::-1]
    our_pct, our_vals = compute_routing_curve(our_idx, base_cer, corr_cer, N)

    # ── Curves ──

    # (a) Ours (no guard) - faded reference
    ax.plot(our_pct, [v*100 for v in our_vals], color='#d62728', linewidth=2,
             alpha=1.0, linestyle='-', label='Ours (no safeguard)')

    # (b) Ours + Safeguard with 30 human-reviewed documents
    pct_g, val_g, n_h = compute_guarded_routing_curve(
        our_idx, base_cer, corr_cer, guard_safe, N, max_human=30)
    ax.plot(pct_g, [v*100 for v in val_g], color='#1f77b4', linewidth=2.2,
             linestyle='-', label=f'Ours + Safeguard (<5% docs human reviewed)')

    # (c) ConfBERT baseline
    ax.plot(confbert_pct, [v*100 for v in confbert_vals], color='#ff7f0e', linestyle='--', linewidth=2,
             label='ConfBERT (Hemmer et al.)')

    # (d) Oracle (perfect routing, no human corrections)
    ax.plot(oracle_pct, [v*100 for v in oracle_vals], color='#7f7f7f',
             linestyle=':', linewidth=1.8, label='Oracle (Perfect Routing)')

    # Baseline reference line
    ax.axhline(avg_base * 100, color='black', linestyle='--', linewidth=1.2, alpha=0.6,
               label=f'Baseline OCR ({avg_base*100:.2f}%)')

    ax.set_xlabel('% of Documents Processed', fontweight='semibold')
    ax.set_ylabel('Overall CER (%)', fontweight='semibold')
    ax.set_title('CER Routing with Safeguard', fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', framealpha=0.9)

    # Top Token Axis
    our_cum_tokens = compute_token_axis(our_idx, token_counts, N)
    ax_twin = ax.twiny()
    tick_pcts = [0, 20, 40, 60, 80, 100]
    tick_tokens = [our_cum_tokens[int(round(p / 100 * N))] for p in tick_pcts]
    ax_twin.set_xlim(ax.get_xlim())
    ax_twin.set_xticks(tick_pcts)
    ax_twin.set_xticklabels([f'{t/1000:.1f}k' for t in tick_tokens], fontsize=8)
    ax_twin.set_xlabel('Cumulative Tokens Processed', fontsize=9, labelpad=6)

    plt.tight_layout()
    FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_OUT_PATH, dpi=300, bbox_inches='tight')
    plt.close()

    # Save copy to artifacts directory
    artifact_fig_dir = BASE / "artifacts"
    artifact_fig_dir.mkdir(parents=True, exist_ok=True)
    artifact_fig_path = artifact_fig_dir / "safeguard_routing_plot.png"
    import shutil
    shutil.copy(str(FIG_OUT_PATH), str(artifact_fig_path))

    print(f"Successfully generated safeguard routing plot at:\n  - {FIG_OUT_PATH}\n  - {artifact_fig_path}")


if __name__ == "__main__":
    main()
