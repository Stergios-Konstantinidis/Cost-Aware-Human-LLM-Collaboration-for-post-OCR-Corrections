"""
experiment_gbt_classifier.py
=============================
Binary classifier (Gradient Boosted Trees) for selective OCR correction routing.

Question: Given raw OCR features alone, can we predict whether applying LLM
correction to a document will produce a meaningful improvement?

Setup
-----
- Engine: Tesseract only (~597 records)
- Features: same 54-d feature set as the regression experiments
    (40 surface + 13 metadata + 1 engine_flag)
- Metrics: WER and CER evaluated independently
- Labels: three binary thresholds per metric
    delta = metric_baseline - metric_corrected  (positive = correction helped)
    ┌─────────────────────────────────────────────────────────────────┐
    │ Exp 0  label=1 if delta > 0      (any improvement)             │
    │ Exp 1  label=1 if delta > 0.03   (>3 pp absolute improvement)  │
    │ Exp 2  label=1 if delta > 0.05   (>5 pp absolute improvement)  │
    └─────────────────────────────────────────────────────────────────┘
- Validation: Leave-One-Out (LOO) — one GBT trained per held-out record
- Correction source: best available saved correction per metric
  (best WER correction and best CER correction tracked independently)

Outputs
-------
  results/gbt_classifier_results.json
"""

import json
import glob
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.regression_features import (
    build_dataset, SURFACE_FEATURE_NAMES, METADATA_FEATURE_NAMES,
    load_metadata_lookup, load_confidence_lookup, enrich_records,
)

# ── paths ─────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parent.parent.parent
RESULTS  = BASE / "results"
GT_PATH  = BASE / "data" / "evaluation_dataset" / "groundtruth.json"
BASELINE = RESULTS / "baselines/baseline_tesseract.json"
ENGINE   = "tesseract"
ENGINE_FLAG = 1.0

# ── label thresholds ──────────────────────────────────────────────────────────
THRESHOLDS = [
    {"name": "delta_gt_0",    "min_delta": 0.00, "label": "delta > 0 (any gain)"},
    {"name": "delta_gt_3pct", "min_delta": 0.03, "label": "delta > 3 pp"},
    {"name": "delta_gt_5pct", "min_delta": 0.05, "label": "delta > 5 pp"},
]

# ── metrics to evaluate ───────────────────────────────────────────────────────
METRICS = ["wer", "cer"]

# ── GBT hyperparameters ───────────────────────────────────────────────────────
GBT_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=5,
    random_state=42,
)


# ─── data loading ─────────────────────────────────────────────────────────────

def load_tesseract_records() -> list[dict]:
    if not BASELINE.exists():
        raise FileNotFoundError(f"Baseline not found: {BASELINE}")
    with open(BASELINE, encoding="utf-8") as f:
        data = json.load(f)
    valid = [r for r in data if r["wer"] <= 5.0 and r.get("raw_ocr", "").strip()]
    for r in valid:
        r["engine"] = ENGINE
    print(f"  [tesseract] {len(valid)} valid baseline records")
    meta_lookup = load_metadata_lookup(GT_PATH)
    conf_lookup = load_confidence_lookup(ENGINE, RESULTS)
    print(f"  [tesseract] confidence lookup: {len(conf_lookup)} entries")
    enrich_records(valid, meta_lookup, confidence_lookup=conf_lookup)
    return valid


def load_llm_corrections(target_file: str) -> dict[str, dict]:
    """
    Load corrections from a single specific LLM run file to avoid oracle leakage.
    Returns: filename → {"wer": float, "cer": float}
    """
    fpath = RESULTS / target_file
    if not fpath.exists():
        raise FileNotFoundError(f"Correction file not found: {fpath}")
    print(f"  Loading corrections from {target_file}")

    corrections = {}
    with open(fpath, encoding="utf-8") as f:
        records = json.load(f)
        for r in records:
            fname = r.get("filename")
            if fname:
                corrections[fname] = {
                    "wer": float(r.get("wer", 1.0)),
                    "cer": float(r.get("cer", 1.0)),
                }

    print(f"  Loaded corrections for {len(corrections)} documents")
    return corrections


# ─── features ─────────────────────────────────────────────────────────────────

def build_features(records: list[dict]) -> np.ndarray:
    print(f"  Building 54-d features for {len(records)} records …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, _, _ = build_dataset(records, use_embeddings=False, verbose=False)
    engine_col = np.full((len(records), 1), ENGINE_FLAG, dtype=np.float32)
    return np.concatenate([X, engine_col], axis=1)   # (N, 55)


def feature_names() -> list[str]:
    return SURFACE_FEATURE_NAMES + METADATA_FEATURE_NAMES + ["engine_flag"]


# ─── labels ───────────────────────────────────────────────────────────────────

def build_labels(
    records: list[dict],
    corrections: dict[str, dict],
    min_delta: float,
    metric: str,          # "wer" or "cer"
) -> tuple[np.ndarray, list[float], list[bool]]:
    corr_key = metric
    y, deltas, has_corr = [], [], []
    for r in records:
        fname = r["filename"]
        base_val = float(r[metric])
        if fname in corrections:
            corr_val = corrections[fname][corr_key]
            delta = base_val - corr_val
            has_corr.append(True)
        else:
            delta = 0.0
            has_corr.append(False)
        deltas.append(delta)
        y.append(1 if delta > min_delta else 0)
    return np.array(y, dtype=int), deltas, has_corr


# ─── LOO ──────────────────────────────────────────────────────────────────────

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict

def run_loo(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    loo = LeaveOneOut()
    n = len(y)
    print(f"  Running LOO ({n} folds) with n_jobs=-1 …", flush=True)

    if len(np.unique(y)) < 2:
        probas = np.full(n, 0.5)
        y_pred = np.full(n, int(y.mean() >= 0.5))
        return probas, y_pred

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('gbt', GradientBoostingClassifier(**GBT_PARAMS))
    ])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probas_all = cross_val_predict(
            pipeline, X, y, cv=loo, n_jobs=-1, method='predict_proba'
        )

    probas = probas_all[:, 1]
    y_pred = (probas >= 0.5).astype(int)
    
    return probas, y_pred


# ─── metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true, probas, y_pred_05,
    records, deltas, has_corr, corrections,
    threshold_cfg, metric,
) -> dict:
    corr_key = metric
    auc  = float(roc_auc_score(y_true, probas)) if len(np.unique(y_true)) > 1 else 0.5
    acc  = float(accuracy_score(y_true, y_pred_05))
    prec = float(precision_score(y_true, y_pred_05, zero_division=0))
    rec  = float(recall_score(y_true, y_pred_05, zero_division=0))
    f1   = float(f1_score(y_true, y_pred_05, zero_division=0))

    # Simulation
    base_vals = [float(r[metric]) for r in records]
    corr_vals = [
        corrections[r["filename"]][corr_key] if r["filename"] in corrections else float(r[metric])
        for r in records
    ]

    def simulate(mask):
        after = [corr_vals[i] if mask[i] else base_vals[i] for i in range(len(records))]
        avg_after    = float(np.mean(after))
        avg_base     = float(np.mean(base_vals))
        avg_full     = float(np.mean(corr_vals))
        reduction    = avg_base - avg_after
        full_gain    = avg_base - avg_full
        recovery     = round(reduction / full_gain, 4) if full_gain > 0 else 0.0
        return {
            "n_routed":         int(mask.sum()),
            "pct_routed":       round(100 * mask.mean(), 1),
            f"avg_{metric}_after": round(avg_after, 6),
            f"{metric}_reduction":  round(reduction, 6),
            "recovery_of_full": recovery,
        }

    report = classification_report(y_true, y_pred_05, digits=4)

    return {
        "metric":               metric,
        "threshold_name":       threshold_cfg["name"],
        "threshold_label":      threshold_cfg["label"],
        "min_delta":            threshold_cfg["min_delta"],
        "n_records":            len(records),
        "n_positive":           int(y_true.sum()),
        "n_negative":           int((y_true == 0).sum()),
        "auc_roc":              round(auc, 6),
        "accuracy":             round(acc, 6),
        "precision_at_05":      round(prec, 6),
        "recall_at_05":         round(rec, 6),
        "f1_at_05":             round(f1, 6),
        "classification_report_05": report,
        f"{metric}_baseline":   round(float(np.mean(base_vals)), 6),
        f"{metric}_always_correct": round(float(np.mean(corr_vals)), 6),
        "routing_at_05":        simulate(y_pred_05),
        "routing_always":       simulate(np.ones(len(records), dtype=int)),
        "routing_never":        simulate(np.zeros(len(records), dtype=int)),
    }


def compute_feature_importances(X, y, feat_names, top_k=20):
    if len(np.unique(y)) < 2:
        return []
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = GradientBoostingClassifier(**GBT_PARAMS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_s, y)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_k]
    return [
        {"rank": i + 1, "feature": feat_names[idx], "importance": round(float(importances[idx]), 6)}
        for i, idx in enumerate(indices)
    ]


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("GBT Classifier — Selective OCR Correction Routing (LOO)")
    print("=" * 65)

    records     = load_tesseract_records()
    target_file = "corrections/tesseract/tesseract_Full_Advanced_5_google__gemini-3-flash-preview.json"
    corrections = load_llm_corrections(target_file)

    print("\n  Building features …")
    X = build_features(records)
    feat_names = feature_names()
    print(f"  Feature matrix: {X.shape}  |  Features: {len(feat_names)}")

    per_threshold_results = []
    per_record_rows       = []

    # ── run 3 thresholds × 2 metrics ─────────────────────────────────────────
    for metric in METRICS:
        print(f"\n{'═' * 65}")
        print(f"  METRIC: {metric.upper()}")
        print(f"{'═' * 65}")

        for thr_cfg in THRESHOLDS:
            print(f"\n{'─' * 60}")
            print(f"  {metric.upper()} | {thr_cfg['label']}")
            print(f"{'─' * 60}")

            y, deltas, has_corr = build_labels(records, corrections, thr_cfg["min_delta"], metric)
            print(f"  Labels: {y.sum()} positive / {(y==0).sum()} negative ({100*y.mean():.1f}% positive)")

            probas, y_pred_05 = run_loo(X, y)

            m = compute_metrics(
                y, probas, y_pred_05,
                records, deltas, has_corr, corrections,
                thr_cfg, metric,
            )
            m["feature_importances_full_model"] = compute_feature_importances(X, y, feat_names)
            per_threshold_results.append(m)

            print(f"\n  AUC-ROC  : {m['auc_roc']:.4f}")
            print(f"  Accuracy : {m['accuracy']:.4f}")
            print(f"  F1@0.5   : {m['f1_at_05']:.4f}  P={m['precision_at_05']:.4f}  R={m['recall_at_05']:.4f}")
            sim = m["routing_at_05"]
            print(f"  WER δ    : {sim[f'{metric}_reduction']:.4f}  ({sim['pct_routed']}% routed)")
            print(f"\n{m['classification_report_05']}")

            for i, r in enumerate(records):
                per_record_rows.append({
                    "filename":       r["filename"],
                    "metric":         metric,
                    "threshold_name": thr_cfg["name"],
                    f"{metric}_baseline":  round(float(r[metric]), 6),
                    f"{metric}_corrected": round(
                        corrections.get(r["filename"], {}).get(metric, float(r[metric])), 6
                    ),
                    "delta":          round(deltas[i], 6),
                    "has_correction": has_corr[i],
                    "true_label":     int(y[i]),
                    "pred_proba":     round(float(probas[i]), 6),
                    "pred_label_05":  int(y_pred_05[i]),
                })

    # ── save ──────────────────────────────────────────────────────────────────
    out = {
        "experiment":    "gbt_classifier_loo",
        "engine":        ENGINE,
        "n_records":     len(records),
        "n_features":    X.shape[1],
        "feature_names": feat_names,
        "gbt_params":    GBT_PARAMS,
        "per_threshold": per_threshold_results,
        "per_record":    per_record_rows,
    }
    out_path = RESULTS / "ml_models/gbt_classifier_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Results saved → {out_path}")

    # ── summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"  {'Metric':<5}  {'Threshold':<18}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  "
          f"{'F1':>6}  {'AUC':>6}  {'%Routed':>8}  {'δ':>7}")
    print("  " + "-" * 96)
    for m in per_threshold_results:
        sim = m["routing_at_05"]
        metric = m["metric"]
        delta_val = sim[f"{metric}_reduction"]
        print(
            f"  {metric.upper():<5}  {m['threshold_label']:<18}  "
            f"{m['accuracy']:>6.3f}  {m['precision_at_05']:>6.3f}  {m['recall_at_05']:>6.3f}  "
            f"{m['f1_at_05']:>6.3f}  {m['auc_roc']:>6.3f}  "
            f"{sim['pct_routed']:>7.1f}%  {delta_val:>7.4f}"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()
