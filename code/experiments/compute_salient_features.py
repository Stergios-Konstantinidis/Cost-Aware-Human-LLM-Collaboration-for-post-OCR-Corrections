import sys
import warnings
import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeCV, LassoCV

# Adjust path to import from experiments
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.experiment_gbt_classifier import (
    load_tesseract_records,
    load_llm_corrections,
    build_features,
    feature_names,
)

def main():
    print("Loading tesseract records...")
    records = load_tesseract_records()
    target_file = "corrections/tesseract/tesseract_Full_Expert_Robuste_8_google__gemini-3-flash-preview.json"
    corrections = load_llm_corrections(target_file)
    cache_path = Path("results/ml_models/cached_features_noemb.npy")
    if cache_path.exists():
        print(f"Loading cached features from {cache_path}...")
        X = np.load(cache_path)
    else:
        X = build_features(records)
    feat_names = feature_names()
    
    base_wer = np.array([float(r["wer"]) for r in records], dtype=np.float32)
    corr_wer = np.array([
        corrections.get(r["filename"], {}).get("wer", float(r["wer"]))
        for r in records
    ], dtype=np.float32)
    
    delta_wer = base_wer - corr_wer
    y = np.array([1 if d > 0.03 else 0 for d in delta_wer], dtype=np.int32)
    
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    print("\nTraining GBT Classifier...")
    gbt = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=5, subsample=0.8, random_state=42
    )
    gbt.fit(X_s, y)
    gbt_importances = gbt.feature_importances_
    
    print("Training Linear SVM Classifier...")
    svm = SVC(kernel="linear", C=1.0, random_state=42)
    svm.fit(X_s, y)
    svm_coefs = svm.coef_[0]
    
    print("Training Ridge Regression (predicting raw WER)...")
    ridge = RidgeCV(alphas=np.logspace(-3, 5, 30), cv=5)
    ridge.fit(X_s, base_wer)
    ridge_coefs = ridge.coef_
    
    print("Training Lasso Regression (predicting raw WER)...")
    lasso = LassoCV(cv=5, max_iter=5000, random_state=42)
    lasso.fit(X_s, base_wer)
    lasso_coefs = lasso.coef_
    
    print("Training Lasso Delta Regression (predicting Delta WER)...")
    lasso_delta = LassoCV(cv=5, max_iter=5000, random_state=42)
    lasso_delta.fit(X_s, delta_wer)
    lasso_delta_coefs = lasso_delta.coef_
    
    def get_top_k(weights, is_importance=False, k=10):
        if is_importance:
            indices = np.argsort(weights)[::-1][:k]
            return [(feat_names[idx], weights[idx]) for idx in indices]
        else:
            abs_weights = np.abs(weights)
            indices = np.argsort(abs_weights)[::-1][:k]
            return [(feat_names[idx], weights[idx]) for idx in indices]
            
    top_gbt = get_top_k(gbt_importances, is_importance=True)
    top_svm = get_top_k(svm_coefs)
    top_ridge = get_top_k(ridge_coefs)
    top_lasso = get_top_k(lasso_coefs)
    top_lasso_delta = get_top_k(lasso_delta_coefs)
    
    results = {
        "gbt": top_gbt,
        "svm": top_svm,
        "ridge": top_ridge,
        "lasso": top_lasso,
        "lasso_delta": top_lasso_delta,
    }
    
    with open("results/salient_features.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print("\n" + "=" * 80)
    print("Most Salient Features Per Model")
    print("=" * 80)
    
    for model_name, top_features in results.items():
        print(f"\n--- Model: {model_name.upper()} ---")
        for rank, (name, val) in enumerate(top_features, 1):
            print(f"  {rank:2d}. {name:35s} | value/importance = {val:+.6f}")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
