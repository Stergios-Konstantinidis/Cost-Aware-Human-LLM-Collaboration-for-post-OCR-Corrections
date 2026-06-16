# Code Directory

Source code for the OCR post-correction routing pipeline.  
Organized into four modules by functional domain.

---

## `plotting/` — Visualization Scripts

Generate all figures used in the paper and exploratory analysis.

| Script | Description |
|--------|-------------|
| `plot_routing_graph_paper.py` | **Main paper figure** — routing frontier comparing Oracle, ConfBERT, Spell-check, and our classifier |
| `plot_routing_graph_guarded.py` | Variant with overcorrection guard applied to each routing curve |
| `plot_routing_frontier.py` | 2×2 grid routing frontier at varying δ thresholds (Tesseract) |
| `plot_routing_frontier_paddle.py` | Same as above but for PaddleOCR engine |
| `plot_threshold_sweep.py` | Sweep across min-delta thresholds; one graph per threshold value |
| `plot_threshold_comparison.py` | Overlay of threshold-sweep results across prediction models |
| `plot_routing_density.py` | Distribution density of routing decisions |
| `plot_strategy_comparison.py` | Bar chart comparing Full, Selective, and Conditional strategies |
| `plot_engine_comparison.py` | Multi-panel engine comparison (baseline bars, heatmap, improvement chart) |
| `plot_confidence.py` | OCR confidence score distributions |
| `plot_error_confidence.py` | Error rate vs. confidence scatter (WER) |
| `plot_error_confidence_cer.py` | Error rate vs. confidence scatter (CER) |
| `generate_regression_figures.py` | Automated generation of regression model diagnostic plots |

## `evaluation/` — OCR Evaluation & Correction Pipelines

Run LLM-based corrections and compute quality metrics.

| Script | Description |
|--------|-------------|
| `run_evaluations.py` | **Main pipeline** — runs Full/Selective/SelectiveNoContext strategies across OCR engines × LLM models × prompt levels. Manages batched API calls, caching, and metric computation. |
| `run_evaluations_conditional.py` | ConditionalFull strategy — corrects only documents whose average OCR confidence falls below a threshold |
| `run_ortho_experiment.py` | Orthographic (spell-check only) correction baseline across confidence thresholds |
| `rebuild_baselines.py` | Recompute baseline WER/CER from raw OCR cache without LLM calls |
| `update_confidence_data.py` | Extract per-document OCR confidence scores and low-confidence word lists |

## `experiments/` — Machine Learning Experiments

Train and evaluate models for routing prediction.

| Script | Description |
|--------|-------------|
| `experiment_gbt_classifier.py` | **Gradient Boosted Tree** binary classifier with LOO validation — primary routing model |
| `experiment_svm_classifier.py` | SVM classifier variant for comparison |
| `experiment_linear_regression.py` | Ridge regression to predict corrected WER/CER (pooled + per-engine) |
| `experiment_nn_regression.py` | MLP neural network regression variant |
| `experiment_regression_routing.py` | Offline routing threshold analysis using regression predictions |
| `confbert_router.py` | ConfBERT baseline — BERT model with confidence-aware embedding injection |
| `experiment_lazypredict.py` | Quick scan of many classifiers via LazyPredict |
| `lazy_clf_scan.py` | Extended classifier scan with custom feature sets |

## `utils/` — Shared Utilities

Feature engineering, data processing, and result aggregation.

| Script | Description |
|--------|-------------|
| `regression_features.py` | **Core feature engineering** — 54-d feature vector (40 surface + 13 metadata + 1 engine flag), spell-correction, embedding computation, dataset builder |
| `process_all_results.py` | Aggregate and display results from all experiment types |
| `process_results.py` | Parse and display a single summary JSON |
| `summarize_results.py` | Print GBT classifier results in tabular form |
| `aggregate_conditional.py` | Build `summary_conditional.json` from ConditionalFull result files |
| `inject_paddle_summary.py` | Inject PaddleOCR results into the main summary and leaderboard |
| `analyze_ortho_threshold.py` | Find optimal confidence threshold for spell-check-only correction |
| `update_eval.py` | Programmatic update of evaluation script internals |
| `test_gt_loader.py` | Smoke test for groundtruth JSON loading |

---

## Dependencies

```
scikit-learn  numpy  pandas  matplotlib  jiwer  tqdm  pyspellchecker
sentence-transformers  python-dotenv  openai
```

## Running Scripts

All scripts use `Path(__file__).resolve()` for path resolution, so they can be run from any directory:

```bash
python code/plotting/plot_routing_graph_paper.py
python code/experiments/experiment_gbt_classifier.py
python code/evaluation/run_evaluations.py --dry-run
```
