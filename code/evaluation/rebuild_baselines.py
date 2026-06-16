"""
rebuild_baselines.py
=====================
Rebuilds baseline_{engine}.json files from scratch so they contain
every entry in data/evaluation_dataset/groundtruth.json.

Steps
-----
1. Load all 609 groundtruth entries.
2. Load existing OCR cache (data/raw_ocr_results.json) — already-OCR'd
   images are re-used; only new ones are OCR'd.
3. Save the updated OCR cache.
4. Recompute WER/CER for every groundtruth entry that has OCR output.
5. Overwrite results/baseline_{engine}.json with the full set.

Usage
-----
    python code/rebuild_baselines.py [--engines easyocr tesseract]
"""

import argparse
import json
import re
import string
import sys
from pathlib import Path

import jiwer
import numpy as np
from tqdm import tqdm

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent.parent
DATA_DIR     = BASE_DIR / "data"
EVAL_DIR     = DATA_DIR / "evaluation_dataset"
IMG_DIR      = EVAL_DIR / "images"
RESULTS_DIR  = BASE_DIR / "results"
OCR_CACHE    = DATA_DIR / "raw_ocr_results.json"


# ── text normalisation (identical to run_evaluations.py) ─────────────────────
def apply_annotator_rules(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.replace("E'", "É").replace("E`", "É")
    text = text.replace("&z", "&")
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+([;.,!?:])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(string.punctuation + " ")


def compute_metrics(groundtruth: str, hypothesis: str):
    try:
        gt_norm = apply_annotator_rules(groundtruth) or "[EMPTY]"
        hyp_norm = apply_annotator_rules(hypothesis) or "[EMPTY]"
        return jiwer.wer(gt_norm, hyp_norm), jiwer.cer(gt_norm, hyp_norm)
    except Exception:
        return 1.0, 1.0


# ── OCR engine setup ──────────────────────────────────────────────────────────
def setup_engines(engine_names: list[str]) -> dict:
    engines = {}

    if "tesseract" in engine_names:
        try:
            import pytesseract
            engines["tesseract"] = lambda img_path: pytesseract.image_to_string(
                str(img_path), lang="fra"
            )
            print("  ✓ Tesseract loaded")
        except Exception as exc:
            print(f"  ✗ Tesseract unavailable: {exc}")

    if "easyocr" in engine_names:
        try:
            import easyocr
            reader = easyocr.Reader(["fr"], gpu=False)
            engines["easyocr"] = lambda img_path: "\n".join(
                reader.readtext(str(img_path), detail=0)
            )
            print("  ✓ EasyOCR loaded")
        except Exception as exc:
            print(f"  ✗ EasyOCR unavailable: {exc}")

    if "paddle" in engine_names:
        try:
            import os, warnings
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            from paddleocr import PaddleOCR
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    ocr = PaddleOCR(use_textline_orientation=True, lang="fr")
                except TypeError:
                    ocr = PaddleOCR(use_angle_cls=True, lang="fr")

            def _paddle(img_path):
                res = ocr.ocr(str(img_path), cls=True)
                if not res or not res[0]:
                    return ""
                return "\n".join(line[1][0] for line in res[0])

            engines["paddle"] = _paddle
            print("  ✓ PaddleOCR loaded")
        except Exception as exc:
            print(f"  ✗ PaddleOCR unavailable: {exc}")

    return engines


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Rebuild baseline OCR result files")
    parser.add_argument(
        "--engines", nargs="+", default=["easyocr", "tesseract"],
        help="OCR engines to rebuild (default: easyocr tesseract)"
    )
    args = parser.parse_args()

    # ── 1. Load groundtruth ──────────────────────────────────────────────────
    gt_path = EVAL_DIR / "groundtruth.json"
    if not gt_path.exists():
        sys.exit(f"Groundtruth not found: {gt_path}")
    with open(gt_path, encoding="utf-8") as f:
        groundtruth = json.load(f)
    print(f"\nGroundtruth entries: {len(groundtruth)}")

    # ── 2. Load existing OCR cache ───────────────────────────────────────────
    cache: dict = {}
    if OCR_CACHE.exists():
        with open(OCR_CACHE, encoding="utf-8") as f:
            raw_cache = json.load(f)
        # Only keep non-empty entries from the cache
        for eng, results in raw_cache.items():
            cache[eng] = {k: v for k, v in results.items() if v.strip()}
        print(f"Loaded OCR cache: {OCR_CACHE}")
        for eng, results in cache.items():
            print(f"  [{eng}] {len(results)} non-empty cached images")
    else:
        print("No existing OCR cache — will run all from scratch.")

    # ── 2b. Seed cache from old baseline files (preserves prior results) ──────
    for eng_name in args.engines:
        baseline_path = RESULTS_DIR / f"baselines/baseline_{eng_name}.json"
        if baseline_path.exists():
            with open(baseline_path, encoding="utf-8") as f:
                old_rows = json.load(f)
            if eng_name not in cache:
                cache[eng_name] = {}
            seeded = 0
            for row in old_rows:
                fname = row["filename"]
                raw = row.get("raw_ocr", "")
                if raw.strip() and fname not in cache[eng_name]:
                    cache[eng_name][fname] = raw
                    seeded += 1
            if seeded:
                print(f"  [{eng_name}] Seeded {seeded} entries from old baseline")

    # ── 3. Load OCR engines ──────────────────────────────────────────────────
    print("\nLoading OCR engines …")
    engines = setup_engines(args.engines)
    if not engines:
        sys.exit("No OCR engines available. Check your environment.")

    # ── 4. Run OCR on missing images ─────────────────────────────────────────
    cache_updated = False
    for eng_name, eng_fn in engines.items():
        if eng_name not in cache:
            cache[eng_name] = {}

        missing = [
            item["filename"]
            for item in groundtruth
            if item["filename"] not in cache[eng_name]
        ]

        if not missing:
            print(f"\n  [{eng_name}] All {len(cache[eng_name])} images already cached ✓")
            continue

        print(f"\n  [{eng_name}] Running OCR on {len(missing)} new images …")
        for fname in tqdm(missing, desc=f"OCR/{eng_name}", unit="img"):
            img_path = IMG_DIR / fname
            if not img_path.exists():
                print(f"    WARNING: Image not found — {img_path}")
                cache[eng_name][fname] = ""
                cache_updated = True
                continue
            try:
                cache[eng_name][fname] = eng_fn(img_path)
                cache_updated = True
            except Exception as exc:
                print(f"    ERROR ({eng_name}, {fname}): {exc}")
                cache[eng_name][fname] = ""
                cache_updated = True

    # ── 5. Persist updated cache ─────────────────────────────────────────────
    if cache_updated:
        DATA_DIR.mkdir(exist_ok=True)
        with open(OCR_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        print(f"\nOCR cache saved → {OCR_CACHE}")

    # ── 6. Rebuild baseline files ────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    print()
    for eng_name in engines:
        eng_cache = cache.get(eng_name, {})
        rows = []
        skipped = 0
        for item in groundtruth:
            fname = item["filename"]
            raw_ocr = eng_cache.get(fname, "")
            if not raw_ocr.strip():
                skipped += 1
                continue
            wer, cer = compute_metrics(item["groundtruth_text"], raw_ocr)
            rows.append({
                "filename":    fname,
                "groundtruth": item["groundtruth_text"],
                "raw_ocr":     raw_ocr,
                "wer":         wer,
                "cer":         cer,
            })

        out_path = RESULTS_DIR / f"baselines/baseline_{eng_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

        wers = [r["wer"] for r in rows]
        cers = [r["cer"] for r in rows]
        print(
            f"  [{eng_name}] {len(rows)} entries written  "
            f"(skipped {skipped} empty)  "
            f"avg WER={np.mean(wers):.4f}  avg CER={np.mean(cers):.4f}"
        )
        print(f"  Saved → {out_path}")

    print("\nDone ✓")


if __name__ == "__main__":
    main()
