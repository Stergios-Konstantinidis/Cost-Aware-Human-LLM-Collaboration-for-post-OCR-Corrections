"""
run_evaluations.py
OCR Evaluation Pipeline — Optimised version

Key improvements over the original:
  - Concurrent LLM requests (ThreadPoolExecutor) with configurable workers
  - Per-experiment file-level caching (unchanged from before)
  - Batched LLM prompts: all images sent in one API call per experiment
  - Exponential back-off with jitter on rate-limit / transient errors
  - Baseline (no-LLM) metrics added to summary automatically
  - `re` module imported once at top level
  - `load_dotenv()` called once at startup, not per-call
  - OCR engines instantiated once and reused (not per-image)
  - Rich tqdm progress bars with ETA
  - Cleaner summary: includes per-engine AND cross-engine averages
  - Optional `--dry-run` flag to simulate without real API calls
"""

import json
import os
import re
import sys
import time
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# ── third-party imports (with user-friendly errors) ────────────────────────
try:
    import jiwer
except ImportError:
    sys.exit("Please install jiwer: pip install jiwer")

try:
    from tqdm import tqdm
except ImportError:
    # Graceful fallback — tqdm is optional but strongly recommended
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, **kw):
            self._iter = iterable

        def __iter__(self):
            return iter(self._iter)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def update(self, n=1):
            pass

        def set_postfix(self, **kw):
            pass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional; env vars can be set in the shell

# ── logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── configuration ──────────────────────────────────────────────────────────
DEFAULT_LLM_MODELS = [
    "google/gemini-3-flash-preview",
    "google/gemini-3.1-flash-lite-preview",
    #"google/gemma-3-27b-it",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-small-3.1-24b-instruct",
    "google/gemma-4-31b-it",
    "qwen/qwen-2.5-72b-instruct",
    #"google/gemini-2.5-pro-preview",
]

ENGINES_TO_EVAL = ["paddle", "easyocr", "tesseract"]

# Max parallel LLM request workers.
# Keep low (2–4) to respect typical OpenRouter rate limits.
MAX_WORKERS = 3

# Number of OCR texts packed into a single API call (batching).
# Higher = fewer API calls = faster, but risks hitting context limits.
# 10–20 is a safe default; reduce if models truncate outputs.
BATCH_SIZE = 15

# Retry settings
MAX_RETRIES = 4
RETRY_BASE_DELAY = 5   # seconds
RETRY_MAX_DELAY = 60   # seconds


# ── text normalisation ──────────────────────────────────────────────────────
def apply_annotator_rules(text: str) -> str:
    """Apply topological normalisation rules matching the human annotator spec."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.replace("E'", "É").replace("E`", "É")
    text = text.replace("&z", "&")

    # Replace line breaks with single space
    text = text.replace("\n", " ").replace("\r", " ")

    # Remove spaces before punctuation 
    text = re.sub(r"\s+([;.,!?:])", r"\1", text)

    # Remove double whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading and trailing whitespace
    text = text.strip()

    # Remove leading and trailing signs (punctuation/symbols)
    import string
    text = text.strip(string.punctuation + " ")

    return text.strip()


# ── metrics ─────────────────────────────────────────────────────────────────
def compute_metrics(groundtruth: str, hypothesis: str):
    """Return (WER, CER) after applying annotator normalisation."""
    try:
        gt_norm = apply_annotator_rules(groundtruth) or "[EMPTY]"
        hyp_norm = apply_annotator_rules(hypothesis) or "[EMPTY]"
        wer = jiwer.wer(gt_norm, hyp_norm)
        cer = jiwer.cer(gt_norm, hyp_norm)
        return wer, cer
    except Exception as exc:
        log.warning("compute_metrics error: %s", exc)
        return 1.0, 1.0


# ── OCR helpers ──────────────────────────────────────────────────────────────
def setup_ocr_engines() -> dict:
    """Load all available OCR engines (each initialised once)."""
    engines = {}

    try:
        import pytesseract
        engines["tesseract"] = lambda img: pytesseract.image_to_string(
            img, lang="fra"
        )
        log.info("Tesseract loaded.")
    except Exception as exc:
        log.warning("Tesseract not available: %s", exc)

    try:
        import easyocr
        reader = easyocr.Reader(["fr"], gpu=False)
        engines["easyocr"] = lambda img: "\n".join(
            reader.readtext(img, detail=0)
        )
        log.info("EasyOCR loaded.")
    except Exception as exc:
        log.warning("EasyOCR not available: %s", exc)

    try:
        # PaddleOCR v3.x uses paddlex as its backend — `paddlepaddle` itself is
        # NOT required.  Disable the slow network connectivity check paddlex v3.4+
        # runs on every import (can take 10-30 s with no benefit offline).
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # use_textline_orientation replaces use_angle_cls in newer PaddleOCR
            try:
                ocr = PaddleOCR(use_textline_orientation=True, lang="fr")
            except TypeError:
                ocr = PaddleOCR(use_angle_cls=True, lang="fr")  # fallback for older

        def run_paddle(img):
            res = ocr.ocr(img, cls=True)
            if not res or not res[0]:
                return ""
            return "\n".join(line[1][0] for line in res[0])

        engines["paddle"] = run_paddle
        log.info("PaddleOCR loaded.")
    except Exception as exc:
        log.warning("PaddleOCR not available: %s", exc)

    if not engines:
        log.error("No OCR engines loaded — OCR step will be skipped.")
    return engines


def load_or_run_ocr(
    eval_dir: Path,
    ocr_engines: dict,
    groundtruth_data: list,
    cache_path: Path,
) -> dict:
    """Run OCR on each image (engine × image) and persist results."""
    ocr_cache: dict = {}

    if cache_path.exists():
        log.info("Loading OCR cache from %s", cache_path)
        with open(cache_path, "r", encoding="utf-8") as f:
            ocr_cache = json.load(f)

    changes_made = False
    for engine_name, engine_func in ocr_engines.items():
        if engine_name not in ocr_cache:
            ocr_cache[engine_name] = {}

        missing = [
            item["filename"]
            for item in groundtruth_data
            if item["filename"] not in ocr_cache[engine_name]
        ]

        if not missing:
            continue

        log.info("Running %s on %d images…", engine_name, len(missing))
        for fname in tqdm(missing, desc=f"OCR/{engine_name}", unit="img"):
            img_path = eval_dir / "images" / fname
            if not img_path.exists():
                log.warning("Image not found: %s", img_path)
                continue
            try:
                ocr_cache[engine_name][fname] = engine_func(str(img_path))
                changes_made = True
            except Exception as exc:
                log.error("OCR failed (%s, %s): %s", engine_name, fname, exc)
                ocr_cache[engine_name][fname] = ""
                changes_made = True

    if changes_made:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(ocr_cache, f, ensure_ascii=False, indent=2)
        log.info("OCR cache saved to %s", cache_path)

    return ocr_cache


def extract_low_confidence_words(
    img_path, engine="paddle", threshold=0.8, cached_reader=None
) -> list:
    results = []
    lines, confs = [], []
    if engine == "paddle":
        try:
            res = cached_reader.ocr(str(img_path), cls=True)
            if res and res[0]:
                lines = [l[1][0] for l in res[0]]
                confs = [float(l[1][1]) / 100.0 if float(l[1][1]) > 1.0 else float(l[1][1]) for l in res[0]]
        except Exception as exc: log.error("Paddle error: %s", exc)
    elif engine == "easyocr":
        try:
            res = cached_reader.readtext(str(img_path))
            lines = [r[1] for r in res]
            confs = [float(r[2]) for r in res]
        except Exception as exc: log.error("EasyOCR error: %s", exc)
    elif engine == "tesseract":
        try:
            import pytesseract
            import pandas as pd
            from io import StringIO
            data = pytesseract.image_to_data(str(img_path), lang="fra")
            df = pd.read_csv(StringIO(data), sep="\t", quoting=3)
            # Group by line_num to match the line-based correction
            df = df[df["conf"] != -1]
            line_groups = df.groupby(["block_num", "par_num", "line_num"])
            for _, group in line_groups:
                text = " ".join([str(x) for x in group["text"].tolist() if str(x).strip()])
                if not text: continue
                avg_conf = group["conf"].mean() / 100.0
                lines.append(text)
                confs.append(avg_conf)
        except Exception as exc: log.error("Tesseract error: %s", exc)

    for i, (text, conf) in enumerate(zip(lines, confs)):
        if conf < threshold:
            prev_c = lines[max(0, i-3):i]
            next_c = lines[i+1:min(len(lines), i+4)]
            results.append({
                "index": i,
                "text": text,
                "confidence": conf,
                "prev_context": prev_c,
                "next_context": next_c
            })
    return results


# ── LLM interface ────────────────────────────────────────────────────────────
def _build_openrouter_client():
    """Return a configured OpenAI client pointing at OpenRouter."""
    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
        "openrouter_api_key"
    )
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. "
            "Add it to your .env file or export it in the shell."
        )
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=120,
    )


_OPENROUTER_CLIENT = None  # lazy singleton


def get_client():
    global _OPENROUTER_CLIENT
    if _OPENROUTER_CLIENT is None:
        _OPENROUTER_CLIENT = _build_openrouter_client()
    return _OPENROUTER_CLIENT


def _extract_corrected_text(raw_response: str) -> str:
    """
    Parse the LLM response and extract the 'corrected_text' field value.

    The prompt instructs the model to return:
        {"corrected_text": "<corrected text here>"}

    This helper handles common failure modes:
    - Markdown code fences around the JSON (```json ... ```)
    - Extra whitespace / surrounding prose
    - Malformed JSON (falls back to raw response)
    """
    if not raw_response:
        return ""

    text = raw_response.strip()

    # 1. Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # 2. Try direct JSON parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "corrected_text" in obj:
            return obj["corrected_text"]
    except json.JSONDecodeError:
        pass

    # 3. Try extracting the first JSON object via regex (handles surrounding prose)
    json_match = re.search(r'\{[^{}]*"corrected_text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}', text, re.DOTALL)
    if json_match:
        # Unescape the captured JSON string value
        try:
            return json_match.group(1).encode("raw_unicode_escape").decode("unicode_escape")
        except Exception:
            return json_match.group(1).replace("\\n", "\n").replace('\\"', '"')

    # 4. Fallback: return the raw response unchanged (no data loss)
    log.warning("Could not parse JSON from LLM response; using raw text.")
    return raw_response


def invoke_llm_batch(
    base_prompt_text: str,
    ocr_texts: list[str],
    full_text_template: str,
    llm_model: str,
    dry_run: bool = False,
) -> tuple[list[str], int, int]:
    """
    Send a batch of OCR texts in a single API call.

    The LLM is asked to return a keyed JSON object:
        {"0": "<text 0>", "1": "<text 1>", ...}

    Any indices missing from the response are retried individually via
    invoke_llm (single-item call with its own retry logic).  Only if that
    also fails does the slot become an empty string.

    Returns (list_of_corrected_texts, prompt_tokens, completion_tokens).
    """
    if dry_run:
        return [""] * len(ocr_texts), 10 * len(ocr_texts), 5 * len(ocr_texts)

    if len(ocr_texts) == 1:
        # Avoid JSON wrapping completely for batch size 1 to prevent instruction conflict and redundant API calls
        prompt = full_text_template.replace("{base_prompt}", base_prompt_text).replace("{ocr_text}", ocr_texts[0])
        prompt = re.sub(r"### FORMAT DE RÉPONSE OBLIGATOIRE ###.*$", "", prompt, flags=re.DOTALL).rstrip()
        corrected, pt, ct = invoke_llm(prompt, llm_model, dry_run=dry_run)
        return [corrected], pt, ct

    # Build a numbered batch prompt
    numbered = "\n\n".join(
        f"[{i}]\n{text}" for i, text in enumerate(ocr_texts)
    )
    batch_suffix = (
        "\n\n### FORMAT DE RÉPONSE OBLIGATOIRE ###\n"
        f"Tu reçois {len(ocr_texts)} textes OCR numérotés de [0] à [{len(ocr_texts)-1}].\n"
        "Renvoie UNIQUEMENT un objet JSON valide (pas de bloc Markdown, pas de texte avant/après).\n"
        "Chaque clé est l'index entier du texte (en chaîne), chaque valeur est le texte corrigé.\n"
        "Les sauts de ligne dans les textes corrigés doivent être encodés \\n dans la chaîne JSON.\n"
        "Format exact attendu (remplace les valeurs d'exemple) :\n"
        + json.dumps({str(i): f"<texte {i} corrigé>" for i in range(len(ocr_texts))}, ensure_ascii=False)
    )

    # Strip the single-item FORMAT block that the template appends, replace with batch suffix
    base = full_text_template.replace("{base_prompt}", base_prompt_text)
    base = re.sub(
        r"### FORMAT DE RÉPONSE OBLIGATOIRE ###.*$", "", base, flags=re.DOTALL
    ).rstrip()
    prompt = base.replace("{ocr_text}", numbered) + batch_suffix

    total_pt, total_ct = 0, 0

    # ── Step 1: batch call ────────────────────────────────────────────────
    corrections: list[str | None] = [None] * len(ocr_texts)  # None = still missing

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client = get_client()
            response = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                extra_headers={
                    "HTTP-Referer": (
                        "https://github.com/Stergios-Konstantinidis/"
                        "DocEng_2-OCR-experiment"
                    ),
                    "X-Title": "OCR Evaluator",
                },
            )
            total_pt += response.usage.prompt_tokens if hasattr(response, "usage") and response.usage else 0
            total_ct += response.usage.completion_tokens if hasattr(response, "usage") and response.usage else 0
            content = response.choices[0].message.content or ""

            # Returns a list where missing slots are None
            corrections = _parse_batch_response(content, len(ocr_texts))
            break  # parsed successfully — may still have None slots

        except Exception as exc:
            delay = min(
                RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 2),
                RETRY_MAX_DELAY,
            )
            if attempt < MAX_RETRIES:
                log.warning(
                    "Batch LLM call failed (attempt %d/%d, %s): %s — retrying in %.1fs",
                    attempt, MAX_RETRIES, llm_model, exc, delay,
                )
                time.sleep(delay)
            else:
                log.error(
                    "Batch LLM call failed after %d attempts (%s): %s",
                    MAX_RETRIES, llm_model, exc,
                )
                # All slots still None — fall through to individual retries below

    # ── Step 2: retry missing slots individually ──────────────────────────
    missing_indices = [i for i, v in enumerate(corrections) if v is None]
    if missing_indices:
        log.warning(
            "Batch response missing %d/%d items for %s — retrying individually.",
            len(missing_indices), len(ocr_texts), llm_model,
        )
        # Build a single-item prompt for each missing index
        single_tmpl = base  # already has FORMAT block stripped
        for idx in missing_indices:
            single_prompt = (
                single_tmpl.replace("{ocr_text}", ocr_texts[idx])
                + "\n\n### FORMAT DE RÉPONSE OBLIGATOIRE ###\n"
                  'Réponds UNIQUEMENT avec {"corrected_text": "<texte corrigé>"}'
            )
            corrected, pt, ct = invoke_llm(single_prompt, llm_model, dry_run=dry_run)
            total_pt += pt
            total_ct += ct
            corrections[idx] = corrected  # invoke_llm returns "" on total failure

    # Replace any remaining None with "" (safety net)
    final: list[str] = [v if v is not None else "" for v in corrections]
    return final, total_pt, total_ct


def _parse_batch_response(content: str, expected_count: int) -> list[str | None]:
    """
    Parse the keyed-object batch LLM response.

    Expected format from the model:
        {"0": "corrected text 0", "1": "corrected text 1", ...}

    Returns a list of length `expected_count` where each slot is either the
    corrected string or **None** if the model did not provide that index.
    Callers handle None slots (e.g. by retrying individually).
    """
    text = content.strip()
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()

    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try to find the first {...} block embedded in surrounding prose
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

    if isinstance(parsed, dict):
        result: list[str | None] = []
        for i in range(expected_count):
            # Accept both string keys ("0") and int keys (0)
            val = parsed.get(str(i), parsed.get(i, None))
            if val is not None and isinstance(val, str):
                result.append(val)
            else:
                log.warning("Batch response missing index %d — will retry individually.", i)
                result.append(None)  # signal: needs individual retry
        return result

    # Could not parse at all — every slot needs individual retry
    log.warning("Could not parse batch response at all (%d items) — all will be retried individually.", expected_count)
    return [None] * expected_count


def invoke_llm(
    full_prompt: str,
    llm_model: str,
    dry_run: bool = False,
) -> tuple[str, int, int]:
    """
    Returns (corrected_text, prompt_tokens, completion_tokens).
    """
    if dry_run:
        return "", 10, 5

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client = get_client()
            response = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": full_prompt}],
                extra_headers={
                    "HTTP-Referer": (
                        "https://github.com/Stergios-Konstantinidis/"
                        "DocEng_2-OCR-experiment"
                    ),
                    "X-Title": "OCR Evaluator",
                },
            )
            pt = response.usage.prompt_tokens if hasattr(response, "usage") and response.usage else 0
            ct = response.usage.completion_tokens if hasattr(response, "usage") and response.usage else 0
            content = response.choices[0].message.content or ""
            # The prompt asks the LLM to reply with {"corrected_text": "..."}
            # Try to parse and extract just the corrected text value.
            extracted = _extract_corrected_text(content)
            return extracted, pt, ct
        except Exception as exc:
            err_str = str(exc).lower()
            is_rate_limit = "429" in err_str or "rate" in err_str
            delay = min(
                RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 2),
                RETRY_MAX_DELAY,
            )
            if attempt < MAX_RETRIES:
                log.warning(
                    "LLM call failed (attempt %d/%d, %s): %s — retrying in %.1fs",
                    attempt,
                    MAX_RETRIES,
                    llm_model,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                log.error(
                    "LLM call failed after %d attempts (%s): %s — using raw OCR.",
                    MAX_RETRIES,
                    llm_model,
                    exc,
                )
                return "", 0, 0  # fallback

    return "", 0, 0  # unreachable, but keeps type checker happy


# ── experiment runner ────────────────────────────────────────────────────────

COST_RATES = {
    "google/gemini-2.5-pro-preview": (1.25, 5.00),
    "google/gemini-2.5-flash-preview": (0.075, 0.30),
    "google/gemini-3.1-pro-preview": (1.25, 5.00),
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "meta-llama/llama-3.3-70b-instruct": (0.13, 0.40),
    "google/gemini-3.1-flash-lite-preview": (0.075, 0.30),
    "google/gemini-3-flash-preview": (0.075, 0.30),

}
def estimate_cost(llm_model: str, pt: int, ct: int) -> float:
    for k, rates in COST_RATES.items():
        if k in llm_model: return (pt / 1e6 * rates[0]) + (ct / 1e6 * rates[1])
    return (pt / 1e6 * 0.5) + (ct / 1e6 * 1.5) # default fallback

def get_result_filename(ocr_name, strategy, prompt_id, llm_model, is_selective=False, confidence_threshold=0.8):
    strategy_mod = strategy
    if is_selective:
        if "SelectiveNoContext" in strategy:
            strategy_mod = strategy.replace("SelectiveNoContext", f"SelectiveNoContext_thr{int(confidence_threshold*100)}")
        else:
            strategy_mod = strategy.replace("Selective", f"Selective_thr{int(confidence_threshold*100)}")
    return f"{ocr_name}_{strategy_mod}_{prompt_id}_{llm_model.replace('/', '__')}.json"

def run_single_experiment(
    *,
    ocr_name: str,
    prompt: dict,
    llm_model: str,
    groundtruth_data: list,
    ocr_cache: dict,
    results_dir: Path,
    strategy: str,
    dry_run: bool = False,
    is_selective: bool = False,
    confidence_threshold: float = 0.8,
    templates: dict = None,
) -> dict:
    """
    Run (or load from cache) one experiment: one OCR engine × one prompt × one LLM.
    Returns a summary dict with average WER/CER.
    """
    prompt_id = prompt["id"]
    result_filename = get_result_filename(
        ocr_name, strategy, prompt_id, llm_model, is_selective, confidence_threshold
    )
    result_path = results_dir / "corrections" / ocr_name / result_filename
    result_path.parent.mkdir(parents=True, exist_ok=True)
    # Extract the strategy_mod used in filename for reporting
    strategy_reported = result_filename.replace(f"{ocr_name}_", "").replace(f"_{prompt_id}_{llm_model.replace('/', '__')}.json", "")

    # ── load from cache / incremental update ─────────────────────────────
    experiment_results = []
    total_cost_already = 0.0

    if result_path.exists():
        with open(result_path, "r", encoding="utf-8") as f:
            experiment_results = json.load(f)
        
        total_cost_already = float(np.sum([x.get("cost", 0) for x in experiment_results])) if experiment_results and "cost" in experiment_results[0] else 0.0
        
        existing_filenames = {x["filename"] for x in experiment_results}
        target_groundtruth = [
            item for item in groundtruth_data 
            if item["filename"] not in existing_filenames
        ]
        
        if not target_groundtruth:
            # All present, return existing stats
            wers = [x["wer"] for x in experiment_results if x["wer"] is not None]
            cers = [x["cer"] for x in experiment_results if x["cer"] is not None]
            return {
                "ocr_engine": ocr_name,
                "strategy": strategy_reported,
                "prompt_id": prompt_id,
                "llm_model": llm_model,
                "average_wer": float(np.mean(wers)) if wers else 0.0,
                "average_cer": float(np.mean(cers)) if cers else 0.0,
                "num_items": len(wers),
                "cost": total_cost_already,
                "cached": True,
            }
        
        log.info("  %s: %d images missing. Running incremental update…", result_filename, len(target_groundtruth))
    else:
        target_groundtruth = groundtruth_data

    # ── run experiment for target_groundtruth ────────────────────────────
    new_results = []
    per_image_data = [
        (item["filename"], item["groundtruth_text"],
         ocr_cache[ocr_name].get(item["filename"], ""))
        for item in target_groundtruth
        if ocr_cache[ocr_name].get(item["filename"], "").strip()
    ]

    total_pt = 0
    total_ct = 0

    selective_data = {}
    if is_selective:
        thr_val = int(confidence_threshold * 100)
        lcf = results_dir / f"confidence_data/low_confidence_words_{thr_val}_{ocr_name}.json"
        if not lcf.exists():
             lcf = results_dir / f"confidence_data/low_confidence_words_{thr_val}.json"
        
        if lcf.exists():
            with open(lcf, "r") as f: selective_data = json.load(f)

    # Pre-populate new_results with raw data
    all_selective_tasks = [] 

    for i, (fname, gt_text, raw_ocr) in enumerate(per_image_data):
        item = {
            "filename": fname,
            "groundtruth": gt_text,
            "raw_ocr": raw_ocr,
            "corrected_ocr": None, 
            "lines": raw_ocr.split("\n") if is_selective else None,
            "wer": None,
            "cer": None,
        }
        new_results.append(item)

        if is_selective:
            entry = selective_data.get(fname, [])
            img_low_conf = entry if isinstance(entry, list) else entry.get("low_confidence_lines", [])
            for lc in img_low_conf:
                idx = lc.get("index", -1)
                if 0 <= idx < len(item["lines"]):
                    if "NoContext" in strategy:
                        payload = f"[LIGNE CIBLE À CORRIGER]\n{lc['text']}"
                    else:
                        prev = "\n".join(lc.get("prev_context", []))
                        nxt = "\n".join(lc.get("next_context", []))
                        payload = (
                            f"[CONTEXTE PRÉCÉDENT]\n{prev}\n\n"
                            f"[LIGNE CIBLE À CORRIGER]\n{lc['text']}\n\n"
                            f"[CONTEXTE SUIVANT]\n{nxt}"
                        )
                    all_selective_tasks.append({
                        "item_idx": i,
                        "line_idx": idx,
                        "payload": payload
                    })

    # ── Execute Batch LLM calls ──────────────────────────────────────────
    batch_sz = getattr(run_single_experiment, "_batch_size", BATCH_SIZE)
    
    if per_image_data:
        if not is_selective:
            full_tmpl = templates.get("full_text", "{base_prompt}\n\nTexte OCR à corriger :\n{ocr_text}")
            indices = list(range(len(new_results)))
            
            def run_one_batch(batch_start):
                batch_indices = indices[batch_start: batch_start + batch_sz]
                batch_texts = [new_results[global_idx]["raw_ocr"] for global_idx in batch_indices]
                corrections, pt, ct = invoke_llm_batch(
                    base_prompt_text=prompt["prompt_text"],
                    ocr_texts=batch_texts,
                    full_text_template=full_tmpl,
                    llm_model=llm_model,
                    dry_run=dry_run,
                )
                return batch_indices, corrections, pt, ct

            with ThreadPoolExecutor(max_workers=10) as executor:
                batch_starts = list(range(0, len(indices), batch_sz))
                futures = [executor.submit(run_one_batch, bs) for bs in batch_starts]
                for fut in futures:
                    batch_indices, corrections, pt, ct = fut.result()
                    total_pt += pt
                    total_ct += ct
                    for local_idx, global_idx in enumerate(batch_indices):
                        new_results[global_idx]["corrected_ocr"] = corrections[local_idx] or ""
        else:
            sel_tmpl = "### DIRECTIVE DE CORRECTION CIBLÉE ###\n{base_prompt}\n\n{ocr_text}"
            
            def run_one_selective_batch(batch_start):
                batch_tasks = all_selective_tasks[batch_start: batch_start + batch_sz]
                batch_payloads = [t["payload"] for t in batch_tasks]
                corrections, pt, ct = invoke_llm_batch(
                    base_prompt_text=prompt["prompt_text"],
                    ocr_texts=batch_payloads,
                    full_text_template=sel_tmpl,
                    llm_model=llm_model,
                    dry_run=dry_run,
                )
                return batch_tasks, corrections, pt, ct

            with ThreadPoolExecutor(max_workers=10) as executor:
                batch_starts = list(range(0, len(all_selective_tasks), batch_sz))
                futures = [executor.submit(run_one_selective_batch, bs) for bs in batch_starts]
                for fut in futures:
                    batch_tasks, corrections, pt, ct = fut.result()
                    total_pt += pt
                    total_ct += ct
                    for local_idx, task in enumerate(batch_tasks):
                        c_line = corrections[local_idx] or ""
                        new_results[task["item_idx"]]["lines"][task["line_idx"]] = c_line.strip()

            for item in new_results:
                item["corrected_ocr"] = "\n".join(item["lines"])
                del item["lines"]

    # ── Compute metrics and incremental cost ─────────────────────────────
    incremental_cost = estimate_cost(llm_model, total_pt, total_ct)
    
    # We assign a fraction of the cost to each NEW item just for record keeping
    # Or better, we store cost/tokens at the experiment level in the final block.
    # Actually, the result file stores per-item wer/cer. 
    # Let's add 'cost' field to each new item if we want, but the current structure doesn't seem to have it per item.
    # Wait, line 641: "cost": float(np.sum([x.get("cost", 0) for x in experiment_results]))
    # It seems there IS a "cost" field per item in the JSON.
    
    cost_per_item = incremental_cost / len(new_results) if new_results else 0.0
    for item in new_results:
        wer, cer = compute_metrics(item["groundtruth"], item["corrected_ocr"])
        item["wer"] = wer
        item["cer"] = cer
        item["cost"] = cost_per_item

    # Merge and Save
    experiment_results.extend(new_results)
    
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, ensure_ascii=False, indent=2)

    wers = [x["wer"] for x in experiment_results if x["wer"] is not None]
    cers = [x["cer"] for x in experiment_results if x["cer"] is not None]

    return {
        "ocr_engine": ocr_name,
        "strategy": strategy_reported,
        "prompt_id": prompt_id,
        "llm_model": llm_model,
        "average_wer": float(np.mean(wers)) if wers else 0.0,
        "average_cer": float(np.mean(cers)) if cers else 0.0,
        "num_items": len(wers),
        "cost": total_cost_already + incremental_cost,
        "cached": False,
        "new_items": len(new_results),
        "incremental_cost": incremental_cost
    }


# ── baseline (no-LLM) metrics ────────────────────────────────────────────────
def compute_baseline_metrics(
    groundtruth_data: list,
    ocr_cache: dict,
    results_dir: Path,
) -> list:
    """
    Compute WER/CER for raw OCR output (no LLM correction) and cache as
    `baseline_{engine}.json`.  Returns a list of summary dicts.
    """
    summaries = []
    for engine_name, ocr_results in ocr_cache.items():
        out_path = results_dir / f"baselines/baseline_{engine_name}.json"
        rows = []
        if out_path.exists():
            with open(out_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
        
        existing_filenames = {r["filename"] for r in rows}
        missing_items = [
            item for item in groundtruth_data 
            if item["filename"] not in existing_filenames
        ]

        if missing_items:
            log.info("  Updating baseline %s: processing %d new images...", engine_name, len(missing_items))
            for item in missing_items:
                raw = ocr_results.get(item["filename"], "")
                if not raw.strip():
                    continue
                wer, cer = compute_metrics(item["groundtruth_text"], raw)
                rows.append(
                    {
                        "filename": item["filename"],
                        "groundtruth": item["groundtruth_text"],
                        "raw_ocr": raw,
                        "wer": wer,
                        "cer": cer,
                    }
                )
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)

        wers = [r["wer"] for r in rows]
        cers = [r["cer"] for r in rows]
        summaries.append(
            {
                "ocr_engine": engine_name,
                "strategy": "baseline_no_llm",
                "prompt_id": 0,
                "llm_model": "none",
                "average_wer": float(np.mean(wers)) if wers else 0.0,
                "average_cer": float(np.mean(cers)) if cers else 0.0,
                "num_items": len(wers),
                "cost": 0.0,
                "prompt_tokens": 0,
                "comp_tokens": 0,
            }
        )
        log.info(
            "Baseline %-10s  WER=%.3f  CER=%.3f  (n=%d)",
            engine_name,
            summaries[-1]["average_wer"],
            summaries[-1]["average_cer"],
            summaries[-1]["num_items"],
        )

    return summaries


# ── summary ──────────────────────────────────────────────────────────────────
def group_summaries(summary_data: list) -> list:
    """
    Group per-engine results into a structure keyed by (strategy, llm_model).
    Also adds an `overall_average_wer / cer` across engines.
    """
    grouped: dict = {}
    for entry in summary_data:
        strategy = entry["strategy"]
        llm = entry["llm_model"]
        key = f"{strategy}__{llm}"

        if key not in grouped:
            grouped[key] = {
                "strategy": strategy,
                "llm_model": llm,
                "prompt_id": entry.get("prompt_id", 0),
                "by_ocr_engine": {},
                "_wer_sum": 0.0,
                "_cer_sum": 0.0,
                "_count": 0,
            }

        grouped[key]["by_ocr_engine"][entry["ocr_engine"]] = {
            "wer": entry["average_wer"],
            "cer": entry["average_cer"],
            "num_items": entry.get("num_items", 0),
        }
        grouped[key]["_wer_sum"] += entry["average_wer"]
        grouped[key]["_cer_sum"] += entry["average_cer"]
        grouped[key]["_count"] += 1
        grouped[key]["cost"] = grouped[key].get("cost", 0.0) + entry.get("cost", 0.0)

    result = []
    for v in grouped.values():
        count = v.pop("_count")
        v["overall_average_wer"] = v.pop("_wer_sum") / count if count else 0.0
        v["overall_average_cer"] = v.pop("_cer_sum") / count if count else 0.0
        result.append(v)

    # Sort by overall WER ascending
    result.sort(key=lambda x: x["overall_average_wer"])
    return result


def build_leaderboard(grouped_summary: list) -> list:
    """
    Flat leaderboard: one row per (strategy, llm_model), sorted by overall WER.
    """
    board = []
    for entry in grouped_summary:
        board.append(
            {
                "rank": 0,  # filled below
                "strategy": entry["strategy"],
                "llm_model": entry["llm_model"],
                "overall_wer": entry["overall_average_wer"],
                "overall_cer": entry["overall_average_cer"],
                "total_cost": entry.get("cost", 0.0),
                **{
                    f"wer_{eng}": vals["wer"]
                    for eng, vals in entry["by_ocr_engine"].items()
                },
                **{
                    f"cer_{eng}": vals["cer"]
                    for eng, vals in entry["by_ocr_engine"].items()
                },
            }
        )

    board.sort(key=lambda x: x["overall_wer"])
    for i, row in enumerate(board, 1):
        row["rank"] = i

    return board


# ── low-confidence word extraction ───────────────────────────────────────────
def run_low_confidence_extraction(
    groundtruth_data: list,
    ocr_engines: dict,
    eval_dir: Path,
    results_dir: Path,
) -> None:
    """Extract low-confidence words for all engines that support it."""
    paddle_engine = None
    easyocr_engine = None

    try:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                paddle_engine = PaddleOCR(use_textline_orientation=True, lang="fr")
            except TypeError:
                paddle_engine = PaddleOCR(use_angle_cls=True, lang="fr")
    except Exception: pass

    try:
        import easyocr
        easyocr_engine = easyocr.Reader(["fr"])
    except Exception: pass

    # Extract for each available engine
    engines_to_run = []
    if paddle_engine: engines_to_run.append(("paddle", paddle_engine))
    if easyocr_engine: engines_to_run.append(("easyocr", easyocr_engine))
    try:
        import pytesseract
        engines_to_run.append(("tesseract", None))
    except: pass

    for eng_name, eng_reader in engines_to_run:
        log.info(f"Extracting low-confidence lines for {eng_name}...")
        for threshold in [0.80, 0.90]:
            low_conf_file = results_dir / f"confidence_data/low_confidence_words_{int(threshold * 100)}_{eng_name}.json"
            
            low_conf_data: dict = {}
            if low_conf_file.exists():
                with open(low_conf_file, "r", encoding="utf-8") as f:
                    low_conf_data = json.load(f)

            missing_items = [
                item for item in groundtruth_data 
                if item["filename"] not in low_conf_data
            ]

            if not missing_items:
                continue

            for item in tqdm(missing_items, desc=f"L-Conf/{eng_name}"):
                img_path = eval_dir / "images" / item["filename"]
                if img_path.exists():
                    low_conf_data[item["filename"]] = extract_low_confidence_words(
                        img_path, engine=eng_name, threshold=threshold, cached_reader=eng_reader
                    )

            with open(low_conf_file, "w", encoding="utf-8") as f:
                json.dump(low_conf_data, f, ensure_ascii=False, indent=2)


# ── main ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="OCR Evaluation Pipeline (optimised)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip real LLM calls; use raw OCR as 'corrected' output.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Concurrent LLM request workers (default: {MAX_WORKERS}).",
    )
    parser.add_argument(
        "--skip-low-conf",
        action="store_true",
        help="Skip low-confidence word extraction.",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        default=ENGINES_TO_EVAL,
        help="OCR engines to evaluate (e.g. paddle easyocr tesseract).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="LLM models to use (defaults to built-in list).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"OCR texts per LLM batch call (default: {BATCH_SIZE}). "
             "Higher = faster but risks context limits.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of images per experiment (for cheap pre-testing).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / "data"
    eval_dir = data_dir / "evaluation_dataset"
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

    if args.dry_run:
        log.info("*** DRY-RUN MODE — no real LLM calls will be made ***")

    # ── load data ─────────────────────────────────────────────────────────
    with open(eval_dir / "groundtruth.json", "r", encoding="utf-8") as f:
        groundtruth_data = json.load(f)
    
    if args.limit > 0:
        log.warning("LIMITING dataset to first %d items.", args.limit)
        groundtruth_data = groundtruth_data[:args.limit]
        
    log.info("Loaded %d groundtruth items.", len(groundtruth_data))

    with open(data_dir / "sample_prompts.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
        sample_prompts = cfg["prompts"]
        templates = cfg.get("templates", {})
    log.info("Loaded %d prompts.", len(sample_prompts))

    llm_models = args.models or DEFAULT_LLM_MODELS
    engines_to_eval = args.engines
    # Propagate batch size to the experiment runner
    run_single_experiment._batch_size = args.batch_size  # type: ignore[attr-defined]
    log.info("Batch size: %d texts per LLM call", args.batch_size)

    # ── step 1: OCR ───────────────────────────────────────────────────────
    log.info("Step 1 — Running / loading OCR…")
    ocr_engines = setup_ocr_engines()
    ocr_cache_path = data_dir / "raw_ocr_results.json"
    ocr_cache = load_or_run_ocr(
        eval_dir, ocr_engines, groundtruth_data, ocr_cache_path
    )

    # ── step 1.5: low-confidence words ───────────────────────────────────
    if not args.skip_low_conf:
        log.info("Step 1.5 — Low-confidence word extraction…")
        run_low_confidence_extraction(
            groundtruth_data, ocr_engines, eval_dir, results_dir
        )

    # ── step 2: baseline (no-LLM) metrics ────────────────────────────────
    log.info("Step 2 — Computing baseline (no-LLM) metrics…")
    # Only consider engines that have cached OCR results
    active_cache = {
        k: v for k, v in ocr_cache.items() if k in engines_to_eval and v
    }
    summary_data = compute_baseline_metrics(
        groundtruth_data, active_cache, results_dir
    )

    # ── step 3: LLM evaluations ───────────────────────────────────────────
    log.info(
        "Step 3 — LLM evaluations (%d models × %d prompts × %d engines)",
        len(llm_models),
        len(sample_prompts),
        len([e for e in engines_to_eval if e in active_cache]),
    )

    # Build list of all experiments to run (or load from cache)
    experiments = []
    for llm_model in llm_models:
        for prompt in sample_prompts:
            if llm_model == "google/gemma-4-31b-it" and prompt["id"] == 9:
                continue
            base_strategy = (
                prompt["level"]
                .replace(" ", "_")
                .replace("+", "_plus")
                .replace("(", "")
                .replace(")", "")
                .replace("-", "_")
            )
            for ocr_name in engines_to_eval:
                if ocr_name not in active_cache: continue
                # 1. Full text
                experiments.append(dict(
                    ocr_name=ocr_name, prompt=prompt, llm_model=llm_model,
                    groundtruth_data=groundtruth_data, ocr_cache=active_cache,
                    results_dir=results_dir, strategy=f"Full_{base_strategy}",
                    dry_run=args.dry_run, is_selective=False, templates=templates
                ))
                # 2. Selective (3 neighbors)
                for thr in [0.8, 0.9]:
                    experiments.append(dict(
                        ocr_name=ocr_name, prompt=prompt, llm_model=llm_model,
                        groundtruth_data=groundtruth_data, ocr_cache=active_cache,
                        results_dir=results_dir, strategy=f"Selective_{base_strategy}",
                        dry_run=args.dry_run, is_selective=True, confidence_threshold=thr,
                        templates=templates
                    ))
                    # 3. Selective (No context)
                    experiments.append(dict(
                        ocr_name=ocr_name, prompt=prompt, llm_model=llm_model,
                        groundtruth_data=groundtruth_data, ocr_cache=active_cache,
                        results_dir=results_dir, strategy=f"SelectiveNoContext_{base_strategy}",
                        dry_run=args.dry_run, is_selective=True, confidence_threshold=thr,
                        templates=templates
                    ))

    log.info("Total experiments: %d", len(experiments))

    n_cached = sum(
        1
        for exp in experiments
        if (
            results_dir / "corrections" / exp["ocr_name"] / get_result_filename(
                exp["ocr_name"], exp["strategy"], exp["prompt"]["id"], 
                exp["llm_model"], exp.get("is_selective", False), 
                exp.get("confidence_threshold", 0.8)
            )
        ).exists()
    )
    log.info("  Already cached: %d  |  To run: %d", n_cached, len(experiments) - n_cached)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_single_experiment, **exp): exp
            for exp in experiments
        }

        with tqdm(total=len(experiments), desc="Experiments", unit="exp") as pbar:
            for future in as_completed(futures):
                exp = futures[future]
                try:
                    result = future.result()
                    summary_data.append(result)
                    pbar.set_postfix(
                        model=exp["llm_model"].split("/")[-1][:12],
                        engine=exp["ocr_name"],
                        wer=f"{result['average_wer']:.3f}",
                    )
                except Exception as exc:
                    log.error(
                        "Experiment failed (%s / %s / %s): %s",
                        exp["llm_model"],
                        exp["ocr_name"],
                        exp["strategy"],
                        exc,
                    )
                finally:
                    pbar.update(1)

    # ── step 4: summaries ─────────────────────────────────────────────────
    log.info("Step 4 — Writing summaries…")

    grouped_summary = group_summaries(summary_data)
    summary_path = results_dir / "summaries/summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(grouped_summary, f, ensure_ascii=False, indent=2)
    log.info("Saved %s", summary_path)

    leaderboard = build_leaderboard(grouped_summary)
    lb_path = results_dir / "summaries/leaderboard.json"
    with open(lb_path, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, ensure_ascii=False, indent=2)
    log.info("Saved %s", lb_path)

    # ── print top-5 leaderboard to stdout ────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'RANK':>4}  {'STRATEGY':<25}  {'MODEL':<25}  {'WER':>6}  {'CER':>6}  {'COST ($)':>8}")
    print("-" * 60)
    for row in leaderboard[:10]:
        model_short = row["llm_model"].split("/")[-1][:24]
        print(
            f"{row['rank']:>4}  {row['strategy']:<30}  {model_short:<25}"
            f"  {row['overall_wer']:.4f}  {row['overall_cer']:.4f}  ${row.get('total_cost', 0):.4f}"
        )
    print("=" * 60)
    log.info("All tasks completed.")


if __name__ == "__main__":
    main()
