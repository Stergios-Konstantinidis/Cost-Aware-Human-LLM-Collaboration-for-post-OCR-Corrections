"""
regression_features.py
Shared feature-engineering utilities for the OCR regression experiments.

Features computed per (groundtruth_text, raw_ocr_text) pair:
  ── Text-surface features (40) ──
  1.  text_length            : number of characters in raw OCR text
  2.  word_count             : number of whitespace-separated words
  3.  avg_word_length        : average word length
  4.  unique_char_ratio      : #unique chars / #total chars
  5.  digit_ratio            : fraction of characters that are digits
  6.  punct_ratio            : fraction of characters that are punctuation
  7.  upper_ratio            : fraction of alpha chars that are uppercase
  8.  newline_density        : newlines / total characters
  9.  space_ratio            : spaces / total characters
  10–35. freq_<letter>       : occurrences of letter / total alpha chars
  36. max_run_length
  37. avg_run_length
  38. ortho_wer_delta        : WER after spell-correct − WER before
  39. ortho_cer_delta        : CER after spell-correct − CER before

  ── Sentence-transformer embedding (384-d) ──
  40–423. emb_0 … emb_383   paraphrase-multilingual-MiniLM-L12-v2

  ── Metadata features (13) ──
  424. num_lines             : number of newline-separated lines in raw OCR
  425. avg_chars_per_line    : layout-density proxy
  426. publication_year      : integer year extracted from date field
  427–435. newspaper_*       : 9-class one-hot newspaper encoding
  436. avg_confidence        : per-document average OCR confidence score
                               (sourced from low_confidence_words_80_{engine}.json;
                                defaults to 1.0 when unavailable)

  Total: 40 surface + 384 embeddings + 13 metadata = 437 (or 54 without embeddings)
"""

import re
import string
import unicodedata
import warnings
from typing import List, Tuple

import jiwer
import numpy as np

LETTERS = list(string.ascii_lowercase)


# ─── annotator normalisation (identical to run_evaluations.py) ──────────────
def apply_annotator_rules(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.replace("E'", "É").replace("E`", "É")
    text = text.replace("&z", "&")
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+([;.,!?:])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(string.punctuation + " ")


def compute_wer_cer(gt: str, hyp: str) -> Tuple[float, float]:
    try:
        g = apply_annotator_rules(gt) or "[EMPTY]"
        h = apply_annotator_rules(hyp) or "[EMPTY]"
        return jiwer.wer(g, h), jiwer.cer(g, h)
    except Exception:
        return 1.0, 1.0


# ─── spellchecker (lazy singleton) ──────────────────────────────────────────
_spell = None


def _get_spell():
    global _spell
    if _spell is None:
        from spellchecker import SpellChecker
        _spell = SpellChecker(language="fr")
    return _spell


def spell_correct(text: str) -> str:
    spell = _get_spell()
    tokens = re.split(r"(\W+)", text)
    res = []
    for t in tokens:
        if t.isalpha():
            c = spell.correction(t.lower())
            if c:
                if t.isupper():
                    res.append(c.upper())
                elif t.istitle():
                    res.append(c.title())
                else:
                    res.append(c)
            else:
                res.append(t)
        else:
            res.append(t)
    return "".join(res)


# ─── sentence-transformer embedding (lazy singleton) ─────────────────────────
_embedder = None
_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _embedder = SentenceTransformer(_EMBED_MODEL)
    return _embedder


def embed_texts(texts: List[str]) -> np.ndarray:
    """Return (N, 384) float32 array."""
    model = _get_embedder()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vecs = model.encode(texts, batch_size=64, show_progress_bar=False,
                            convert_to_numpy=True)
    return vecs.astype(np.float32)


# ─── surface / structural features ───────────────────────────────────────────
def _letter_freqs(text: str) -> np.ndarray:
    """26-d vector: count of each a-z letter / total alpha chars (or 0)."""
    lc = text.lower()
    alpha = [c for c in lc if c.isalpha()]
    total = max(len(alpha), 1)
    return np.array([alpha.count(l) / total for l in LETTERS], dtype=np.float32)


def _run_stats(text: str) -> Tuple[float, float]:
    """(max_run_length, avg_run_length) over consecutive identical characters."""
    if not text:
        return 0.0, 0.0
    runs = [len(m.group()) for m in re.finditer(r"(.)\1*", text)]
    return float(max(runs)), float(np.mean(runs))


def surface_features(raw_ocr: str) -> np.ndarray:
    """Return 1-D array of 40 surface/structural features.
    Requires ONLY the raw OCR text — no ground truth at any stage.
    """
    t = raw_ocr
    n = max(len(t), 1)

    # Basic counts
    words = t.split()
    word_count = len(words)
    avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0
    unique_chars = len(set(t))
    digits = sum(c.isdigit() for c in t)
    puncts = sum(c in string.punctuation for c in t)
    alphas = sum(c.isalpha() for c in t)
    uppers = sum(c.isupper() for c in t if c.isalpha())
    newlines = t.count("\n")
    spaces = t.count(" ")

    surface = np.array([
        len(t),                                   # 0  text_length
        word_count,                               # 1  word_count
        avg_word_len,                             # 2  avg_word_length
        unique_chars / n,                         # 3  unique_char_ratio
        digits / n,                               # 4  digit_ratio
        puncts / n,                               # 5  punct_ratio
        uppers / max(alphas, 1),                  # 6  upper_ratio
        newlines / n,                             # 7  newline_density
        spaces / n,                               # 8  space_ratio
    ], dtype=np.float32)

    letter_f = _letter_freqs(t)                  # 9-34  freq_a … freq_z
    max_run, avg_run = _run_stats(t)
    run_f = np.array([max_run, avg_run], dtype=np.float32)  # 35-36

    # ── Orthographic Integrity + Spell-Length Faithfulness (37-39) ────────────
    # All three features below are computed from the OCR text + a French
    # spell-checker ONLY.  NO ground truth is consulted at any point.
    #
    # spell_length_ratio (37): len(raw_ocr) / len(spell_corrected_ocr).
    #   Captures how much the corrector expands/contracts the token stream.
    #   1.0 = no length change; <1.0 = corrector added chars; >1.0 = shrunk.
    #   Distinct from ortho_integrity_char (which measures edit similarity,
    #   not length ratio) and from ortho_integrity_word (word-level count).
    #
    # ortho_integrity_word (38): fraction of words the spell-corrector leaves
    #   unchanged.  1.0 = every word is a valid French word; 0.0 = all flagged.
    #
    # ortho_integrity_char (39): character-level SequenceMatcher similarity
    #   between raw OCR and spell-corrected text.  1.0 = identical.
    ortho_text  = spell_correct(t)  # computed once, reused for all three below
    raw_words   = t.split()
    corr_words  = ortho_text.split()

    spell_length_ratio = len(t) / max(len(ortho_text), 1)          # (37) GT-free

    n_changed   = (
        sum(1 for a, b in zip(raw_words, corr_words) if a != b)
        + abs(len(raw_words) - len(corr_words))
    )
    ortho_integrity_word = 1.0 - n_changed / max(len(raw_words), 1)  # (38)

    import difflib
    matcher = difflib.SequenceMatcher(None, t, ortho_text, autojunk=False)
    ortho_integrity_char = matcher.ratio()                            # (39)

    # NEW (40): Dictionary Hit Rate
    spell = _get_spell()
    alpha_words = [w.strip(string.punctuation).lower() for w in raw_words if w.strip(string.punctuation).isalpha()]
    if alpha_words:
        known_words = spell.known(alpha_words)
        dict_hit_rate = len(known_words) / len(alpha_words)
    else:
        dict_hit_rate = 0.0

    ortho_f = np.array(
        [spell_length_ratio, ortho_integrity_word, ortho_integrity_char, dict_hit_rate],
        dtype=np.float32
    )

    return np.concatenate([surface, letter_f, run_f, ortho_f])





# ─── metadata features ───────────────────────────────────────────────────────
# Sorted list of all newspapers in the dataset — fixed mapping for stable encoding
NEWSPAPER_CODES: dict[str, int] = {
    name: i for i, name in enumerate(sorted([
        "ACI", "Feuille d'Avis de Lausanne", "LP", "ME",
        "Nouvelliste Vaudois", "RL", "RLP", "TL", "esta",
    ]))
}


def load_metadata_lookup(groundtruth_path) -> dict[str, dict]:
    """
    Load the groundtruth JSON and return a dict keyed by filename with:
        year         : int   (from the 'date' field, e.g. '1822-12-18' → 1822)
        newspaper_id : int   (label-encoded, see NEWSPAPER_CODES)
        gt_len       : int   (character count of groundtruth_text)
    Call this once and pass the result to build_dataset via records.
    """
    import json
    from pathlib import Path
    with open(groundtruth_path, encoding="utf-8") as f:
        gt = json.load(f)
    lookup = {}
    for item in gt:
        try:
            year = int(item["date"][:4])
        except (KeyError, ValueError):
            year = 0
        lookup[item["filename"]] = {
            "year":          year,
            "newspaper_id":  NEWSPAPER_CODES.get(item.get("newspaper", ""), -1),
            "gt_len":        len(item.get("groundtruth_text", "")),
        }
    return lookup


def metadata_features(record: dict) -> np.ndarray:
    """
    Return 13-d float32 array of metadata features — ALL GT-free:
      [num_lines, avg_chars_per_line, publication_year,
       newspaper_ACI, newspaper_Feuille_dAvis_de_Lausanne, newspaper_LP,
       newspaper_ME, newspaper_Nouvelliste_Vaudois, newspaper_RL,
       newspaper_RLP, newspaper_TL, newspaper_esta,
       avg_confidence]
    Newspaper is one-hot encoded (no ordinal assumption).
    avg_confidence is the per-document average OCR confidence score injected
    by enrich_records(); defaults to 1.0 when unavailable.
    """
    raw_ocr       = record.get("raw_ocr", "")
    num_lines     = float(len(raw_ocr.split("\n")))
    # avg_chars_per_line: character density per line — GT-free, non-redundant
    # (distinct from text_length which is total chars, and num_lines which is line count)
    avg_chars_per_line = len(raw_ocr) / max(num_lines, 1.0)
    year          = float(record.get("_year", 0))
    nid           = int(record.get("_newspaper_id", -1))  # -1 → all zeros
    one_hot       = np.zeros(len(NEWSPAPER_NAMES), dtype=np.float32)
    if 0 <= nid < len(NEWSPAPER_NAMES):
        one_hot[nid] = 1.0
    avg_conf = float(record.get("_avg_confidence", 1.0))
    return np.concatenate([[num_lines, avg_chars_per_line, year], one_hot, [avg_conf]])




def load_confidence_lookup(engine: str, results_dir) -> dict:
    """
    Load per-document average OCR confidence scores for a given engine.
    Reads ``low_confidence_words_80_{engine}.json`` (threshold choice is
    irrelevant — avg_confidence is the same across threshold files).

    Returns a dict keyed by filename → float avg_confidence.
    Returns an empty dict if the file does not exist.
    """
    from pathlib import Path
    path = Path(results_dir) / f"confidence_data/low_confidence_words_80_{engine}.json"
    if not path.exists():
        return {}
    import json
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {fname: entry.get("avg_confidence", 1.0) for fname, entry in raw.items()
            if isinstance(entry, dict)}


def enrich_records(
    records: List[dict],
    metadata_lookup: dict,
    confidence_lookup: dict | None = None,
) -> List[dict]:
    """
    Inject metadata fields into each record in-place.
    Records that have no groundtruth entry get default zeros.

    Parameters
    ----------
    confidence_lookup : optional dict mapping filename → avg_confidence.
        When provided (typically from load_confidence_lookup), the value is
        stored as ``_avg_confidence``.  Defaults to 1.0 when missing or None.
    """
    conf = confidence_lookup or {}
    for r in records:
        fname = r.get("filename", "")
        meta  = metadata_lookup.get(fname, {})
        r["_year"]           = meta.get("year", 0)
        r["_newspaper_id"]   = meta.get("newspaper_id", -1)
        r["_gt_len"]         = meta.get("gt_len", len(r.get("groundtruth", "")))
        r["_avg_confidence"] = conf.get(fname, 1.0)
    return records


# ─── feature names ────────────────────────────────────────────────────────────
SURFACE_FEATURE_NAMES = (
    ["text_length", "word_count", "avg_word_length", "unique_char_ratio",
     "digit_ratio", "punct_ratio", "upper_ratio", "newline_density", "space_ratio"]
    + [f"freq_{l}" for l in LETTERS]
    + ["max_run_length", "avg_run_length",
       "spell_length_ratio",      # len(ocr)/len(spell_corrected) — GT-free
       "ortho_integrity_word",    # fraction of words left unchanged by spell-checker
       "ortho_integrity_char",    # SequenceMatcher ratio vs spell-corrected text
       "dict_hit_rate"]           # fraction of alpha words found in PySpellChecker dictionary
)  # 41 features — ALL computable from raw OCR output alone, zero ground-truth dependency

NEWSPAPER_NAMES = sorted([
    "ACI", "Feuille d'Avis de Lausanne", "LP", "ME",
    "Nouvelliste Vaudois", "RL", "RLP", "TL", "esta",
])  # length 9 — used for one-hot encoding
METADATA_FEATURE_NAMES = (
    ["num_lines", "avg_chars_per_line", "publication_year"]
    + [f"newspaper_{n.replace(' ', '_').replace("'", '')}" for n in NEWSPAPER_NAMES]
    + ["avg_confidence"]
)  # 3 scalar + 9 one-hot + 1 confidence = 13 total

EMBED_FEATURE_NAMES = [f"emb_{i}" for i in range(EMBED_DIM)]
# Without embeddings: 54 total  |  With embeddings: 438 total
ALL_FEATURE_NAMES = SURFACE_FEATURE_NAMES + EMBED_FEATURE_NAMES + METADATA_FEATURE_NAMES



# ─── dataset builder ─────────────────────────────────────────────────────────
def build_dataset(
    records: List[dict],
    use_embeddings: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build X, y_wer, y_cer from a list of dicts with keys:
        groundtruth, raw_ocr, wer, cer
    Optionally: _year, _newspaper_id, _gt_len  (injected by enrich_records)

    Returns:
        X     : (N, F) float32  — 44 or 428 features
        y_wer : (N,)  float32
        y_cer : (N,)  float32
    """
    if verbose:
        print(f"  Building features for {len(records)} samples …")

    surface_list  = []
    metadata_list = []
    for r in records:
        surface_list.append(surface_features(r["raw_ocr"]))
        metadata_list.append(metadata_features(r))

    surface_arr  = np.stack(surface_list)   # (N, 40)
    metadata_arr = np.stack(metadata_list)  # (N, 4)

    if use_embeddings:
        if verbose:
            print(f"  Computing {_EMBED_MODEL} embeddings …")
        emb_arr = embed_texts([r["raw_ocr"] for r in records])  # (N, 384)
        X = np.concatenate([surface_arr, emb_arr, metadata_arr], axis=1)  # (N, 428)
    else:
        X = np.concatenate([surface_arr, metadata_arr], axis=1)  # (N, 44)

    # Clip targets to [0, 1]: WER > 1.0 indicates a segmentation failure
    # (the OCR engine produced text unrelated to the GT region), not a
    # meaningful quality signal.  Including unclipped WER=5.0 values would
    # teach the model to associate normal surface features with pathological
    # outputs.  Clipping is methodologically standard for bounded-metric
    # regression (see Levenshtein-ratio literature).
    y_wer = np.clip(
        np.array([r["wer"] for r in records], dtype=np.float32), 0.0, 1.0
    )
    y_cer = np.clip(
        np.array([r["cer"] for r in records], dtype=np.float32), 0.0, 1.0
    )

    return X, y_wer, y_cer
