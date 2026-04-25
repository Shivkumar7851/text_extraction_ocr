
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract
from PIL import Image


try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    _SPACY_OK = True
except Exception:
    _SPACY_OK = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# ABSOLUTE ACCURACY — Levenshtein / CER / WER
# ══════════════════════════════════════════════════════════════════════════════

def _levenshtein(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein (edit) distance between two strings.
    Uses the standard DP approach — O(len(s1) * len(s2)) time.
    No external library required.
    """
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    # Keep only two rows to save memory
    prev = list(range(len(s2) + 1))
    curr = [0] * (len(s2) + 1)

    for i, c1 in enumerate(s1, 1):
        curr[0] = i
        for j, c2 in enumerate(s2, 1):
            if c1 == c2:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j],      # deletion
                                   curr[j - 1],   # insertion
                                   prev[j - 1])   # substitution
        prev, curr = curr, prev

    return prev[len(s2)]


def _normalise_for_comparison(text: str) -> str:
    """
    Lower-case, collapse whitespace, strip punctuation noise.
    Keeps digits and letters — focuses comparison on content words.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)   # drop punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_absolute_accuracy(ocr_text: str, ground_truth: str) -> dict:
    """
    Compare OCR output against a ground-truth string and return:

    ┌─────────────────────────────────────────────────────────────────────┐
    │  metric              description                                    │
    ├─────────────────────────────────────────────────────────────────────┤
    │  char_accuracy       1 − (edit_dist / max(|gt|, |ocr|))  × 100 %   │
    │  cer                 Character Error Rate  (lower = better)        │
    │  word_accuracy       1 − WER                               × 100 % │
    │  wer                 Word Error Rate       (lower = better)        │
    │  levenshtein_distance  raw character-level edit distance           │
    │  overall_accuracy    simple average of char_accuracy & word_acc    │
    └─────────────────────────────────────────────────────────────────────┘

    All scores are 0–100 (percentage).  Comparison is case-insensitive and
    punctuation-normalised so minor OCR noise does not skew results.
    """
    norm_ocr = _normalise_for_comparison(ocr_text)
    norm_gt  = _normalise_for_comparison(ground_truth)

    # ── Character-level ───────────────────────────────────────────────────────
    edit_dist = _levenshtein(norm_ocr, norm_gt)
    max_len   = max(len(norm_gt), len(norm_ocr), 1)
    cer       = round(edit_dist / max_len, 4)
    char_acc  = round((1 - cer) * 100, 2)

    # ── Word-level  (WER = word-edit-distance / |gt_words|) ──────────────────
    ocr_words = norm_ocr.split()
    gt_words  = norm_gt.split()
    word_edit = _levenshtein(ocr_words, gt_words)          # reuse same DP on lists
    gt_word_count = max(len(gt_words), 1)
    wer       = round(word_edit / gt_word_count, 4)
    word_acc  = round(max(0.0, (1 - wer)) * 100, 2)

    overall   = round((char_acc + word_acc) / 2, 2)

    log.info(
        "Absolute accuracy — char_acc=%.2f%%  word_acc=%.2f%%  "
        "CER=%.4f  WER=%.4f  edit_dist=%d",
        char_acc, word_acc, cer, wer, edit_dist,
    )

    return {
        "char_accuracy":         char_acc,        # % — higher is better
        "cer":                   cer,              # 0–1 — lower is better
        "word_accuracy":         word_acc,         # % — higher is better
        "wer":                   wer,              # 0–1 — lower is better
        "levenshtein_distance":  edit_dist,        # raw char edit ops
        "overall_accuracy":      overall,          # simple avg of char+word acc
        "quality": (
            "high"   if overall >= 85 else
            "medium" if overall >= 65 else
            "low"
        ),
        "ground_truth_length":   len(norm_gt),
        "ocr_text_length":       len(norm_ocr),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE  — Heuristic Accuracy (no ground truth)
# ══════════════════════════════════════════════════════════════════════════════

def estimate_accuracy(ocr_result: dict, text_info: dict) -> dict:
    conf = ocr_result.get("avg_confidence", 0)
    words = ocr_result.get("word_count", 0)
    noise_penalty = len(text_info.get("noise_issues", [])) * 2

    score = conf
    if words < 5:
        score -= 10
    score -= noise_penalty
    score = max(0, min(100, round(score, 2)))

    if score >= 85:
        label = "high"
    elif score >= 65:
        label = "medium"
    else:
        label = "low"

    return {
        # ─────────────────────────────────────────────────────────────────
        # This score is ESTIMATED from Tesseract's internal confidence
        # signal minus noise penalties.  It is NOT an absolute accuracy
        # measurement.  To get real accuracy, pass a ground_truth string
        # — the result will then include an "absolute_accuracy" block with
        # CER / WER computed via Levenshtein distance.
        # ─────────────────────────────────────────────────────────────────
        "method":                   "tesseract_confidence",
        "estimated_accuracy_score": score,
        "quality":                  label,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Pre-processing
# ══════════════════════════════════════════════════════════════════════════════

def _measure_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _deskew(gray: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(gray < 128))
    if len(coords) < 100:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def preprocess(image_path: str, doc_type: str) -> tuple[np.ndarray, dict]:
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    sharpness = _measure_sharpness(gray)

    quality: dict[str, Any] = {
        "original_size_hw": [h, w],
        "sharpness": round(sharpness, 2),
        "is_blurry": sharpness < 80,
    }

    if max(h, w) < 1200:
        scale = 1800 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
        quality["upscaled_factor"] = round(scale, 2)

    if doc_type == "screenshot":
        pass
    elif doc_type == "handwritten":
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    else:
        gray = cv2.fastNlMeansDenoising(gray, h=10)

    if doc_type == "printed":
        gray = _deskew(gray)

    if doc_type == "screenshot":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif doc_type == "printed":
        h2, w2 = gray.shape
        big = cv2.resize(gray, (w2 * 2, h2 * 2), interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 41, 15,
        )

    return binary, quality


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Document-type Classifier
# ══════════════════════════════════════════════════════════════════════════════

def _is_screenshot(gray: np.ndarray) -> bool:
    mid_gray = ((gray > 30) & (gray < 225)).sum()
    pct_mid = mid_gray / gray.size
    return pct_mid < 0.15


def classify(image_path: str) -> str:
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if _is_screenshot(gray):
        log.info("Classifier → screenshot  (bimodal pixel distribution)")
        return "screenshot"

    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    widths = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
    heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)

    if len(widths) == 0:
        return "printed"

    w_cv = widths.std() / (widths.mean() + 1e-9)
    h_cv = heights.std() / (heights.mean() + 1e-9)
    variance = (w_cv + h_cv) / 2.0

    if variance > 1.3:
        result = "handwritten"
    elif variance > 0.65:
        result = "mixed"
    else:
        result = "printed"

    log.info("Classifier → %s  (stroke-width CV=%.2f)", result, variance)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — OCR
# ══════════════════════════════════════════════════════════════════════════════

def _tesseract_run(img: np.ndarray, psm: str, oem: str,
                   lang: str = "eng") -> tuple[str, float]:
    config = f"--oem {oem} --psm {psm} -l {lang}"
    pil = Image.fromarray(img)
    raw = pytesseract.image_to_string(pil, config=config)
    data = pytesseract.image_to_data(pil, config=config,
                                     output_type=pytesseract.Output.DICT)
    confs = [int(c) for c in data["conf"]
             if str(c).lstrip("-").isdigit() and int(c) >= 0]
    avg = round(sum(confs) / len(confs), 1) if confs else 0.0
    return raw.strip(), avg


def _tesseract_run_config(img: np.ndarray, config: str) -> tuple[str, float]:
    pil = Image.fromarray(img)
    raw = pytesseract.image_to_string(pil, config=config)
    return raw.strip(), 0.0


def ocr(image: np.ndarray, doc_type: str) -> dict:
    import math
    oem = "3"

    if doc_type == "screenshot":
        WHITELIST = (
            "./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_- "
        )
        wl_text, _ = _tesseract_run_config(
            image, f"--oem 3 --psm 6 -c tessedit_char_whitelist={WHITELIST}"
        )
        _, conf = _tesseract_run(image, "6", oem)
        log.info("OCR complete -- doc_type=screenshot  avg_conf=%.1f  words=%d",
                 conf, len(wl_text.split()))
        return {
            "raw_text": wl_text,
            "avg_confidence": conf,
            "best_psm_mode": "6+whitelist",
            "ocr_engine": f"tesseract-{pytesseract.get_tesseract_version()}",
            "word_count": len(wl_text.split()),
            "is_low_confidence": conf < 50,
            "note": "",
        }

    psm_candidates: dict[str, list[str]] = {
        "printed":     ["6", "4", "3", "11"],
        "handwritten": ["6", "11", "4", "3"],
        "mixed":       ["6", "3", "11", "4"],
    }
    candidates = psm_candidates.get(doc_type, ["6", "4", "3"])

    best_text, best_conf, best_psm = "", 0.0, candidates[0]

    for psm in candidates:
        t, c = _tesseract_run(image, psm, oem)
        words = len(t.split())
        score = c * math.log1p(words)
        best_score = best_conf * math.log1p(len(best_text.split()))
        log.debug("PSM %s -> conf=%.1f  words=%d  score=%.1f", psm, c, words, score)

        if score > best_score:
            best_text, best_conf, best_psm = t, c, psm

        if best_conf >= 75 and len(best_text.split()) >= 5:
            break

    log.info("OCR complete -- doc_type=%s  best_psm=%s  avg_conf=%.1f  words=%d",
             doc_type, best_psm, best_conf, len(best_text.split()))

    return {
        "raw_text": best_text,
        "avg_confidence": best_conf,
        "best_psm_mode": best_psm,
        "ocr_engine": f"tesseract-{pytesseract.get_tesseract_version()}",
        "word_count": len(best_text.split()),
        "is_low_confidence": best_conf < 50,
        "note": (
            "Handwritten/cursive text has inherently low Tesseract accuracy. "
            "For production-grade handwriting recognition consider TrOCR "
            "(HuggingFace, free, offline): "
            "pip install transformers torch  --  model: microsoft/trocr-base-handwritten"
            if doc_type == "handwritten" and best_conf < 50 else ""
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — OCR Noise Correction
# ══════════════════════════════════════════════════════════════════════════════

_CHAR_FIXES: list[tuple[str, str]] = [
    (r"(?<=[A-Za-z_])0(?=[A-Za-z_])", "o"),
    (r"(?<=[A-Za-z_])1(?=[A-Za-z_])", "l"),
    (r"\b([A-Z])1\b", r"\1l"),
    (r"[ \t]+\.", "."),
    (r"[ \t]+,", ","),
    (r"\.[ \t]+jpg", ".jpg"),
    (r"[ \t]{2,}", " "),
    (r"\n{3,}", "\n\n"),
    (r"^[+\-=|~`]{1,2}\s+", ""),
]

_NOISE_PATTERNS: list[str] = [
    r"[|\\]{2,}",
    r"[~`]{3,}",
    r"[^\x00-\x7F]+",
]


def clean(raw: str) -> tuple[str, list[str]]:
    issues: list[str] = []
    text = raw

    for pat in _NOISE_PATTERNS:
        hits = re.findall(pat, text)
        if hits:
            issues.append(f"noise pattern '{pat}' found {len(hits)}x")
        text = re.sub(pat, "", text)

    for pat, rep in _CHAR_FIXES:
        hits = re.findall(pat, text, flags=re.MULTILINE)
        if hits:
            issues.append(f"char-fix '{pat}' applied {len(hits)}x")
        text = re.sub(pat, rep, text, flags=re.MULTILINE)

    return text.strip(), issues


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — Information Extraction
# ══════════════════════════════════════════════════════════════════════════════

_RE_ENTITIES: dict[str, str] = {
    "file_paths":       r"\./[\w/._-]+",
    "numeric_ids":      r"\b\d{4,8}\b",
    "emails":           r"[\w.+-]+@[\w-]+\.[a-z]{2,}",
    "urls":             r"https?://\S+",
    "dates_raw":        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    "currency_amounts": r"\$\s?\d[\d,]*(?:\.\d{2})?",
    "phone_numbers":    r"\b(?:\+?\d[\d\s\-().]{7,14}\d)\b",
    "zip_codes":        r"\b\d{5}(?:-\d{4})?\b",
    "percentages":      r"\b\d+(?:\.\d+)?%",
    "version_numbers":  r"\bv?\d+\.\d+(?:\.\d+)?\b",
}


def _normalise_date(raw: str) -> str:
    parts = re.split(r"[/-]", raw)
    if len(parts) != 3:
        return raw
    try:
        a, b, c = [int(p) for p in parts]
        if c > 31:
            year, month, day = c, a, b
        elif a > 31:
            year, month, day = a, b, c
        else:
            month, day, year = a, b, c
        if year < 100:
            year += 2000
        return f"{year:04d}-{month:02d}-{day:02d}"
    except ValueError:
        return raw


def extract_entities(text: str) -> dict[str, Any]:
    found: dict[str, Any] = {}

    for label, pat in _RE_ENTITIES.items():
        matches = list(dict.fromkeys(re.findall(pat, text)))
        if matches:
            found[label] = matches

    if "dates_raw" in found:
        found["dates_normalised"] = [_normalise_date(d) for d in found["dates_raw"]]

    if _SPACY_OK:
        doc = _nlp(text)
        ner: dict[str, list[str]] = {}
        for ent in doc.ents:
            ner.setdefault(ent.label_, [])
            if ent.text not in ner[ent.label_]:
                ner[ent.label_].append(ent.text)
        if ner:
            found["named_entities_spacy"] = ner

    return found


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — Heuristic Semantic Analysis
# ══════════════════════════════════════════════════════════════════════════════

_CATEGORY_SIGNALS: dict[str, list[str]] = {
    "invoice":          ["invoice", "total", "amount due", "bill to", "tax", "subtotal"],
    "receipt":          ["receipt", "change due", "payment", "thank you for"],
    "file_listing":     ["./", ".jpg", ".png", ".pdf", ".txt", ".csv"],
    "letter":           ["dear ", "sincerely", "regards", "to whom"],
    "form":             ["name:", "date:", "signature", "please fill"],
    "handwritten_note": ["reminder", "matter", "loved", "important", "you are"],
    "report":           ["summary", "conclusion", "findings", "analysis", "figure"],
    "table":            ["\t", "  |  ", "---"],
}

_SENTIMENT_POS = re.compile(
    r"\b(great|good|excellent|loved|important|matter|thank|happy|positive|success)\b", re.I)
_SENTIMENT_NEG = re.compile(
    r"\b(error|fail|bad|wrong|issue|problem|missing|incorrect|noise)\b", re.I)


def _detect_category(text: str) -> str:
    lower = text.lower()
    scores: dict[str, int] = {}
    for cat, signals in _CATEGORY_SIGNALS.items():
        scores[cat] = sum(1 for s in signals if s in lower)
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "document"


def _detect_sentiment(text: str) -> str:
    pos = len(_SENTIMENT_POS.findall(text))
    neg = len(_SENTIMENT_NEG.findall(text))
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def _summary(text: str, doc_type: str, category: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    word_count = len(text.split())
    line_count = len(lines)
    first_line = lines[0][:80] if lines else ""
    return (
        f"{doc_type.capitalize()} {category} with {line_count} lines "
        f"and {word_count} words. First line: \"{first_line}\"."
    )


def analyse(text: str, doc_type: str) -> dict:
    category  = _detect_category(text)
    sentiment = _detect_sentiment(text)

    lines = [l for l in text.splitlines() if l.strip()]
    words = text.split()
    avg_line_len = round(sum(len(l) for l in lines) / max(len(lines), 1), 1)

    freq: dict[str, int] = {}
    for w in re.findall(r"[A-Za-z]{4,}", text.lower()):
        freq[w] = freq.get(w, 0) + 1
    top_words = sorted(freq, key=lambda k: -freq[k])[:10]

    return {
        "document_category":  category,
        "sentiment":          sentiment,
        "summary":            _summary(text, doc_type, category),
        "statistics": {
            "total_words":    len(words),
            "total_lines":    len(lines),
            "avg_line_length": avg_line_len,
            "unique_words":   len(set(w.lower() for w in words)),
        },
        "top_keywords": top_words,
        "language_hint": "en" if re.search(r"\b(the|and|for|you|are)\b", text, re.I) else "unknown",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def process(image_path: str, output_dir: str = "ocr_outputs",
            force_type: str | None = None,
            ground_truth: str | None = None) -> dict:
    
    image_path = str(Path(image_path).resolve())
    log.info("━" * 60)
    log.info("Processing: %s", image_path)

    result: dict[str, Any] = {
        "pipeline_version": "2.1.0",
        "api_dependencies":  "none",
        "processed_at":     datetime.now(timezone.utc).isoformat(),
        "source_file":      image_path,
        "file_hash_md5":    _md5(image_path),
    }

    doc_type = force_type or classify(image_path)
    result["document_type"] = doc_type

    processed_img, quality = preprocess(image_path, doc_type)
    result["image_quality"] = quality

    ocr_result = ocr(processed_img, doc_type)
    result["ocr"] = ocr_result
    raw_text = ocr_result["raw_text"]

    cleaned, noise_issues = clean(raw_text)
    result["text"] = {
        "original":   raw_text,
        "cleaned":    cleaned,
        "noise_issues": noise_issues,
        "char_count": len(cleaned),
        "line_count": cleaned.count("\n") + 1,
    }

    result["confidence_estimate"] = estimate_accuracy(result["ocr"], result["text"])

    # ── Absolute accuracy (only when ground_truth is provided) ────────────────
    if ground_truth is not None:
        if not ground_truth.strip():
            log.warning(
                "ground_truth is an empty/whitespace string for %s — "
                "absolute_accuracy will NOT be calculated. "
                "Pass None to silence this warning, or provide real reference text.",
                image_path,
            )
        else:
            result["absolute_accuracy"] = compute_absolute_accuracy(cleaned, ground_truth)
            log.info(
                "Absolute accuracy: overall=%.2f%%  char=%.2f%%  word=%.2f%%",
                result["absolute_accuracy"]["overall_accuracy"],
                result["absolute_accuracy"]["char_accuracy"],
                result["absolute_accuracy"]["word_accuracy"],
            )

    result["extracted_entities"]  = extract_entities(cleaned)
    result["semantic_analysis"]   = analyse(cleaned, doc_type)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stem     = Path(image_path).stem
    out_path = Path(output_dir) / f"{stem}_ocr.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info("✓  Written → %s", out_path)

    return result


def batch(file_paths: list[str], output_dir: str = "ocr_outputs") -> list[dict]:
    results = []
    for path in file_paths:
        try:
            r = process(path, output_dir=output_dir)
            results.append({"file": path, "status": "success", "result": r})
        except Exception as exc:
            log.error("FAILED %s — %s", path, exc)
            results.append({"file": path, "status": "error", "error": str(exc)})

    summary_path = Path(output_dir) / "batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Batch summary → %s", summary_path)
    return results


def process_with_ground_truth(
    gt_records: list[dict],
    output_dir: str = "ocr_outputs",
    force_type: str | None = None,
) -> list[dict]:
    """
    Accept a list of ground-truth records in the format:

        [
          { "image": "invoice1.jpg", "ground_truth": "Invoice No 12345 Total Amount 5000" },
          ...
        ]

    Runs the full pipeline for every record and returns a list of results,
    each augmented with ``absolute_accuracy`` metrics.

    A consolidated ``gt_summary.json`` is written to ``output_dir``.
    """
    if not gt_records:
        raise ValueError("gt_records list is empty")

    all_results: list[dict] = []

    for rec in gt_records:
        image_path = rec.get("image", "")
        gt_text    = rec.get("ground_truth", "")

        if not image_path:
            log.warning("Record missing 'image' key — skipping: %s", rec)
            all_results.append({"record": rec, "status": "skipped",
                                 "reason": "missing 'image' key"})
            continue

        try:
            result = process(
                image_path,
                output_dir=output_dir,
                force_type=force_type,
                ground_truth=gt_text if gt_text else None,
            )
            all_results.append({
                "image":      image_path,
                "status":     "success",
                "absolute_accuracy": result.get("absolute_accuracy"),
                "full_result": result,
            })
        except Exception as exc:
            log.error("FAILED %s — %s", image_path, exc)
            all_results.append({
                "image":  image_path,
                "status": "error",
                "error":  str(exc),
            })

    # ── Write consolidated summary ─────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summary_path = Path(output_dir) / "gt_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log.info("Ground-truth summary → %s", summary_path)

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _emit(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="OCR Pipeline — offline, no API key required",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python ocr_pipeline.py invoice.jpg

  # Batch
  python ocr_pipeline.py scan1.png scan2.png --output-dir results/

  # Force doc type
  python ocr_pipeline.py note.jpg --type handwritten

  # Ground-truth JSON (absolute accuracy)
  python ocr_pipeline.py --gt-file ground_truths.json
  python ocr_pipeline.py --gt-json '[{"image":"inv.jpg","ground_truth":"Invoice No 12345"}]'
        """,
    )
    parser.add_argument("images", nargs="*", help="Image file(s) to process")
    parser.add_argument("--output-dir", default="ocr_outputs",
                        help="Directory for JSON output files (default: ocr_outputs/)")
    parser.add_argument("--type", choices=["printed", "handwritten", "mixed"],
                        default=None, dest="force_type",
                        help="Override auto document-type detection")

    # ── Ground-truth modes ────────────────────────────────────────────────────
    gt_group = parser.add_mutually_exclusive_group()
    gt_group.add_argument(
        "--gt-file", metavar="FILE",
        help="Path to a JSON file containing ground-truth records "
             '[{"image": "...", "ground_truth": "..."}]',
    )
    gt_group.add_argument(
        "--gt-json", metavar="JSON",
        help="Inline JSON string of ground-truth records",
    )
    gt_group.add_argument(
        "--ground-truth", metavar="TEXT",
        help="Ground-truth string for a single image (use with exactly one positional image)",
    )

    args = parser.parse_args()

    try:
        # ── Ground-truth batch from file ──────────────────────────────────────
        if args.gt_file:
            with open(args.gt_file, encoding="utf-8") as f:
                records = json.load(f)
            results = process_with_ground_truth(
                records, output_dir=args.output_dir, force_type=args.force_type
            )
            _emit(results)

        # ── Ground-truth batch from inline JSON ───────────────────────────────
        elif args.gt_json:
            records = json.loads(args.gt_json)
            results = process_with_ground_truth(
                records, output_dir=args.output_dir, force_type=args.force_type
            )
            _emit(results)

        # ── Single image with --ground-truth ─────────────────────────────────
        elif args.ground_truth:
            if len(args.images) != 1:
                parser.error("--ground-truth requires exactly one positional image")
            result = process(
                args.images[0],
                output_dir=args.output_dir,
                force_type=args.force_type,
                ground_truth=args.ground_truth,
            )
            _emit(result)

        # ── Normal (no ground truth) ──────────────────────────────────────────
        elif len(args.images) == 1:
            result = process(args.images[0], output_dir=args.output_dir,
                             force_type=args.force_type)
            _emit(result)
        elif args.images:
            results = batch(args.images, output_dir=args.output_dir)
            _emit(results)
        else:
            parser.print_help()

    except Exception as exc:
        _emit({
            "status": "error",
            "error": str(exc),
            "files": args.images,
        })
        sys.exit(1)


if __name__ == "__main__":
    _cli()