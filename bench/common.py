"""
bench/common.py

Shared code for every OCR benchmark worker. All three engines use the SAME
image decoding, the SAME timer, the SAME field parser and the SAME output
format, so the only thing that differs between them is the OCR call itself.

This file only uses the standard library, plus OpenCV/numpy (imported lazily
inside decode_image), so the notebook can import it for scoring without
loading any OCR engine.

A worker script only needs to provide two functions:
    load()            -> model object (called once)
    ocr(model, rgb)   -> str, full-card text with one text line per "\n"
and then call run_worker("engine_name", load, ocr, version_fn).
"""

import argparse
import csv
import json
import os
import platform
import re
import sys
import time
import traceback
import unicodedata
from datetime import date
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths and field definitions
# ----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
TRUTH_CSV = DATA_DIR / "data.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# internal field key -> column name in data.csv
FIELDS = {
    "name_bn": "Bangla Name",
    "name_en": "English Name",
    "father_bn": "Father's Name",
    "mother_bn": "Mother's Name",
    "dob": "DoB",
    "nid": "NID",
}
FIELD_KEYS = list(FIELDS)

# ----------------------------------------------------------------------------
# Ground truth and images
# ----------------------------------------------------------------------------


def read_truth(csv_path=TRUTH_CSV):
    """Return {image_id: {field_key: value}}. Everything is read as text."""
    truth = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            image_id = (row.get("image_id") or "").strip()
            if not image_id:
                continue
            truth[image_id] = {k: (row.get(col) or "").strip() for k, col in FIELDS.items()}
    return truth


def list_images(image_dir=IMAGE_DIR, only_ids=None):
    """Return [(image_id, path)] where image_id is the file name without extension."""
    files = {
        p.stem: p
        for p in sorted(Path(image_dir).iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    }
    if only_ids is not None:
        return [(i, files[i]) for i in only_ids if i in files]
    return list(files.items())


def load_image_bytes(image_dir=IMAGE_DIR, only_ids=None):
    """
    Read every image file into memory as raw bytes, BEFORE any timing starts.
    This simulates the server having already received the upload, so disk
    speed is not counted as OCR speed.
    """
    return [(image_id, path.read_bytes()) for image_id, path in list_images(image_dir, only_ids)]


def decode_image(data: bytes):
    """Uploaded file bytes -> RGB uint8 numpy array. Part of the timed request."""
    import cv2
    import numpy as np

    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode image bytes")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ----------------------------------------------------------------------------
# Turning detector boxes into text lines
# ----------------------------------------------------------------------------


def boxes_to_lines(items, row_tol=0.5):
    """
    EasyOCR and PaddleOCR return many small boxes. The parser needs text lines
    in reading order. items: list of (box, text), box = 4 points [[x, y], ...].

    Two boxes are on the same line when their vertical centres differ by less
    than row_tol x the median box height. Lines are then read left to right.
    """
    boxes = []
    for box, text in items:
        text = (text or "").strip()
        if not text:
            continue
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        boxes.append(((min(ys) + max(ys)) / 2.0, min(xs), max(ys) - min(ys), text))
    if not boxes:
        return ""

    heights = sorted(b[2] for b in boxes)
    tol = max(heights[len(heights) // 2] * row_tol, 1.0)

    boxes.sort(key=lambda b: b[0])
    lines, current = [], [boxes[0]]
    for b in boxes[1:]:
        line_centre = sum(x[0] for x in current) / len(current)
        if abs(b[0] - line_centre) <= tol:
            current.append(b)
        else:
            lines.append(current)
            current = [b]
    lines.append(current)

    return "\n".join(" ".join(b[3] for b in sorted(line, key=lambda b: b[1])) for line in lines)


# ----------------------------------------------------------------------------
# Text cleaning
# ----------------------------------------------------------------------------

BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"), None)


def clean_text(s):
    """Unicode-normalise, drop zero-width characters, Bangla digits -> Latin digits."""
    s = unicodedata.normalize("NFC", s or "")
    s = s.translate(ZERO_WIDTH).translate(BN_DIGITS)
    return re.sub(r"[ \t]+", " ", s).strip()


# ----------------------------------------------------------------------------
# Dates
# ----------------------------------------------------------------------------

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}

# Bangla month names as printed on older paper NIDs.
BN_MONTHS = {
    "জানুয়ারি": 1, "ফেব্রুয়ারি": 2, "মার্চ": 3, "এপ্রিল": 4, "মে": 5, "জুন": 6,
    "জুলাই": 7, "আগস্ট": 8, "সেপ্টেম্বর": 9, "অক্টোবর": 10, "নভেম্বর": 11, "ডিসেম্বর": 12,
}

# Characters OCR commonly substitutes inside month names, folded before matching.
MONTH_FIXES = str.maketrans({"0": "o", "1": "l", "5": "s", "8": "b",
                             "]": "l", "|": "l", "[": "l", "!": "l", "€": "e", "@": "a"})

# Separators are deliberately loose: OCR inserts commas, dots, dashes and stray spaces.
_DSEP = r"[\s,.\-/]*"
DATE_RE = re.compile(r"(\d{1,2})" + _DSEP + r"([A-Za-z\]\|\[!0-9]{3,9})" + _DSEP + r"(\d{4}|\d{2})\b")
BN_DATE_RE = re.compile(r"(\d{1,2})" + _DSEP + r"([ঀ-৿]{2,12})" + _DSEP + r"(\d{4}|\d{2})\b")
NUM_DATE_RE = re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4}|\d{2})\b")
ISO_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")


def _match_month(token):
    """
    Map an OCR'd month token to 1-12. Exact prefix first, then a one-edit
    fallback so 'Ju]' / 'Dee' / '0ct' still resolve. Returns None if unsure.
    """
    t = token.lower().translate(MONTH_FIXES)
    t = re.sub(r"[^a-z]", "", t)[:3]
    if len(t) < 3:
        return None
    if t in MONTHS:
        return MONTHS[t]
    best, best_d = None, 99
    for name, num in MONTHS.items():
        d = sum(1 for a, b in zip(t, name) if a != b)
        if d < best_d:
            best, best_d = num, d
        elif d == best_d:
            best = None          # ambiguous tie -> refuse to guess
    return best if best_d <= 1 and best is not None else None


def _full_year(y):
    """
    '1957' -> 1957. Two-digit years need a guess: NID holders are 18+, so
    anything up to (this year - 16) is read as 20xx, everything else as 19xx.
    """
    if len(y) == 4:
        return int(y)
    yy = int(y)
    cutoff = (date.today().year - 16) % 100
    return 2000 + yy if yy <= cutoff else 1900 + yy


def _build(day, month, year):
    try:
        return date(_full_year(year), month, int(day)).isoformat()
    except (ValueError, TypeError):
        return ""


def normalize_date(s):
    """Any supported date string -> 'YYYY-MM-DD', or '' if nothing valid is found."""
    s = clean_text(s)

    m = ISO_RE.search(s)
    if m:
        got = _build(m.group(3), int(m.group(2)), m.group(1))
        if got:
            return got

    for m in DATE_RE.finditer(s):                       # 10 Dec 1989 / 10-Dec-89 / 22 Ju] 1957
        got = _build(m.group(1), _match_month(m.group(2)), m.group(3))
        if got:
            return got

    for m in BN_DATE_RE.finditer(s):                    # ১০ ডিসেম্বর ১৯৮৯ (digits already folded)
        got = _build(m.group(1), BN_MONTHS.get(m.group(2)), m.group(3))
        if got:
            return got

    for m in NUM_DATE_RE.finditer(s):                   # 10/12/1989, assumed day-month-year
        got = _build(m.group(1), int(m.group(2)), m.group(3))
        if got:
            return got

    return ""


# ----------------------------------------------------------------------------
# The field parser (identical for every engine)
# ----------------------------------------------------------------------------

LABELS = {
    "name_bn": r"নাম",
    "name_en": r"name",
    "father_bn": r"পিতা",
    "mother_bn": r"মাতা",
}
# Value separators seen on NIDs: ':' and the Bangla visarga 'ঃ' used as a colon.
_SEP = r"\s*[:ঃ;.\-]*\s*"


def _value_after_label(lines, label):
    """Find a line starting with the label; return the text after it, or the next line."""
    pattern = re.compile(r"^\W{0,3}" + label + _SEP + r"(.*)$", re.IGNORECASE)
    for i, line in enumerate(lines):
        m = pattern.match(line)
        if not m:
            continue
        value = m.group(1).strip()
        if value:
            return value
        for nxt in lines[i + 1:i + 3]:
            if nxt.strip():
                return nxt.strip()
    return ""


def _keep_bangla(s):
    s = re.sub(r"[^\u0980-\u09FF .]", " ", s)
    return re.sub(r"\s+", " ", s).strip(" .")


def _keep_latin(s):
    s = re.sub(r"[^A-Za-z .]", " ", s)
    return re.sub(r"\s+", " ", s).strip(" .")


def parse_nid(lines):
    """
    NID numbers are 10 (smart card), 13 or 17 digits (older cards), sometimes
    printed with spaces. Prefer a number on a line with an ID label, then the
    longest valid length.
    """
    candidates = []
    for line in lines:
        labelled = bool(re.search(r"\b(n?id)\b|আইডি", line, re.IGNORECASE))
        for m in re.finditer(r"\d[\d ]{8,24}\d", line):
            digits = m.group().replace(" ", "")
            if len(digits) in (10, 13, 17):
                candidates.append((labelled, len(digits), digits))
    if not candidates:
        return ""
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    return candidates[0][2]


def parse_dob(lines):
    """Prefer a date on a line mentioning 'Birth'; otherwise the first valid date."""
    birth_lines = [l for l in lines if re.search(r"birth|জন্ম", l, re.IGNORECASE)]
    for line in birth_lines + lines:
        d = normalize_date(line)
        if d:
            return d
    return ""


def parse_fields(text):
    """Full-card OCR text -> {field_key: value}. Missing fields are ''."""
    lines = [clean_text(l) for l in (text or "").splitlines()]
    lines = [l for l in lines if l]
    return {
        "name_bn": _keep_bangla(_value_after_label(lines, LABELS["name_bn"])),
        "name_en": _keep_latin(_value_after_label(lines, LABELS["name_en"])),
        "father_bn": _keep_bangla(_value_after_label(lines, LABELS["father_bn"])),
        "mother_bn": _keep_bangla(_value_after_label(lines, LABELS["mother_bn"])),
        "dob": parse_dob(lines),
        "nid": parse_nid(lines),
    }


# ----------------------------------------------------------------------------
# Scoring helpers (used by the notebook)
# ----------------------------------------------------------------------------


def normalize_for_compare(field, value):
    """Put prediction and ground truth into the same form before comparing."""
    v = clean_text(value)
    if field == "nid":
        return re.sub(r"\D", "", v)
    if field == "dob":
        return normalize_date(v)
    if field == "name_en":
        return re.sub(r"[^a-z]", "", v.lower())
    return re.sub(r"[\s.:ঃ,\-]", "", v)  # Bangla fields


def exact_match(field, pred, truth):
    t = normalize_for_compare(field, truth)
    return bool(t) and normalize_for_compare(field, pred) == t


def char_error_rate(field, pred, truth):
    """Edit distance / length of truth, after normalisation. 0.0 = perfect."""
    p, t = normalize_for_compare(field, pred), normalize_for_compare(field, truth)
    if not t:
        return float("nan")
    prev = list(range(len(t) + 1))
    for i, pc in enumerate(p, 1):
        cur = [i]
        for j, tc in enumerate(t, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (pc != tc)))
        prev = cur
    return prev[-1] / len(t)


# ----------------------------------------------------------------------------
# One request = decode bytes -> OCR -> parse. This is what gets timed.
# ----------------------------------------------------------------------------


def process_request(ocr_fn, model, data):
    t0 = time.perf_counter()
    try:
        rgb = decode_image(data)
        text = ocr_fn(model, rgb) or ""
        fields = parse_fields(text)
        ms = (time.perf_counter() - t0) * 1000.0
        return {"ms": ms, "status": "ok", "error": "", "text": text, "fields": fields}
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000.0
        return {"ms": ms, "status": "error", "error": f"{type(e).__name__}: {e}",
                "text": "", "fields": {k: "" for k in FIELD_KEYS}}


# ----------------------------------------------------------------------------
# Worker entry point (shared by all three worker scripts)
# ----------------------------------------------------------------------------


def _parse_args():
    ap = argparse.ArgumentParser(description="OCR benchmark worker")
    ap.add_argument("--mode", required=True,
                    choices=["check", "accuracy", "latency", "throughput"])
    ap.add_argument("--out", help="output CSV path (not needed for --mode check)")
    ap.add_argument("--images", default=str(IMAGE_DIR))
    ap.add_argument("--truth", default=str(TRUTH_CSV))
    ap.add_argument("--warmup", type=int, default=3, help="untimed requests before measuring")
    ap.add_argument("--repeats", type=int, default=5, help="latency mode: passes over the images")
    ap.add_argument("--duration", type=float, default=60.0, help="throughput mode: seconds")
    ap.add_argument("--start-at", type=float, default=0.0,
                    help="throughput mode: Unix time at which all workers start together")
    ap.add_argument("--worker-id", type=int, default=0)
    return ap.parse_args()


def _write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def run_worker(engine_name, load_fn, ocr_fn, version_fn=None):
    # Windows consoles default to a code page that cannot print Bangla.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = _parse_args()

    t0 = time.perf_counter()
    model = load_fn()
    load_seconds = time.perf_counter() - t0

    meta = {
        "engine": engine_name,
        "version": version_fn() if version_fn else "unknown",
        "mode": args.mode,
        "worker_id": args.worker_id,
        "pid": os.getpid(),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor(),
        "load_seconds": round(load_seconds, 3),
        "threads_env": {k: os.environ.get(k, "") for k in
                        ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS")},
    }

    if args.mode == "check":
        print(json.dumps(meta, ensure_ascii=False))
        return

    if not args.out:
        sys.exit("--out is required for this mode")

    # Only benchmark images that have ground truth, so extra files don't skew results.
    truth_ids = list(read_truth(args.truth)) if Path(args.truth).exists() else None
    images = load_image_bytes(args.images, truth_ids)
    if not images:
        sys.exit("no images found (check --images and that image names match image_id)")
    meta["images"] = len(images)

    # Warm-up: first calls include one-off setup cost, so they are not measured.
    for k in range(args.warmup):
        process_request(ocr_fn, model, images[k % len(images)][1])

    rows = []

    if args.mode == "accuracy":
        cols = ["engine", "image_id", "ms", "status", "error", "raw_text"] + FIELD_KEYS
        for image_id, data in images:
            r = process_request(ocr_fn, model, data)
            rows.append({"engine": engine_name, "image_id": image_id, "ms": round(r["ms"], 2),
                         "status": r["status"], "error": r["error"], "raw_text": r["text"],
                         **r["fields"]})

    elif args.mode == "latency":
        cols = ["engine", "pass", "image_id", "ms", "status"]
        for p in range(args.repeats):
            for image_id, data in images:
                r = process_request(ocr_fn, model, data)
                rows.append({"engine": engine_name, "pass": p, "image_id": image_id,
                             "ms": round(r["ms"], 2), "status": r["status"]})

    else:  # throughput
        cols = ["engine", "worker_id", "seq", "image_id", "ms", "status", "t_done"]
        wait = args.start_at - time.time()
        if wait > 0:
            time.sleep(wait)
        start = time.perf_counter()
        seq = 0
        while time.perf_counter() - start < args.duration:
            image_id, data = images[seq % len(images)]
            r = process_request(ocr_fn, model, data)
            rows.append({"engine": engine_name, "worker_id": args.worker_id, "seq": seq,
                         "image_id": image_id, "ms": round(r["ms"], 2), "status": r["status"],
                         "t_done": round(time.perf_counter() - start, 4)})
            seq += 1
        meta["completed"] = seq
        meta["elapsed_seconds"] = round(time.perf_counter() - start, 3)

    _write_csv(args.out, rows, cols)
    Path(args.out).with_suffix(".json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    # Quick self-test of the parser, no OCR engine needed:  python bench/common.py
    sample = "নাম: আশিষ ঘোষ\nName: ASISH GHOSH\nপিতা: গোকুল চন্দ্র ঘোষ\nমাতা: বীনা পানি ঘোষ\n" \
             "Date of Birth: 10 Dec 1989\nNID No: 1989 0615 1328 36383"
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(parse_fields(sample), ensure_ascii=False, indent=2))
    print(normalize_date("22-Jul-57"), normalize_date("10 Dec 1989"))