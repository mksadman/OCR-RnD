"""
Final environment check for the EasyOCR venv (Kona NID OCR R&D).

Run from the OCR-RnD folder with the .venv-easyocr environment active:
    python check_env.py

Every check prints PASS, WARN or FAIL. The script exits with code 1 if anything FAILs.
"""

import hashlib
import importlib.metadata as md
import re
import subprocess
import sys
import time
from pathlib import Path

# Change this to any real NID photo in your images folder
TEST_IMAGE = Path("images/16.jpg")

MODEL_DIR = Path.home() / ".EasyOCR" / "model"
REQUIRED_MODELS = ["craft_mlt_25k.pth", "bengali.pth"]
BANGLA_CHARS = re.compile(r"[\u0980-\u09FF]")

failures = 0


def report(status, name, detail=""):
    global failures
    if status == "FAIL":
        failures += 1
    line = f"[{status}] {name}"
    if detail:
        line += f"  ->  {detail}"
    print(line)


def version_of(dist_name):
    try:
        return md.version(dist_name)
    except md.PackageNotFoundError:
        return None


print("=" * 70)
print("1. Python and virtual environment")
print("=" * 70)

py = sys.version_info
report("PASS" if (py.major, py.minor) == (3, 12) else "FAIL",
       "Python version", f"{py.major}.{py.minor}.{py.micro}")

in_venv = sys.prefix != sys.base_prefix
report("PASS" if in_venv and ".venv-easyocr" in sys.prefix else "FAIL",
       "Running inside .venv-easyocr", sys.prefix)

print()
print("=" * 70)
print("2. Package versions")
print("=" * 70)

numpy_ver = version_of("numpy")
if numpy_ver and int(numpy_ver.split(".")[0]) >= 2:
    report("PASS", "NumPy 2.x", numpy_ver)
else:
    report("FAIL", "NumPy 2.x", f"found {numpy_ver}; run: pip install --upgrade \"numpy>=2\"")

opencv_dists = {
    name: version_of(name)
    for name in ["opencv-python", "opencv-python-headless",
                 "opencv-contrib-python", "opencv-contrib-python-headless"]
}
installed_cv = {k: v for k, v in opencv_dists.items() if v}
if list(installed_cv) == ["opencv-python-headless"]:
    report("PASS", "Exactly one OpenCV build (headless)", installed_cv["opencv-python-headless"])
else:
    report("FAIL", "Exactly one OpenCV build (headless)",
           f"found {installed_cv or 'none'}; keep only opencv-python-headless")

for dist in ["torch", "torchvision", "easyocr"]:
    ver = version_of(dist)
    report("PASS" if ver else "FAIL", f"{dist} installed", ver or "missing")

print()
print("=" * 70)
print("3. Imports and dependency consistency")
print("=" * 70)

try:
    import numpy
    import cv2
    import torch
    import easyocr
    report("PASS", "numpy, cv2, torch, easyocr import together",
           f"cv2 {cv2.__version__}, torch {torch.__version__}")
except Exception as exc:
    report("FAIL", "Core imports", repr(exc))
    print("\nCannot continue without the core imports.")
    sys.exit(1)

report("PASS", "Compute device",
       "CUDA GPU available" if torch.cuda.is_available() else "CPU only (expected on this laptop)")

pip_check = subprocess.run([sys.executable, "-m", "pip", "check"],
                           capture_output=True, text=True)
if pip_check.returncode == 0:
    report("PASS", "pip check", pip_check.stdout.strip())
else:
    report("FAIL", "pip check", "broken requirements:\n" + pip_check.stdout.strip())

print()
print("=" * 70)
print("4. Model files on disk")
print("=" * 70)


def expected_md5s():
    """Read the expected checksums from EasyOCR's own config."""
    import easyocr.config as cfg
    found = {}

    def walk(obj):
        if isinstance(obj, dict):
            if "filename" in obj and "md5sum" in obj:
                found[obj["filename"]] = obj["md5sum"]
            for value in obj.values():
                walk(value)

    walk(getattr(cfg, "detection_models", {}))
    walk(getattr(cfg, "recognition_models", {}))
    return found


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


md5_table = expected_md5s()
for model in REQUIRED_MODELS:
    path = MODEL_DIR / model
    if not path.exists():
        report("FAIL", f"{model} present", f"not found in {MODEL_DIR}")
        continue
    size_mb = path.stat().st_size / 1e6
    expected = md5_table.get(model)
    actual = file_md5(path)
    if expected is None:
        report("WARN", f"{model} checksum", f"{size_mb:.0f} MB, no expected MD5 in config")
    elif actual == expected:
        report("PASS", f"{model} checksum matches", f"{size_mb:.0f} MB")
    else:
        report("FAIL", f"{model} checksum", f"corrupt: got {actual}, expected {expected}")

leftovers = [p.name for p in MODEL_DIR.glob("*.zip")] if MODEL_DIR.exists() else []
report("PASS" if not leftovers else "WARN", "No leftover partial downloads",
       ", ".join(leftovers) if leftovers else "")

print()
print("=" * 70)
print("5. Reader loads offline")
print("=" * 70)

try:
    t0 = time.perf_counter()
    reader = easyocr.Reader(["bn", "en"], gpu=False,
                            download_enabled=False, verbose=False)
    load_s = time.perf_counter() - t0
    report("PASS", "Reader(['bn','en']) with downloads disabled", f"loaded in {load_s:.1f}s")
except Exception as exc:
    report("FAIL", "Reader load", repr(exc))
    print("\nCannot continue without a working Reader.")
    sys.exit(1)

print()
print("=" * 70)
print("6. Recognition-only path (the one your field crops will use)")
print("=" * 70)

# Draw a known digit string so we know exactly what the right answer is
canvas = numpy.full((110, 760), 255, dtype=numpy.uint8)
cv2.putText(canvas, "1234567890", (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
            2.4, 0, 5, cv2.LINE_AA)

t0 = time.perf_counter()
rec = reader.recognize(canvas, horizontal_list=None, free_list=None,
                       allowlist="0123456789", decoder="greedy", detail=1)
rec_ms = (time.perf_counter() - t0) * 1000
read_text = rec[0][1].replace(" ", "") if rec else ""
if read_text == "1234567890":
    report("PASS", "recognize() reads a synthetic digit strip",
           f"'{read_text}' in {rec_ms:.0f} ms")
else:
    report("WARN", "recognize() reads a synthetic digit strip",
           f"read '{read_text}' (expected 1234567890); the path works, but check the font rendering")

print()
print("=" * 70)
print("7. Full pipeline on a real NID photo")
print("=" * 70)

if not TEST_IMAGE.exists():
    report("FAIL", "Test image found", f"{TEST_IMAGE} does not exist; edit TEST_IMAGE at the top")
else:
    img = cv2.imread(str(TEST_IMAGE))
    if img is None:
        report("FAIL", "Test image readable", "cv2.imread returned None")
    else:
        reader.readtext(img)  # warm-up run, not timed
        t0 = time.perf_counter()
        boxes = reader.readtext(img, detail=1, paragraph=False)
        warm_ms = (time.perf_counter() - t0) * 1000

        bangla = [t for _, t, _ in boxes if BANGLA_CHARS.search(t)]
        digits = [t for _, t, _ in boxes if re.search(r"\d{4,}", t)]

        report("PASS" if boxes else "FAIL", "readtext() finds text",
               f"{len(boxes)} boxes, warm run {warm_ms:.0f} ms, image {img.shape[1]}x{img.shape[0]}")
        report("PASS" if bangla else "WARN", "Bangla text detected",
               f"{len(bangla)} boxes, e.g. {bangla[:2]}")
        report("PASS" if digits else "WARN", "Long digit run detected (NID/DOB)",
               f"e.g. {digits[:2]}")

print()
print("=" * 70)
if failures == 0:
    print("ALL REQUIRED CHECKS PASSED. Now freeze the environment:")
    print("    pip freeze > requirements-easyocr.txt")
else:
    print(f"{failures} CHECK(S) FAILED. Fix those lines and run again.")
print("=" * 70)
sys.exit(1 if failures else 0)