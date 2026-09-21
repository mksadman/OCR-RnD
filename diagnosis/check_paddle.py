import os
import sys
import time
import glob
import ctypes
import struct
import statistics
import warnings

# Hide the harmless ccache warning Paddle prints on import
warnings.filterwarnings("ignore", message=".*ccache.*")

results = []


def record(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    results.append((name, status, detail))
    print(f"[{status}] {name}" + (f"  ->  {detail}" if detail else ""))


def warn(name, detail):
    results.append((name, "WARN", detail))
    print(f"[WARN] {name}  ->  {detail}")


print("=" * 70)
print("PaddleOCR setup check")
print("=" * 70)

# 1. Python version, 64-bit, and venv
py_ok = sys.version_info[:2] == (3, 12)
record("Python 3.12", py_ok, sys.version.split()[0])

bits = struct.calcsize("P") * 8
record("64-bit Python", bits == 64, f"{bits}-bit")

in_venv = sys.prefix != sys.base_prefix
record("Running inside a venv", in_venv, sys.prefix)
if in_venv and ".venv-paddle" not in sys.prefix:
    warn("Venv name", "Expected .venv-paddle - is the right venv activated?")

# 2. Project folder location (OneDrive can break venvs)
cwd = os.getcwd()
if "onedrive" in cwd.lower():
    warn("Project location", "Folder is inside OneDrive - consider moving to C:\\dev")
else:
    record("Project not inside OneDrive", True, cwd)

# 3. Visual C++ runtime DLLs, including the OpenMP one that was missing
runtime_dlls = [
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "msvcp140.dll",
    "vcomp140.dll",
]
missing = []
for dll in runtime_dlls:
    try:
        ctypes.WinDLL(dll)
    except OSError:
        missing.append(dll)
record("Visual C++ runtime (incl. vcomp140)", not missing,
       "all present" if not missing else "missing: " + ", ".join(missing))

# 4. PaddlePaddle framework
paddle_ok = False
try:
    import paddle
    paddle_ok = True
    record("import paddle", True, paddle.__version__)
    record("Paddle is CPU build", not paddle.is_compiled_with_cuda(),
           "CPU" if not paddle.is_compiled_with_cuda() else "CUDA build found")
    # Paddle's own self-test: builds and runs a tiny network
    paddle.utils.run_check()
    record("paddle.utils.run_check()", True)
except Exception as exc:
    record("import paddle", False, repr(exc))

# 5. PaddleOCR package
ocr_pkg_ok = False
try:
    import paddleocr
    ocr_pkg_ok = True
    record("import paddleocr", True, paddleocr.__version__)
except Exception as exc:
    record("import paddleocr", False, repr(exc))

# 6. Model download source
source = os.environ.get("PADDLE_PDX_MODEL_SOURCE")
if source:
    record("Model source env var", True, source)
else:
    warn("Model source env var", "Not set - will use HuggingFace (set to BOS if downloads are slow)")

# 7. Find a test image
image_files = []
for ext in ("jpg", "jpeg", "png"):
    image_files += glob.glob(os.path.join("images", f"*.{ext}"))
image_files.sort()
record("Test images found in ./images", bool(image_files),
       f"{len(image_files)} images" if image_files else "no images found")

# 8. Build the pipeline and run real OCR
if paddle_ok and ocr_pkg_ok and image_files:
    try:
        from paddleocr import PaddleOCR

        t0 = time.perf_counter()
        ocr = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
        )
        load_s = time.perf_counter() - t0
        record("Pipeline created (models loaded)", True, f"{load_s:.1f} s")

        test_image = image_files[0]

        # First run includes one-time warm-up, so time it separately
        t0 = time.perf_counter()
        out = ocr.predict(test_image)
        first_ms = (time.perf_counter() - t0) * 1000

        payload = out[0].json
        payload = payload["res"] if "res" in payload else payload
        texts = payload.get("rec_texts", [])
        scores = payload.get("rec_scores", [])

        record("OCR ran on a real image", len(texts) > 0,
               f"{os.path.basename(test_image)}: {len(texts)} text lines found")

        # Warm runs give the realistic per-image time
        warm = []
        for _ in range(3):
            t0 = time.perf_counter()
            ocr.predict(test_image)
            warm.append((time.perf_counter() - t0) * 1000)
        median_ms = statistics.median(warm)
        record("Timing measured", True,
               f"first run {first_ms:.0f} ms, warm median {median_ms:.0f} ms")

        print("\nSample output (first 10 lines):")
        for text, score in list(zip(texts, scores))[:10]:
            print(f"   {score:.3f}  {text}")

        os.makedirs("paddle_out", exist_ok=True)
        out[0].save_to_img(save_path="./paddle_out/")
        out[0].save_to_json(save_path="./paddle_out/")
        record("Annotated image saved", True, "./paddle_out/")
    except Exception as exc:
        record("OCR pipeline", False, repr(exc))
else:
    warn("OCR pipeline", "Skipped because an earlier check failed")

# 9. Model cache location
cache_dir = os.path.join(os.path.expanduser("~"), ".paddlex", "official_models")
if os.path.isdir(cache_dir):
    models = sorted(os.listdir(cache_dir))
    record("Model cache exists", True, ", ".join(models))
else:
    warn("Model cache", f"Not found at {cache_dir}")

# Summary
print("\n" + "=" * 70)
fails = [r for r in results if r[1] == "FAIL"]
warns = [r for r in results if r[1] == "WARN"]
print(f"Summary: {len(results) - len(fails) - len(warns)} passed, "
      f"{len(warns)} warnings, {len(fails)} failed")
if fails:
    print("Fix these first:")
    for name, _, detail in fails:
        print(f"   - {name}: {detail}")
else:
    print("PaddleOCR environment is ready.")
print("=" * 70)