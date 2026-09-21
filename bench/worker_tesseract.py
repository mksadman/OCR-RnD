"""
bench/worker_tesseract.py   -- run with the Tesseract venv's python

Reads the WHOLE card in one pass with lang="ben+eng", then common.py parses
the fields. Nothing here is engine-specific except the OCR call itself.

Backend note (important for the speed numbers):
  tesserocr  calls Tesseract's C++ API inside this process.
  pytesseract writes a temp file and launches tesseract.exe for EVERY request,
             which adds process-start overhead that a real server would not pay.
  So tesserocr is used when it is installed, and the backend name is recorded
  in the results meta. Force one with BENCH_TESS_BACKEND=tesserocr|pytesseract.

Environment variables (all optional):
  TESSDATA_PREFIX       folder holding ben.traineddata / eng.traineddata
  TESSERACT_EXE         path to tesseract.exe   (pytesseract backend only)
  BENCH_TESS_LANG       default "ben+eng"
  BENCH_TESS_PSM        page segmentation mode, default 3 (automatic)
  BENCH_TESS_OEM        OCR engine mode, default 1 (LSTM only)
  BENCH_TESS_BACKEND    auto | tesserocr | pytesseract
"""

import os

# Thread limits must be set BEFORE the OCR libraries are imported, otherwise
# they have already decided how many threads to use. The notebook sets these
# per run; "1" is the safe default so one worker means one core.
for _v in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

TESSDATA = os.environ.get("TESSDATA_PREFIX", r"C:\tessdata_best")
LANG = os.environ.get("BENCH_TESS_LANG", "ben+eng")
PSM = int(os.environ.get("BENCH_TESS_PSM", "3"))
OEM = int(os.environ.get("BENCH_TESS_OEM", "1"))
BACKEND = os.environ.get("BENCH_TESS_BACKEND", "auto").lower()

_state = {"backend": None, "version": "unknown"}


def load():
    """Create the recogniser once. Returns a callable: rgb array -> text."""
    want = BACKEND

    if want in ("auto", "tesserocr"):
        try:
            import tesserocr
            from PIL import Image

            # psm/oem MUST be plain ints. tesserocr.PSM and tesserocr.OEM are
            # Cython enum classes, not Python enums, so tesserocr.PSM(3) raises
            # "__init__() takes exactly 0 positional arguments (1 given)".
            path = TESSDATA.rstrip("\\/") + os.sep   # trailing separator required
            try:
                api = tesserocr.PyTessBaseAPI(path=path, lang=LANG, psm=PSM, oem=OEM)
            except TypeError:
                # Older builds do not accept oem in the constructor.
                api = tesserocr.PyTessBaseAPI(path=path, lang=LANG, psm=PSM)

            _state["backend"] = "tesserocr"
            _state["version"] = str(tesserocr.tesseract_version()).split()[1]

            def _run(rgb):
                # PIL.Image.fromarray treats a 3-channel array as RGB, which is
                # exactly what common.decode_image returns. No conversion needed.
                api.SetImage(Image.fromarray(rgb))
                return api.GetUTF8Text()

            return _run
        except Exception as e:
            if want == "tesserocr":
                raise
            print(f"[tesseract] tesserocr unavailable ({type(e).__name__}: {e}); "
                  f"falling back to pytesseract", file=sys.stderr)

    import pytesseract

    exe = os.environ.get("TESSERACT_EXE")
    if exe:
        pytesseract.pytesseract.tesseract_cmd = exe
    os.environ["TESSDATA_PREFIX"] = TESSDATA

    _state["backend"] = "pytesseract"
    _state["version"] = str(pytesseract.get_tesseract_version())

    langs = pytesseract.get_languages(config="")
    missing = [l for l in LANG.split("+") if l not in langs]
    if missing:
        raise RuntimeError(
            f"missing traineddata for {missing} in {TESSDATA}. "
            f"Tesseract sees: {sorted(langs)}")

    cfg = f"--psm {PSM} --oem {OEM}"

    def _run(rgb):
        # pytesseract converts the array via PIL, so RGB is correct here too.
        return pytesseract.image_to_string(rgb, lang=LANG, config=cfg)

    return _run


def ocr(model, rgb):
    return model(rgb)


def version():
    return f"tesseract {_state['version']} via {_state['backend']} (lang={LANG}, psm={PSM}, oem={OEM})"


if __name__ == "__main__":
    common.run_worker("tesseract", load, ocr, version)