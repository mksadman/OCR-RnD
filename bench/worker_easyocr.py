"""
bench/worker_easyocr.py   -- run with the EasyOCR venv's python

EasyOCR's Bengali model group already contains English, so Reader(["bn","en"])
is one model that reads both scripts on the card. It is a detector plus a
recogniser, so it returns many boxes; common.boxes_to_lines puts them back into
reading order before the shared parser runs.

Environment variables (all optional):
  BENCH_GPU        0 (default) | 1 | auto.  Default CPU so the cost model and
                   the other engines stay comparable; run GPU as a separate,
                   clearly labelled run.
  BENCH_EASY_LANGS default "bn,en"
  EASYOCR_MODULE_PATH  where EasyOCR keeps its downloaded weights
"""

import os

# Must happen before torch is imported (see worker_tesseract.py for why).
for _v in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

LANGS = [s.strip() for s in os.environ.get("BENCH_EASY_LANGS", "bn,en").split(",") if s.strip()]
GPU_SETTING = os.environ.get("BENCH_GPU", "0").lower()

_state = {"version": "unknown", "gpu": False}


def _want_gpu():
    if GPU_SETTING in ("1", "true", "yes"):
        return True
    if GPU_SETTING == "auto":
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False
    return False


def load():
    import easyocr
    import torch

    # Keep one worker on one core so "workers" and "cores" mean the same thing
    # in the throughput test.
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

    gpu = _want_gpu()
    _state["gpu"] = gpu
    _state["version"] = getattr(easyocr, "__version__", "unknown")

    return easyocr.Reader(LANGS, gpu=gpu, verbose=False)


def ocr(model, rgb):
    import cv2

    # EasyOCR treats a 3-channel numpy array as BGR (it runs cv2.COLOR_BGR2GRAY
    # on it internally). Handing it RGB would silently change the greyscale
    # weighting and therefore the contrast the detector sees, so convert.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # paragraph=False keeps one entry per detected line; grouping is done by
    # common.boxes_to_lines so every engine is grouped the same way.
    out = model.readtext(bgr, detail=1, paragraph=False)

    pairs = []
    for item in out:
        # (box, text, confidence), but be tolerant of shorter tuples.
        if len(item) >= 2:
            pairs.append((item[0], item[1]))
    return common.boxes_to_lines(pairs)


def version():
    return f"easyocr {_state['version']} (langs={'+'.join(LANGS)}, gpu={_state['gpu']})"


if __name__ == "__main__":
    common.run_worker("easyocr", load, ocr, version)