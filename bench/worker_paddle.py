"""
bench/worker_paddle.py   -- run with the PaddleOCR venv's python

IMPORTANT, and this belongs in the report: the fast PP-OCR models have no
Bangla recogniser. Bengali only exists in PaddleOCR-VL, a 0.9B vision-language
model in a completely different speed and cost class, which is not what you
would deploy behind an eKYC call.

So this worker runs lang="en" and declares the three Bangla name fields as
UNSUPPORTED. They are reported as "not supported", NOT as 0% accuracy: scoring
an engine zero on a language it never claimed to read would be misleading.
PaddleOCR still competes where it matters most, on NID number and date of
birth, which are digits.

Environment variables (all optional):
  BENCH_GPU          0 (default) | 1 | auto
  BENCH_PADDLE_LANG  default "en"
  BENCH_PADDLE_DET   e.g. PP-OCRv6_small_det   (see the tier note below)
  BENCH_PADDLE_REC   e.g. PP-OCRv6_small_rec
"""

import os

for _v in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "FLAGS_use_mkldnn"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

LANG = os.environ.get("BENCH_PADDLE_LANG", "en")
GPU_SETTING = os.environ.get("BENCH_GPU", "0").lower()

# PaddleOCR picks PP-OCRv6_medium by default, which is its LARGEST and slowest
# tier (det 59.4 MB + rec 73.3 MB). For a speed/cost benchmark the tier has to
# be a deliberate choice, so it is set here. Available v6 tiers:
#   tiny    det 1.9 MB  / rec 4.4 MB    fastest, lowest accuracy
#   small   det 9.6 MB  / rec 20.4 MB   usually the best speed/accuracy trade
#   medium  det 59.4 MB / rec 73.3 MB   most accurate, the library default
# Leave these empty to accept whatever PaddleOCR chooses.
DET_MODEL = os.environ.get("BENCH_PADDLE_DET", "").strip()
REC_MODEL = os.environ.get("BENCH_PADDLE_REC", "").strip()

# No Bangla recogniser in the PP-OCR pipeline.
UNSUPPORTED = ("name_bn", "father_bn", "mother_bn")

_state = {"version": "unknown", "api": "unknown", "gpu": False,
          "det": "library default", "rec": "library default"}


def _want_gpu():
    if GPU_SETTING in ("1", "true", "yes"):
        return True
    if GPU_SETTING == "auto":
        try:
            import paddle
            return bool(paddle.device.is_compiled_with_cuda()
                        and paddle.device.cuda.device_count() > 0)
        except Exception:
            return False
    return False


def load():
    import paddleocr
    from paddleocr import PaddleOCR

    _state["version"] = getattr(paddleocr, "__version__", "unknown")
    gpu = _want_gpu()
    _state["gpu"] = gpu

    try:
        import paddle
        paddle.set_device("gpu" if gpu else "cpu")
    except Exception:
        pass

    threads = int(os.environ.get("OMP_NUM_THREADS", "1"))

    # Explicit model tier, 3.x only.
    models = {}
    if DET_MODEL:
        models["text_detection_model_name"] = DET_MODEL
    if REC_MODEL:
        models["text_recognition_model_name"] = REC_MODEL

    # These three sub-steps are off because a cropped NID photo does not need
    # document unwarping or page rotation, and each one costs time per request.
    v3 = dict(lang=LANG,
              use_doc_orientation_classify=False,
              use_doc_unwarping=False,
              use_textline_orientation=False)

    # PaddleOCR changed its constructor between 2.x and 3.x, and again within
    # 3.x. Try the richest call first and drop arguments until one is accepted.
    attempts = [
        dict(v3, **models, cpu_threads=threads),
        dict(v3, **models),
        dict(v3, cpu_threads=threads),
        v3,
        dict(lang=LANG, use_angle_cls=False, show_log=False, cpu_threads=threads),
        dict(lang=LANG, use_angle_cls=False),
        dict(lang=LANG),
    ]
    last = None
    for kwargs in attempts:
        try:
            model = PaddleOCR(**kwargs)
            _state["api"] = "3x" if "use_textline_orientation" in kwargs else "2x"
            _state["det"] = kwargs.get("text_detection_model_name", "library default")
            _state["rec"] = kwargs.get("text_recognition_model_name", "library default")
            return model
        except TypeError as e:
            last = e
    raise RuntimeError(f"could not construct PaddleOCR: {last}")


def _as_dict(page):
    """PaddleOCR 3.x returns result objects that behave like dicts; 2.x returns lists."""
    if isinstance(page, dict):
        return page
    try:
        if "rec_texts" in page:
            return page
    except Exception:
        pass
    j = getattr(page, "json", None)
    if isinstance(j, dict):
        return j.get("res", j)
    return None


def _pairs(raw):
    """Normalise every known output shape into [(box, text), ...]."""
    pairs = []
    if not raw:
        return pairs

    for page in raw:
        if page is None:
            continue

        d = _as_dict(page)
        if d is not None:                                  # 3.x
            texts = d["rec_texts"] if "rec_texts" in d else []
            boxes = None
            for key in ("rec_polys", "dt_polys", "rec_boxes"):
                if key in d and d[key] is not None and len(d[key]):
                    boxes = d[key]
                    break
            for i, t in enumerate(texts):
                if boxes is not None and i < len(boxes):
                    box = boxes[i]
                    # rec_boxes may be [x1,y1,x2,y2] instead of 4 points.
                    if len(box) == 4 and not hasattr(box[0], "__len__"):
                        x1, y1, x2, y2 = [float(v) for v in box]
                        box = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                else:
                    box = [[0, i * 10], [1, i * 10], [1, i * 10 + 1], [0, i * 10 + 1]]
                pairs.append((box, t))
            continue

        if isinstance(page, list):                          # 2.x
            for line in page:
                if line and len(line) >= 2:
                    box, rec = line[0], line[1]
                    text = rec[0] if isinstance(rec, (list, tuple)) else rec
                    pairs.append((box, text))

    return pairs


def ocr(model, rgb):
    import cv2

    # PaddleOCR expects BGR arrays, same reasoning as the EasyOCR worker.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if hasattr(model, "predict"):
        raw = model.predict(bgr)
    else:
        raw = model.ocr(bgr, cls=False)

    return common.boxes_to_lines(_pairs(raw))


def version():
    return (f"paddleocr {_state['version']} ({_state['api']} API, lang={LANG}, "
            f"gpu={_state['gpu']}, det={_state['det']}, rec={_state['rec']})")


if __name__ == "__main__":
    common.run_worker("paddleocr", load, ocr, version, unsupported=UNSUPPORTED)