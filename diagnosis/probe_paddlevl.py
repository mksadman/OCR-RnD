"""
diagnosis/probe_paddlevl.py  --  run with .venv-paddlevl\Scripts\python.exe

PaddleOCR-VL is a document parser, not a detector + recogniser, so its output
shape is different from PP-OCR's. This probe answers three questions before any
worker is written:

  1. Does it load and run on this CPU at all?
  2. How long does ONE card take?  (this decides whether it is benchmarkable)
  3. What does the result object actually look like, and where is the text?

Usage:
    .venv-paddlevl\Scripts\python.exe diagnosis\probe_paddlevl.py
    .venv-paddlevl\Scripts\python.exe diagnosis\probe_paddlevl.py data\images\1a.jpg
"""

import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows console can't print Bangla otherwise
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "data" / "images"


def pick_image():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    for p in sorted(IMAGES.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
            return p
    raise SystemExit(f"no images found in {IMAGES}")


def describe(obj, label, depth=0):
    """Print what this object is and where text might live inside it."""
    pad = "  " * depth
    print(f"{pad}{label}: type={type(obj).__name__}")
    if depth > 2:
        return

    # Mapping-like?
    try:
        keys = list(obj.keys())
        print(f"{pad}  keys: {keys}")
        for k in keys:
            v = obj[k]
            note = f"len={len(v)}" if hasattr(v, "__len__") and not isinstance(v, str) else repr(v)[:80]
            print(f"{pad}    {k}: {type(v).__name__}  {note}")
        return
    except Exception:
        pass

    # Otherwise list the interesting public attributes.
    attrs = [a for a in dir(obj) if not a.startswith("_")]
    print(f"{pad}  attributes: {attrs}")
    for a in ("markdown", "json", "text", "res", "rec_texts"):
        if a in attrs:
            try:
                v = getattr(obj, a)
                print(f"{pad}  .{a} -> {type(v).__name__}: {repr(v)[:300]}")
            except Exception as e:
                print(f"{pad}  .{a} raised {type(e).__name__}: {e}")


def main():
    img = pick_image()
    print(f"image: {img}  ({img.stat().st_size/1024:.0f} KB)\n")

    import paddleocr
    print("paddleocr version:", getattr(paddleocr, "__version__", "unknown"))

    from paddleocr import PaddleOCRVL

    print("\nloading model (first run downloads weights, this can take a while) ...")
    t0 = time.perf_counter()
    pipeline = PaddleOCRVL()
    load_s = time.perf_counter() - t0
    print(f"loaded in {load_s:.1f}s")

    print("\nrunning inference on ONE card ...")
    t0 = time.perf_counter()
    output = pipeline.predict(str(img))
    infer_s = time.perf_counter() - t0
    print(f"*** inference took {infer_s:.1f}s for one card ***\n")

    results = list(output)
    print(f"predict() returned {type(output).__name__}, {len(results)} result(s)\n")

    for i, res in enumerate(results):
        print("=" * 70)
        describe(res, f"result[{i}]")
        print("=" * 70)

        # The most likely places the recognised text lives.
        for attr in ("markdown", "json"):
            try:
                val = getattr(res, attr, None)
                if val is None:
                    continue
                print(f"\n--- res.{attr} ---")
                if isinstance(val, dict):
                    print("dict keys:", list(val.keys()))
                    for k, v in val.items():
                        print(f"  {k}: {type(v).__name__} {repr(v)[:400]}")
                else:
                    print(repr(val)[:2000])
            except Exception as e:
                print(f"res.{attr} raised {type(e).__name__}: {e}")

        print("\n--- res.print() ---")
        try:
            res.print()
        except Exception as e:
            print(f"res.print() raised {type(e).__name__}: {e}")

    print("\n" + "=" * 70)
    print(f"SUMMARY: load {load_s:.1f}s, inference {infer_s:.1f}s per card")
    print("At this speed, 20 cards would take about "
          f"{infer_s * 20 / 60:.1f} minutes for a single accuracy pass.")
    print("=" * 70)


if __name__ == "__main__":
    main()