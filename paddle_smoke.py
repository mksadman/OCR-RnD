from paddleocr import PaddleOCR

# lang="en" selects an English + digits recognition model, which is what
# matters for the NID number and date of birth fields.
# The three "use_" flags are optional preprocessing stages, off here for a
# clean fast baseline:
#   use_doc_orientation_classify - detects whole-page 90/180/270 rotation
#   use_doc_unwarping            - flattens curved or warped pages
#   use_textline_orientation     - detects upside-down individual lines
ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="cpu",
)

IMAGE_PATH = r"images/nid_sample_01.jpg"   # change to one of your real cards

results = ocr.predict(IMAGE_PATH)

for res in results:
    res.print()

    payload = res.json["res"] if "res" in res.json else res.json
    print("\nAvailable keys:", list(payload.keys()))

    res.save_to_img(save_path="./paddle_out/")
    res.save_to_json(save_path="./paddle_out/")

print("\nDone. Check ./paddle_out/ for the annotated image.")