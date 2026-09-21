"""Print the download URLs, filenames and checksums EasyOCR will use."""

import easyocr.config as cfg

def walk(name, obj, depth=0):
    if isinstance(obj, dict):
        if 'url' in obj and 'filename' in obj:
            print(f"{'  ' * depth}{name}")
            print(f"{'  ' * depth}  filename: {obj.get('filename')}")
            print(f"{'  ' * depth}  md5sum  : {obj.get('md5sum')}")
            print(f"{'  ' * depth}  url     : {obj.get('url')}\n")
        else:
            for key, value in obj.items():
                walk(key, value, depth + 1)

for attr in ('detection_models', 'recognition_models'):
    if hasattr(cfg, attr):
        print(f"=== {attr} ===")
        walk(attr, getattr(cfg, attr))