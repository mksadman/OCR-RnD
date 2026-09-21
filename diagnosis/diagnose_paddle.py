import os
import sysconfig
import ctypes
import pefile

site = sysconfig.get_paths()["purelib"]
paddle_dir = os.path.join(site, "paddle")
pyd_path = os.path.join(paddle_dir, "base", "libpaddle.pyd")
libs_dir = os.path.join(paddle_dir, "libs")
base_dir = os.path.join(paddle_dir, "base")

print("paddle package:", paddle_dir)
print("libpaddle.pyd exists:", os.path.exists(pyd_path))
print()

# Show what DLLs actually shipped with the wheel.
for folder in (libs_dir, base_dir):
    print("Contents of", folder)
    if os.path.isdir(folder):
        dlls = sorted(f for f in os.listdir(folder) if f.lower().endswith((".dll", ".pyd")))
        for name in dlls:
            size = os.path.getsize(os.path.join(folder, name))
            print(f"   {name}  ({size:,} bytes)")
        if not dlls:
            print("   (no DLLs here - this is suspicious)")
    else:
        print("   (folder missing)")
    print()

# Python 3.8+ does not search PATH for extension DLLs, so register
# the folders explicitly the same way paddle does at import time.
for folder in (libs_dir, base_dir):
    if os.path.isdir(folder):
        os.add_dll_directory(folder)

# Read the import table of libpaddle.pyd and test each dependency.
pe = pefile.PE(pyd_path, fast_load=True)
pe.parse_data_directories(
    directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]]
)

print("Direct dependencies of libpaddle.pyd:")
missing = []
for entry in pe.DIRECTORY_ENTRY_IMPORT:
    dll_name = entry.dll.decode("utf-8")
    try:
        ctypes.WinDLL(dll_name)
        print(f"   OK      {dll_name}")
    except OSError as exc:
        print(f"   FAILED  {dll_name}   ->  {exc}")
        missing.append(dll_name)

print()
if missing:
    print("Could not load:", ", ".join(missing))
else:
    print("All direct dependencies loaded. The problem is one level deeper -")
    print("one of these DLLs is itself missing something.")