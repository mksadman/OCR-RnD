import os
import sysconfig
import ctypes
import pefile

site = sysconfig.get_paths()["purelib"]
paddle_dir = os.path.join(site, "paddle")
libs_dir = os.path.join(paddle_dir, "libs")
base_dir = os.path.join(paddle_dir, "base")

for folder in (libs_dir, base_dir):
    if os.path.isdir(folder):
        os.add_dll_directory(folder)

phi_path = os.path.join(libs_dir, "phi.dll")
common_path = os.path.join(libs_dir, "common.dll")

# Test 1: load order. Paddle loads common.dll before phi.dll at import time,
# and phi.dll may simply need it resident first.
print("Test 1: loading common.dll then phi.dll by full path")
try:
    ctypes.WinDLL(common_path)
    print("   common.dll loaded")
except OSError as exc:
    print(f"   common.dll FAILED -> {exc}")

try:
    ctypes.WinDLL(phi_path)
    print("   phi.dll loaded - load order was the whole problem")
except OSError as exc:
    print(f"   phi.dll FAILED -> {exc}")
print()

# Test 2: read phi.dll's own import table and try each dependency.
pe = pefile.PE(phi_path, fast_load=True)
pe.parse_data_directories(
    directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]]
)

print("Test 2: direct dependencies of phi.dll")
missing = []
for entry in pe.DIRECTORY_ENTRY_IMPORT:
    dll_name = entry.dll.decode("utf-8")
    try:
        ctypes.WinDLL(dll_name)
        print(f"   OK      {dll_name}")
    except OSError:
        # api-ms-win-* are virtual API sets; failing to load them by name
        # is normal and not a real problem.
        tag = "(api-set, ignore)" if dll_name.lower().startswith("api-ms-win") else ""
        print(f"   FAILED  {dll_name} {tag}")
        if not tag:
            missing.append(dll_name)
print()

# Test 3: the full modern VC++ runtime set, individually.
print("Test 3: Visual C++ runtime components")
runtime_dlls = [
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll",
    "msvcp140_codecvt_ids.dll",
    "concrt140.dll",
    "vcomp140.dll",
]
for name in runtime_dlls:
    try:
        ctypes.WinDLL(name)
        print(f"   OK      {name}")
    except OSError:
        print(f"   MISSING {name}")
        missing.append(name)
print()

if missing:
    print("Real missing dependencies:", ", ".join(sorted(set(missing))))
else:
    print("Nothing missing at this level.")