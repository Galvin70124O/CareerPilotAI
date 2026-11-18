"""Fresh process: time to `import app` (Flask + model.pkl + JSON data), working set after it and after one prediction.
    python startup.py <code dir>"""
import ctypes
import json
import os
import sys
import time
from ctypes import wintypes


class Counters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def working_set_mb():
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    ctypes.windll.psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    return counters.WorkingSetSize / 2 ** 20, counters.PeakWorkingSetSize / 2 ** 20


code = sys.argv[1]
os.chdir(code)
sys.path.insert(0, code)
start = time.perf_counter()
import app  # noqa: E402
load_s = time.perf_counter() - start
after_import, _ = working_set_mb()
with open(os.path.join(code, "data", "sample_resume.txt"), encoding="utf-8") as f:
    text = f.read()
top = app.predict_top_careers(text)
try:
    from features.ranking import fresher_roles
    fresher_roles(text, top)
except ImportError:
    pass
after_predict, peak = working_set_mb()
print(json.dumps({"load_s": load_s, "mb_after_import": after_import, "mb_after_predict": after_predict, "mb_peak": peak}))
