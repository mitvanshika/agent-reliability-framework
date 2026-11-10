"""
Memory of the dashboard in demo mode, after every page and API has been hit
once (what one Render worker holds). Also lists heavy modules if any got
imported. No API calls.
    python tests/measure_demo_memory.py            # DEMO_MODE=1 (default)
    python tests/measure_demo_memory.py --normal   # for comparison
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
if "--normal" not in sys.argv:
    os.environ["DEMO_MODE"] = "1"
else:
    # /api/approvals opens the action store: keep it out of the repo's real actions.db
    import tempfile
    os.environ["ACTIONS_DB"] = os.path.join(tempfile.mkdtemp(), "actions.db")

HEAVY = ("langchain", "langchain_core", "langgraph", "langchain_groq", "groq", "scipy", "numpy", "pydantic",
         "deepeval", "fastembed", "httpx")


def rss_mb():
    """(current, peak) resident memory in MB."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return pmc.WorkingSetSize / 2**20, pmc.PeakWorkingSetSize / 2**20
    import resource
    with open("/proc/self/status") as f:
        cur = next((int(ln.split()[1]) / 1024 for ln in f if ln.startswith("VmRSS:")), 0.0)
    return cur, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    base = rss_mb()[0]
    import app as appmod
    c = appmod.app.test_client()
    runs = [r["run_id"] for r in c.get("/api/experiments/runs").get_json()]
    paths = ["/", "/experiments", "/evidence", "/systems", "/approvals", "/api/data",
             "/api/experiments/baselines", "/api/approvals", "/api/experiments/ask/suggestions"]
    paths += [p for e in c.get("/evidence").get_data(as_text=True).split('href="/evidence/')[1:]
              for p in ["/evidence/" + e.split('"')[0]]]
    for r in runs:
        paths.append(f"/api/experiments/runs/{r}")
        for s in (c.get(f"/api/experiments/runs/{r}").get_json() or {}).get("scenarios", [])[:3]:
            paths += [f"/runs/{r}/{s['scenario_id']}", f"/api/experiments/runs/{r}/steps/{s['scenario_id']}"]
    if not appmod.DEMO_MODE:   # demo mode hides Track 1 (/track1 redirects to /)
        paths.append("/track1")
    bad = [(p, c.get(p).status_code) for p in paths if c.get(p).status_code != 200]
    for q in ("Why did exp_01 decline?", "anything else"):
        c.post("/api/experiments/ask", json={"question": q})
    cur, peak = rss_mb()
    heavy = sorted({m.split(".")[0] for m in sys.modules} & set(HEAVY))
    print(f"mode: {'DEMO' if appmod.DEMO_MODE else 'normal'}; {len(paths)} URLs hit, non-200: {bad or 'none'}")
    print(f"memory: python start {base:.1f} MB, after all pages {cur:.1f} MB (peak {peak:.1f} MB)")
    print(f"heavy modules loaded: {heavy or 'none'}")
    return heavy, cur, peak, bad


if __name__ == "__main__":
    main()
