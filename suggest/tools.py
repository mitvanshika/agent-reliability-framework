"""
The fix suggester's tools: READ-ONLY, inside the folders it's allowed to see.

    box = Sandbox([framework_repo, system_source_dir])
    box.list_files("adapters")             -> names
    box.read_file("store.py", 1, 80)       -> numbered lines
    box.search_code("search_faq")          -> file:line matches
    check_patch(box, patch_text)           -> (applies?, why): does it fit the current code? Never applied.

Nothing here writes, runs code or touches the network. Never readable, even inside an
allowed folder: .env files, key/certificate files, .git, virtual environments, caches,
runs/ and demo_data/ (run files reach the model only through the problem's evidence).
Anything that looks like an API key is replaced by [REDACTED] before the model sees it.
"""

import fnmatch
import os
import re

BLOCKED_DIRS = {".git", ".venv", "venv", "env", "__pycache__", "node_modules", ".mypy_cache", ".pytest_cache",
                "runs", "demo_data", ".cache", "site-packages", ".idea", ".vscode", "finetune"}
BLOCKED_FILES = [".env", ".env.*", "*.env", "*.pem", "*.key", "*.p12", "*.pfx", "id_rsa*", "*secret*", "*credential*",
                 "*.db", "*.sqlite", "*.pkl", "*.npy", "*.faiss", "*.bin", "*.zip", "*.pdf"]
SECRET = re.compile(r"(gsk_[A-Za-z0-9]{10,}|sk-[A-Za-z0-9_\-]{16,}|sk-ant-[A-Za-z0-9_\-]{10,}|AKIA[0-9A-Z]{16}"
                    r"|ghp_[A-Za-z0-9]{20,}|xox[bap]-[A-Za-z0-9\-]{10,})")
MAX_LINES = 80           # per read_file call (every call re-sends what was read: keep it small)
MAX_FILE_BYTES = 400_000
MAX_MATCHES = 40


def redact(text):
    return SECRET.sub("[REDACTED]", text)


class Refused(Exception):
    pass


class Sandbox:
    def __init__(self, roots):
        self.roots = []
        for r in roots:
            r = os.path.realpath(r)
            if os.path.isdir(r) and r not in self.roots:
                self.roots.append(r)
        if not self.roots:
            raise ValueError("no readable folder for the fix suggester")
        self.reads = []          # every file the model looked at (shown in the report)

    # ------------------------------------------------------------- paths
    def describe(self):
        return [{"name": f"root{i}" if i else "framework", "path": r} for i, r in enumerate(self.roots)]

    def resolve(self, path):
        """An allowed real path for `path` (relative to a root, or absolute inside one), else Refused."""
        path = str(path or ".").strip().replace("\\", "/")
        if os.path.isabs(path):
            candidates = [path]
        else:
            candidates = []
            for i, r in enumerate(self.roots):
                name = os.path.basename(r)
                if i and (path == name or path.startswith(name + "/")):   # shown as <folder name>/<path>
                    candidates.append(os.path.join(r, path[len(name) + 1:] or "."))
            candidates += [os.path.join(r, path) for r in self.roots]
        for c in candidates:
            real = os.path.realpath(c)
            root = next((r for r in self.roots if real == r or real.startswith(r + os.sep)), None)
            if root is None or not os.path.exists(real):
                continue
            rel = os.path.relpath(real, root)
            parts = [] if rel == "." else rel.split(os.sep)
            if any(p in BLOCKED_DIRS or p.endswith(".egg-info") for p in parts):
                raise Refused(f"{path}: not readable (folder is excluded)")
            if os.path.isfile(real) and _blocked_file(os.path.basename(real)):
                raise Refused(f"{path}: not readable (secrets or data files are never shown)")
            return real, root
        raise Refused(f"{path}: not found inside the allowed folders")

    def show(self, real, root):
        rel = os.path.relpath(real, root).replace(os.sep, "/")
        return rel if root == self.roots[0] else f"{os.path.basename(root)}/{rel}"

    # ------------------------------------------------------------- tools
    def list_files(self, path="."):
        real, root = self.resolve(path)
        if not os.path.isdir(real):
            raise Refused(f"{path}: not a folder")
        out = []
        for name in sorted(os.listdir(real)):
            full = os.path.join(real, name)
            if os.path.isdir(full):
                if name not in BLOCKED_DIRS and not name.endswith(".egg-info") and not name.startswith("."):
                    out.append(name + "/")
            elif not _blocked_file(name):
                out.append(name)
        return "\n".join(out[:300]) or "(empty)"

    def read_file(self, path, start=1, end=None):
        real, root = self.resolve(path)
        if not os.path.isfile(real):
            raise Refused(f"{path}: not a file")
        if os.path.getsize(real) > MAX_FILE_BYTES:
            raise Refused(f"{path}: too large to read")
        text = _text(real)
        if text is None:
            raise Refused(f"{path}: not a text file")
        lines = text.splitlines()
        start = max(1, int(start or 1))
        end = min(len(lines), int(end) if end else start + MAX_LINES - 1, start + MAX_LINES - 1)
        self.reads.append(f"{self.show(real, root)}:{start}-{end}")
        body = "\n".join(f"{i:5d}  {lines[i - 1]}" for i in range(start, end + 1))
        more = f"\n... ({len(lines)} lines in total; read more with start={end + 1})" if end < len(lines) else ""
        return redact(f"{self.show(real, root)} lines {start}-{end}:\n{body}{more}")

    def search_code(self, text, path="."):
        """Lines containing `text`. With no path: every allowed folder, the system's own folders first."""
        needle = str(text or "").strip()
        if len(needle) < 3:
            raise Refused("search for at least 3 characters")
        if str(path or ".").strip() in (".", "", "/"):
            hits = []
            for r in self.roots[1:] + self.roots[:1]:
                found = self._search(needle, r, r)
                if not found.startswith("no matches"):
                    hits.append(found)
            return "\n".join(hits) or f"no matches for {needle!r}"
        real, root = self.resolve(path)
        return self._search(needle, real, root)

    def _search(self, needle, real, root):
        hits = []
        walk = [(real, [], [os.path.basename(real)])] if os.path.isfile(real) else os.walk(real)
        for d, dirs, files in walk:
            if os.path.isfile(real):
                d = os.path.dirname(real)
            dirs[:] = [x for x in dirs if x not in BLOCKED_DIRS and not x.startswith(".") and not x.endswith(".egg-info")]
            for name in files:
                if _blocked_file(name) or not name.endswith((".py", ".yaml", ".yml", ".toml", ".md", ".txt", ".json", ".cfg", ".ini", ".j2", ".html")):
                    continue
                full = os.path.join(d, name)
                if os.path.getsize(full) > MAX_FILE_BYTES:
                    continue
                for i, line in enumerate((_text(full) or "").splitlines(), 1):
                    if needle.lower() in line.lower():
                        hits.append(f"{self.show(full, root)}:{i}: {line.strip()[:200]}")
                        if len(hits) >= MAX_MATCHES:
                            return redact("\n".join(hits) + "\n... (more matches: search a narrower folder)")
        return redact("\n".join(hits)) or f"no matches for {needle!r}"

    def file_lines(self, path):
        real, root = self.resolve(path)
        text = _text(real) if os.path.isfile(real) else None
        if text is None:
            raise Refused(f"{path}: not a readable text file")
        return text.splitlines()


def _blocked_file(name):
    low = name.lower()
    return any(fnmatch.fnmatch(low, p) for p in BLOCKED_FILES)


def _text(path):
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError:
        return None
    if b"\x00" in raw[:4096]:
        return None
    return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------- patch check (never applies)

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def check_patch(box, patch):
    """(applies: bool, why). Checks a unified diff against the current files WITHOUT changing them:
    every file must be readable in the sandbox and every hunk's old lines must be found in order."""
    if not patch or not patch.strip():
        return False, "no patch"
    files, cur = [], None
    for line in patch.splitlines():
        if line.startswith("--- "):
            cur = {"old": _strip_prefix(line[4:]), "hunks": []}
        elif line.startswith("+++ ") and cur is not None:
            cur["new"] = _strip_prefix(line[4:])
            files.append(cur)
        elif HUNK.match(line) and files:
            files[-1]["hunks"].append({"start": int(HUNK.match(line).group(1)), "old": [], "new": []})
        elif files and files[-1]["hunks"]:
            h = files[-1]["hunks"][-1]
            if line.startswith("-"):
                h["old"].append(line[1:])
            elif line.startswith("+"):
                h["new"].append(line[1:])
            elif line.startswith(" ") or line == "":
                h["old"].append(line[1:] if line else "")
                h["new"].append(line[1:] if line else "")
    if not files or not any(f["hunks"] for f in files):
        return False, "not a unified diff (needs ---/+++ lines and @@ hunks)"
    for f in files:
        path = f["new"] if f["old"] == "/dev/null" else f["old"]
        if f["old"] == "/dev/null":
            continue     # a new file: nothing to match
        try:
            lines = box.file_lines(path)
        except Refused as e:
            return False, str(e)
        pos = 0
        for h in f["hunks"]:
            at = _find(lines, h["old"], max(pos, h["start"] - 1 - 50))
            if at is None:
                at = _find(lines, h["old"], 0)
            if at is None:
                first = next((x for x in h["old"] if x.strip()), "")
                return False, f"{path}: the lines the patch changes aren't in the current file (near {first.strip()[:60]!r})"
            pos = at + len(h["old"])
    return True, f"fits the current code ({sum(len(f['hunks']) for f in files)} change(s) in {len(files)} file(s))"


def _strip_prefix(p):
    p = p.split("\t")[0].strip()
    return p[2:] if p.startswith(("a/", "b/")) else p


def _find(lines, block, start):
    if not block:
        return start
    norm = [b.rstrip() for b in block]
    for i in range(max(0, start), len(lines) - len(block) + 1):
        if all(lines[i + j].rstrip() == norm[j] for j in range(len(block))):
            return i
    return None
