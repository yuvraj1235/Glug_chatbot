"""
Regenerates manifest.json by reading the REAL file tree of the pyq-dummy repo
via the GitHub API. Run this whenever new PYQs are added to the repo.

    python build_manifest.py

This is the fix for the root cause of the broken links: the old code built
URLs by guessing a path pattern (`sem{N}/{subject}/{year}.pdf`) that never
matched the repo's actual layout. Instead, we read the true tree once here
and store it, so the chatbot only ever serves paths that are confirmed to
exist -- it can't hallucinate a 404 anymore.
"""

import json
import re
import urllib.request
from collections import defaultdict

REPO = "Ankit-Gope007/pyq-dummy"
BRANCH = "main"
TREE_URL = f"https://api.github.com/repos/{REPO}/git/trees/{BRANCH}?recursive=1"

PATH_RE = re.compile(r"^pyqs/([^/]+)/sem(\d+)/([^/]+)/(midsem|endsem)/(\d{4})\.pdf$")


def fetch_tree() -> list[dict]:
    req = urllib.request.Request(TREE_URL, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req) as resp:
        data = json.load(resp)
    if data.get("truncated"):
        raise RuntimeError("GitHub tree response was truncated -- repo too large for this simple call")
    return data["tree"]


def build_manifest(tree: list[dict]) -> dict:
    manifest = defaultdict(lambda: {"branch": None, "sem": None, "exam_types": defaultdict(list)})

    for item in tree:
        if item["type"] != "blob":
            continue
        m = PATH_RE.match(item["path"])
        if not m:
            continue
        branch, sem, code, exam_type, year = m.groups()
        entry = manifest[code]
        entry["branch"] = branch
        entry["sem"] = int(sem)
        entry["exam_types"][exam_type].append(year)

    out = {}
    for code, entry in manifest.items():
        out[code] = {
            "branch": entry["branch"],
            "sem": entry["sem"],
            "exam_types": {et: sorted(years) for et, years in entry["exam_types"].items()},
        }
    return out


if __name__ == "__main__":
    tree = fetch_tree()
    manifest = build_manifest(tree)
    with open("manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"Wrote manifest.json with {len(manifest)} subject codes")