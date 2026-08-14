import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MANIFEST_PATH = Path(__file__).parent / "manifest.json"


def _load_manifest() -> dict:
    """
    Load manifest.json from disk. If it's missing (e.g. forgot to commit it,
    fresh checkout), fall back to building it live from the GitHub API so the
    app doesn't crash on startup -- and write it to disk so next boot is fast.
    """
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)

    logger.warning("manifest.json not found at %s -- building it live from GitHub", MANIFEST_PATH)
    try:
        from build_manifest import fetch_tree, build_manifest  # project-root script
        manifest = build_manifest(fetch_tree())
        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        logger.info("Built manifest.json live (%d subject codes)", len(manifest))
        return manifest
    except Exception:
        logger.exception(
            "Could not build manifest.json live either (network/rate-limit?). "
            "PYQ lookups will be disabled until manifest.json is added to %s",
            MANIFEST_PATH.parent,
        )
        return {}


MANIFEST: dict = _load_manifest()

CDN_BASE = "https://cdn.jsdelivr.net/gh/Ankit-Gope007/pyq-dummy@main/pyqs"

# Confirmed sem1/sem2 codes -> full name aliases (semesters 1-2 are shared
# across all branches, so they live under pyqs/common/, not a branch folder)
SUBJECT_MAP = {
    # cs01 -> Data Structures and Algorithms 1
    "data structures and algorithms": "cs01",
    "data structures and algorithms 1": "cs01",
    "data structures": "cs01",
    "dsa": "cs01",
    "dsa1": "cs01",
    "dsa 1": "cs01",
    "ds": "cs01",

    # ma01 -> Engineering Mathematics 1 / ma02 -> presumably Maths 2
    "engineering mathematics": "ma01",
    "engineering mathematics 1": "ma01",
    "maths 1": "ma01",
    "math 1": "ma01",
    "maths": "ma01",
    "math": "ma01",
    "mathematics": "ma01",

    # cy01 -> Chemistry
    "chemistry": "cy01",
    "chem": "cy01",
    "cy": "cy01",

    # ph01 -> Physics
    "physics": "ph01",
    "phy": "ph01",
    "ph": "ph01",

    # es01 -> Ecology
    "ecology": "es01",
    "es": "es01",

    # hs01 -> English
    "english": "hs01",
    "hs": "hs01",

    # xe01 -> Basic Electrical and Electronics Engineering
    "basic electrical and electronics engineering": "xe01",
    "basic electrical and electronics": "xe01",
    "basic electrical": "xe01",
    "electrical engineering": "xe01",
    "electrical": "xe01",
    "ee": "xe01",
    "xe": "xe01",
    "xeco1": "xe01",
    "xeco": "xe01",
}

# Direct code passthrough: if someone just says "cs01" or "cec301", match as-is
CODE_RE = re.compile(r"\b([a-z]{2,4}\d{2,3})\b")

# Branch aliases -> the folder name used in the repo. Sem1/2 always live
# under "common" regardless of branch, so those aren't listed here.
BRANCH_MAP = {
    "cse": "cse", "computer science": "cse", "cs": "cse",
    "cve": "cve", "civil": "cve", "civil engineering": "cve",
    # add more as new branch folders appear in the repo, e.g.:
    # "ece": "ece", "electronics": "ece",
    # "me": "me", "mechanical": "me",
}

PYQ_TRIGGERS = [
    r"\bpyqs?\b",
    r"\bprevious\s+year\s+questions?\b",
    r"\bprevious\s+year\s+papers?\b",
    r"\bquestion\s+papers?\b",
    r"\bexam\s+papers?\b",
    r"\bpast\s+papers?\b",
    r"\bquestion\s+banks?\b",
    r"\bpast\s+exams?\b",
]

EXAM_TYPE_PATTERNS = {
    "midsem": r"\bmid\s?-?sem(?:ester)?\b|\bmid\s?-?term\b",
    "endsem": r"\bend\s?-?sem(?:ester)?\b|\bfinal\b",
}

ALL_RE = re.compile(r"\ball\b")


def _extract_subject_code(text: str) -> Optional[str]:
    # 1. Try known aliases (longest phrase first, so "dsa 1" beats "dsa")
    for key in sorted(SUBJECT_MAP.keys(), key=len, reverse=True):
        if re.search(r"\b" + re.escape(key) + r"\b", text):
            code = SUBJECT_MAP[key]
            if code in MANIFEST:
                return code

    # 2. Try a literal subject code the user typed directly (e.g. "cs301")
    for m in CODE_RE.finditer(text):
        if m.group(1) in MANIFEST:
            return m.group(1)

    return None


def _extract_branch(text: str) -> Optional[str]:
    for key in sorted(BRANCH_MAP.keys(), key=len, reverse=True):
        if re.search(r"\b" + re.escape(key) + r"\b", text):
            return BRANCH_MAP[key]
    return None


def _extract_semester(text: str) -> Optional[int]:
    patterns = [
        r"\bsem(?:ester)?\s*[-_]?\s*([1-8])\b",
        r"\b([1-8])(?:st|nd|rd|th)?\s*sem(?:ester)?\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return int(m.group(1))
    return None


def _codes_for(branch: Optional[str], sem: Optional[int]) -> list[str]:
    return sorted(
        code for code, entry in MANIFEST.items()
        if (branch is None or entry["branch"] == branch)
        and (sem is None or entry["sem"] == sem)
    )


def _extract_exam_type(text: str) -> Optional[str]:
    for exam_type, pattern in EXAM_TYPE_PATTERNS.items():
        if re.search(pattern, text):
            return exam_type
    return None  # unspecified -> caller shows both


def _extract_years(text: str, available: list[str]) -> list[str]:
    requested = re.findall(r"\b(20\d{2})\b", text)
    if requested:
        # Only keep years that actually exist for this subject/exam-type
        valid = [y for y in dict.fromkeys(requested) if y in available]
        return sorted(valid, reverse=True) if valid else sorted(available, reverse=True)[:2]
    if ALL_RE.search(text):
        return sorted(available, reverse=True)  # every year on file
    return sorted(available, reverse=True)[:2]  # default: 2 most recent


def _build_files(code: str, text: str, latest_only: bool = False) -> list[dict]:
    """
    All file entries for one subject code, respecting exam-type/year filters
    in text. When latest_only=True, ignores year filters entirely and returns
    just the single most recent year per exam type -- used for "all subjects"
    requests so we don't dump every subject's full history in one response.
    """
    entry = MANIFEST[code]
    branch, sem = entry["branch"], entry["sem"]
    exam_type = _extract_exam_type(text)
    exam_types_to_show = [exam_type] if exam_type else list(entry["exam_types"].keys())

    files = []
    for et in exam_types_to_show:
        available_years = entry["exam_types"].get(et)
        if not available_years:
            continue
        years = [sorted(available_years, reverse=True)[0]] if latest_only else _extract_years(text, available_years)
        for year in years:
            files.append({
                "year": year,
                "exam_type": et,
                "label": f"{year} {et.capitalize()}",
                "url": f"{CDN_BASE}/{branch}/sem{sem}/{code}/{et}/{year}.pdf",
            })
    return files


async def get_pyq_response(message: str) -> Optional[dict]:
    """
    Matches a PYQ request against the real manifest.
    Returns None if the message isn't a PYQ request at all.
    Otherwise returns a dict shaped as one of:

    {
        "type": "results",
        "semester": 1,           # only set when every subject shares one semester
        "message": "Semester 1 XE01 PYQs",
        "subjects": [
            {"code": "XE01", "files": [
                {"year": "2025", "exam_type": "endsem", "label": "2025 Endsem",
                 "url": "https://.../2025.pdf"},
                ...
            ]},
            ...   # more than one entry when "all subjects" was requested
        ]
    }

    {
        "type": "clarify",
        "message": "...",
        "options": ["CS301", "CS302", ...]   # may be empty
    }

    {
        "type": "empty",
        "code": "XE01",
        "message": "..."
    }
    """
    text = message.lower().strip()

    if not any(re.search(p, text) for p in PYQ_TRIGGERS):
        return None

    code = _extract_subject_code(text)
    branch = _extract_branch(text)
    sem = _extract_semester(text)

    # "give me all sem 3 cse pyqs" -> no single subject named, but "all" +
    # a known sem/branch means: return every subject in that sem/branch.
    if code is None and ALL_RE.search(text) and sem is not None:
        candidates = _codes_for(branch, sem)
        if not candidates and branch is not None:
            candidates = _codes_for("common", sem)  # fall back to common core
        subjects = []
        for c in candidates:
            files = _build_files(c, text, latest_only=True)  # newest paper per subject, not full history
            if files:
                subjects.append({"code": c.upper(), "files": files})
        if subjects:
            where = f"{branch or 'common'}, sem {sem}" if branch else f"sem {sem}"
            return {
                "type": "results",
                "semester": sem,
                "message": f"Latest PYQs for {where}",
                "subjects": subjects,
            }
        # matched "all" + sem but nothing on file at all -> fall through
        # to the normal clarify/empty handling below

    if code is None:
        # If we know branch+sem (or just sem, for common subjects), hand back
        # the real codes available instead of sending the user off to GitHub.
        if sem is not None:
            candidates = _codes_for(branch, sem)
            if not candidates and branch is not None:
                candidates = _codes_for("common", sem)
            if candidates:
                where = f"{branch or 'common'}, sem {sem}" if branch else f"sem {sem}"
                return {
                    "type": "clarify",
                    "message": f"I know the subjects for {where}, but not which one you mean. "
                               f"Reply with a code (or say \"all\") and I'll pull the PYQs.",
                    "options": [c.upper() for c in candidates],
                }

        return {
            "type": "clarify",
            "message": "I couldn't tell which subject you mean. Could you give the subject "
                        "code (e.g. cs01, cs301, cec301) along with the semester? You can "
                        "check the code in your timetable/registration portal, or tell me "
                        "the branch + semester and I'll list the codes for you.",
            "options": [],
        }

    entry = MANIFEST[code]
    sem = entry["sem"]
    files = _build_files(code, text)

    if not files:
        return {
            "type": "empty",
            "code": code.upper(),
            "message": f"I found the subject ({code.upper()}) but there are no papers on file for it yet.",
        }

    return {
        "type": "results",
        "semester": sem,
        "message": f"Semester {sem} {code.upper()} PYQs",
        "subjects": [{"code": code.upper(), "files": files}],
    }