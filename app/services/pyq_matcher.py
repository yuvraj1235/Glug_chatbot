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
    "data structures and algorithms": "cs01",
    "data structures and algorithms 1": "cs01",
    "data structures": "cs01",
    "dsa": "cs01",
    "dsa1": "cs01",
    "dsa 1": "cs01",
    "ds": "cs01",
    "engineering mathematics": "ma01",
    "engineering mathematics 1": "ma01",
    "maths 1": "ma01",
    "math 1": "ma01",
    "maths": "ma01",
    "math": "ma01",
    "mathematics": "ma01",
    "chemistry": "cy01",
    "chem": "cy01",
    "cy": "cy01",
    "physics": "ph01",
    "phy": "ph01",
    "ph": "ph01",
    "ecology": "es01",
    "es": "es01",
    "english": "hs01",
    "hs": "hs01",
    "basic electrical and electronics engineering": "xe01",
    "basic electrical and electronics": "xe01",
    "basic electrical": "xe01",
    "electrical engineering": "xe01",
    "electrical": "xe01",
    "ee": "xe01",
    "xe": "xe01",
    "xeco1": "xe01",
    "xeco": "xe01",
    "mathematics - i": "mac01",
    "mathematics 1": "mac01",
    "computer programming": "csc01",
    "engineering mechanics": "xec01",
    "engineering physics": "phc01",
    "professional communication": "hsc01",
    "computer programming laboratory": "css51",
    "engineering graphics": "xes51",
    "engineering physics laboratory": "phs51",
    "extra academic activities": "xxs51",
    "mathematics - ii": "mac02",
    "mathematics 2": "mac02",
    "data structure and algorithms": "csc02",
    "ecology and environment": "esc01",
    "engineering chemistry": "cyc01",
    "engineering chemistry laboratory": "cys51",
    "data structure and algorithms laboratory": "css52",
    "basic electrical and electronics engineering laboratory": "xes52",
    "mathematics iii": "mac331",
    "mathematics 3": "mac331",
    "biochemistry & enzyme technology": "btc301",
    "biochemistry and enzyme technology": "btc301",
    "process calculations and thermodynamics": "btc302",
    "microbiology & bioprocess technology": "btc303",
    "microbiology and bioprocess technology": "btc303",
    "database management systems": "csc404",
    "microbiology laboratory": "bts351",
    "biochemistry laboratory": "cys453",
    "database management systems laboratory": "css381",
    "molecular biology & genetic engineering": "btc401",
    "molecular biology and genetic engineering": "btc401",
    "cell biology & genetics": "btc402",
    "cell biology and genetics": "btc402",
    "plant & animal biotechnology": "btc403",
    "plant and animal biotechnology": "btc403",
    "immunology": "btc404",
    "unit operations of chemical engineering i": "chc431",
    "unit operations of chemical engineering 1": "chc431",
    "unit operations of chemical engineering laboratory i": "chs481",
    "unit operations of chemical engineering laboratory 1": "chs481",
    "molecular biology & genetic engineering laboratory": "bts451",
    "molecular biology and genetic engineering laboratory": "bts451",
    "cell biology and genetics laboratory": "bts452",
    "bioreactor design & analysis": "btc501",
    "bioreactor design and analysis": "btc501",
    "bioseparation engineering": "btc502",
    "bioinformatics": "btc503",
    "unit operations of chemical engineering ii": "chc531",
    "unit operations of chemical engineering 2": "chc531",
    "immunology laboratory": "bts551",
    "bioinformatics laboratory": "bts552",
    "unit operations of chemical engineering laboratory ii": "chs581",
    "unit operations of chemical engineering laboratory 2": "chs581",
    "process control & instrumentation": "chc631",
    "process control and instrumentation": "chc631",
    "economics and management accountancy": "hsc631",
    "artificial intelligence & machine learning": "csc631",
    "artificial intelligence and machine learning": "csc602",
    "plant and animal biotechnology laboratory": "bts651",
    "bioseparation engineering laboratory": "bts652",
    "principles of management": "msc731",
    "data analytics in biotechnology": "btc701",
    "bioprocess engineering laboratory": "bts751",
    "summer internship": "css753",
    "project - ii": "ees851",
    "project 2": "ees851",
    "comprehensive viva": "css852",
    "process calculations": "chc301",
    "chemical engineering thermodynamics": "chc302",
    "fluid mechanics": "mec303",
    "numerical methods in chemical engineering": "chc304",
    "industrial chemistry": "cyc504",
    "instrumental analysis laboratory": "cys381",
    "fuel laboratory": "chs351",
    "heat transfer": "chc401",
    "mechanical operations": "chc402",
    "mass transfer- i": "chc403",
    "mass transfer- 1": "chc403",
    "chemical reaction engineering": "chc404",
    "reaction engineering laboratory": "chs451",
    "fluid mechanics laboratory": "chs452",
    "instrumentation and process control": "chc501",
    "mass transfer- ii": "chc502",
    "mass transfer- 2": "chc502",
    "chemical process technology": "chc503",
    "industrial safety and risk management": "chc504",
    "heat transfer laboratory": "mes552",
    "mechanical operations laboratory": "chs552",
    "economics and accountancy": "hsc631",
    "chemical plant design and economics": "chc601",
    "petroleum refining and petrochemicals": "chc602",
    "ai & ml": "csc631",
    "ai and ml": "csc631",
    "process control laboratory": "chs651",
    "mass transfer laboratory": "chs652",
    "chemical process equipment design": "chs653",
    "transport phenomena": "chc702",
    "depth elective - 4": "mme710",
    "depth elective 4": "mme710",
    "depth elective -5": "che72",
    "open elective-1": "yyo74",
    "process modelling and simulation laboratory": "chs751",
    "industrial training / internship and seminar": "chs752",
    "interdisciplinary research-based mini-project": "chs753",
    "capstone project/internship project thesis": "chs851",
    "technical communication on project/internship": "chs852",
    "comprehensive viva voce": "chs853",
    "mathematics - iii": "mac331",
    "state of matter and chemical thermodynamics": "cyc301",
    "atomic structure and chemical bonding": "cyc302",
    "stereochemistry and basic principle of organic chemistry": "cyc303",
    "physics ii": "phc334",
    "physics 2": "phc334",
    "physics ii laboratory": "phs384",
    "qualitative analysis of organic samples laboratory": "cys351",
    "structure and function": "cyc401",
    "phase-equilibrium and chemical kinetics": "cyc402",
    "chemistry of elements and radioactivity": "cyc403",
    "organic reaction mechanism and reactive intermediates": "cyc404",
    "thermodynamic properties of solution and mixture laboratory": "cys451",
    "identification of acidic and basic radicals laboratory": "cys452",
    "fundamentals of electrochemistry and data analysis": "cyc501",
    "chemistry in solution and solid state chemistry": "cyc502",
    "chemistry of heterocyclic compounds": "cyc503",
    "ionic equilibria and surface chemistry": "cyc505",
    "surface chemistry and conductometric analysis": "cys551",
    "quantitative estimation of metal ions in mixture": "cys552",
    "quantitative analysis of organic samples": "cys553",
    "spectroscopy and group theory": "cyc601",
    "coordination chemistry": "cyc602",
    "reagents in organic synthesis": "cyc603",
    "potentiometric and colorimetric analysis": "cys651",
    "analysis of ores and alloys": "cys652",
    "single step organic synthesis laboratory": "cys653",
    "quantum chemistry": "cyc701",
    "inorganic reaction mechanisms and magnetochemistry": "cyc702",
    "concept of organic synthesis and assymteric synthesis": "cyc703",
    "open elective": "yyo74",
    "spectrophotochemical analysis": "cys751",
    "spectrophotometric estimation of cations and anions": "cys752",
    "identification of organic compounds from binary mixture": "cys753",
    "statistical thermodynamics and electrochemistry": "cyc801",
    "organometallic compounds and bioinorganic chemistry": "cyc802",
    "pericyclic reactions and organic photochemistry": "cyc803",
    "theory and applications": "cyc804",
    "advanced practical on physical chemistry": "cys851",
    "synthesis and characterisation of inorganic complexes": "cys852",
    "chromatographic separation of organic compounds": "cys853",
    "mathematics – iii": "mac331",
    "mathematics – 3": "mac331",
    "geology for civil engineering": "esc331",
    "solid mechanics": "mec301",
    "material and concrete technology": "cec302",
    "civil engineering drawing": "ces352",
    "estimation and costing sessional": "ces353",
    "co-curricular activities – iii": "xxs381",
    "co-curricular activities – 3": "xxs381",
    "structural analysis – i": "cec401",
    "structural analysis – 1": "cec401",
    "design of concrete structures": "cec402",
    "water resources engineering": "cec403",
    "environmental engineering": "cec404",
    "surveying": "cec405",
    "structural mechanics laboratory": "cec405",
    "design of concrete structures sessional": "cec405",
    "surveying laboratory": "cec405",
    "co-curricular activities – iv": "xxs481",
    "co-curricular activities – 4": "xxs481",
    "structural analysis – ii": "cec501",
    "structural analysis – 2": "cec501",
    "design of steel structures": "cec502",
    "transportation engineering": "cec503",
    "soil mechanics": "cec504",
    "structural analysis sessional": "ces551",
    "design of steel structures sessional": "ces552",
    "environmental and water resource engineering laboratory": "ces553",
    "co-curricular activities - v": "xxs581",
    "co-curricular activities v": "xxs581",
    "foundation engineering": "cec601",
    "structural engineering laboratory": "ces651",
    "civil engineering computation & software laboratory": "ces652",
    "civil engineering computation and software laboratory": "ces652",
    "soil mechanics and foundation engineering laboratory": "ces653",
    "co-curricular activities - vi": "xxs681",
    "co-curricular activities vi": "xxs681",
    "disaster mitigation and management": "cec701",
    "project - i": "ees755",
    "project 1": "ees755",
    "transportation engineering laboratory": "ces753",
    "project – ii": "css851",
    "project – 2": "css851",
    "discrete mathematics": "csc301",
    "digital logic design": "csc302",
    "algorithm design and analysis- i": "csc303",
    "algorithm design and analysis- 1": "csc303",
    "object oriented programming": "csc304",
    "digital logic design laboratory": "css351",
    "algorithms design laboratory": "css352",
    "object oriented programming laboratory": "css353",
    "computer organization and architecture": "ecc502",
    "theory of computation": "csc402",
    "operating systems": "csc403",
    "computer organization laboratory": "css451",
    "operating systems laboratory": "css452",
    "database managements system laboratory": "css453",
    "compiler design": "csc501",
    "data communication and computer networks": "csc502",
    "embedded systems": "csc503",
    "algorithm design and analysis- ii": "csc504",
    "algorithm design and analysis- 2": "csc504",
    "compiler laboratory": "css551",
    "embedded system laboratory": "css552",
    "software engineering": "csc601",
    "data communication and computer networks laboratory": "css651",
    "software engineering laboratory": "css652",
    "artificial intelligence and machine learning lab": "css653",
    "data science": "csc701",
    "open elective - i": "yyo74",
    "open elective 1": "yyo74",
    "data science laboratory": "css751",
    "internet technologies laboratory": "css752",
    "project-i": "css754",
    "network analysis and synthesis": "eec301",
    "semiconductor devices and technology": "ecc302",
    "signals and systems": "ecc303",
    "digital circuits and systems": "ecc304",
    "network analysis and synthesis laboratory": "ees451",
    "semiconductor devices laboratory": "ecs352",
    "digital circuits and systems laboratory": "ecs353",
    "co-curricular activities - iii": "xxs381",
    "co-curricular activities 3": "xxs381",
    "communication systems i": "ecc401",
    "communication systems 1": "ecc401",
    "digital signal processing": "ecc402",
    "electromagnetic theory and transmission lines": "ecc403",
    "control systems": "eec502",
    "microelectronic circuits": "ecc404",
    "communication systems laboratory i": "ecs451",
    "communication systems laboratory 1": "ecs451",
    "simulation laboratory": "ecs452",
    "microelectronic circuits laboratory": "ecs453",
    "co-curricular activities - iv": "xxs481",
    "co-curricular activities 4": "xxs481",
    "communication systems ii": "ecc501",
    "communication systems 2": "ecc501",
    "microcontrollers and embedded systems": "ecc503",
    "professional elective paper 1": "ece510",
    "communication systems laboratory ii": "ecs551",
    "communication systems laboratory 2": "ecs551",
    "digital signal processing laboratory": "ecs552",
    "microcontrollers and embedded systems laboratory": "ecs553",
    "vlsi design": "ecc601",
    "microwave and antenna engineering": "ecc602",
    "professional elective paper 2": "ece610",
    "professional elective paper 3": "ece610",
    "vlsi design laboratory": "ecs651",
    "microwave and mm wave laboratory": "ecs652",
    "capstone project – i": "ecs653",
    "capstone project – 1": "ecs653",
    "professional ethics for engineers &/ principles of management": "msc731",
    "professional ethics for engineers and/ principles of management": "msc731",
    "professional elective paper 4": "ece710",
    "professional elective paper 5": "ece710",
    "electronic system design laboratory": "ecs751",
    "summer internship and seminar": "mes753",
    "capstone project - ii": "ecs753",
    "capstone project 2": "ecs753",
    "mathematics-iii": "mac331",
    "electrical and electronics measurements": "eec302",
    "electromagnetic field theory": "eec303",
    "analog electronics": "ecc331",
    "analog electronics laboratory": "ecs381",
    "electrical and electronics measurements lab": "ees351",
    "power systems – i": "eec401",
    "power systems – 1": "eec401",
    "electrical machines – i": "eec402",
    "electrical machines – 1": "eec402",
    "digital electronics": "eec403",
    "microprocessor and microcontroller": "eec404",
    "fluid and thermal engineering": "mec431",
    "fluid and thermal engineering laboratory": "mes481",
    "electrical machines – ii": "eec501",
    "electrical machines – 2": "eec501",
    "power systems – ii": "eec503",
    "power systems – 2": "eec503",
    "power electronics": "eec504",
    "depth elective - 1": "mme510",
    "depth elective 1": "mme510",
    "digital electronics laboratory": "ecs581",
    "control systems laboratory": "ees551",
    "electrical machines laboratory – i": "ees552",
    "electrical machines laboratory – 1": "ees552",
    "high voltage and insulation engineering": "eec601",
    "depth elective - 2": "mme610",
    "depth elective 2": "mme610",
    "depth elective - 3": "mme610",
    "depth elective 3": "mme610",
    "electrical machines - ii laboratory": "ees651",
    "electrical machines ii laboratory": "ees651",
    "power electronics laboratory": "ees652",
    "power system laboratory": "ees653",
    "power system operation and control": "eec701",
    "depth elective - 5": "mme710",
    "depth elective 5": "mme710",
    "open elective - 1": "yyo74",
    "microprocessor and microcontroller laboratory": "ees751",
    "high voltage and insulation engineering laboratory": "ees752",
    "electrical machine design sessional": "ees753",
    "vocational training / summer internship and seminar": "ees754",
    "theory of machines and mechanisms": "mec302",
    "engineering thermodynamics": "mec304",
    "machine drawing & solid modeling": "mes351",
    "machine drawing and solid modeling": "mes351",
    "fluid mechanics lab": "mes352",
    "workshop practice-i": "wss381",
    "design of machine elements-i": "mec401",
    "forming and welding": "mec402",
    "heat & mass transfer": "mec403",
    "heat and mass transfer": "mec403",
    "dynamics of machines": "mec404",
    "fluid machines": "mec405",
    "solid mechanics lab": "mes451",
    "mechanism lab": "mes452",
    "workshop practice-ii": "wss481",
    "machining and machine tools": "mec501",
    "ic engine & gas turbine": "mec502",
    "ic engine and gas turbine": "mec502",
    "design of machine elements-ii": "mec503",
    "computer aided manufacturing & robotics": "mec504",
    "computer aided manufacturing and robotics": "mec504",
    "hydraulic machines laboratory": "mes551",
    "ai & mechatronics lab.": "mes553",
    "ai and mechatronics lab.": "mes553",
    "power generation technologies": "mec601",
    "power generation laboratory": "mes651",
    "machine design sessional": "mes652",
    "manufacturing laboratory": "mes653",
    "principle of management": "msc731",
    "industrial engineering & engineering measurement": "mec701",
    "industrial engineering and engineering measurement": "mec701",
    "open elective – 1": "yyo74",
    "machine dynamics laboratory": "mes751",
    "engineering measurement laboratory": "mes752",
    "project- i": "mes754",
    "project- 1": "mes754",
    "project – ii / industry internship": "mes851",
    "comprehensive viva-voce": "mes852",
    "mathematics- iii": "mac331",
    "mathematics- 3": "mac331",
    "introduction to metallurgy and materials": "mmc301",
    "metallurgical thermodynamics and kinetics": "mmc302",
    "non - ferrous process metallurgy": "mmc303",
    "non ferrous process metallurgy": "mmc303",
    "computational materials science": "mmc304",
    "metallurgical thermodynamics and kinetics laboratory": "mms351",
    "mineral beneficiation laboratory": "mms352",
    "computational materials science laboratory": "mms353",
    "transport phenomena in metallurgical processes": "mmc401",
    "phase transformation and phase equilibria": "mmc402",
    "materials characterization": "mmc403",
    "physics of materials": "mmc404",
    "manufacturing processes": "mmc405",
    "transport phenomena laboratory": "mms451",
    "phase transformation and phase equilibria lab": "mms452",
    "materials characterization laboratory": "mms453",
    "modelling and simulation of metallurgical processes": "mmc501",
    "engineering materials and heat treatment": "mmc502",
    "mechanical behaviour of materials": "mmc503",
    "iron making": "mmc504",
    "manufacturing processes laboratory - i": "mms551",
    "manufacturing processes laboratory 1": "mms551",
    "heat treatment of materials laboratory": "mms552",
    "mechanical behaviour of materials laboratory": "mms553",
    "mechanical working of materials": "mmc601",
    "ai) and machine learning": "csc631",
    "mechanical working of materials laboratory": "mms651",
    "manufacturing processes laboratory - ii": "mms652",
    "manufacturing processes laboratory 2": "mms652",
    "steel making": "mmc701",
    "ferrous process metallurgy laboratory": "mms751",
    "materials testing laboratory": "mms752",
    "project – i": "mms754",
    "project – 1": "mms754",
    "probability and statistics": "mac332",
    "digital computer design": "csc305",
    "digital computer design laboratory": "css354",
    "statistical methods laboratory": "mas351",
    "optimization techniques": "mac431",
    "design and analysis of algorithms": "csc405",
    "mathematical finance": "msc531",
    "stochastic process": "mac531",
    "real analysis": "mac532",
    "stochastic process laboratory": "mas551",
    "finance technology laboratory": "mss551",
    "numerical analysis laboratory": "mas651",
    "artificial intelligence and machine learning laboratory": "css653",
    "design and analysis of algorithm laboratory": "css654",
    "functional analysis": "mac731",
    "system design laboratory": "css755"
}

# Direct code passthrough: if someone just says "cs01" or "cec301", match as-is
CODE_RE = re.compile(r"\b([a-z]{2,4}\d{2,3})\b")

# Branch aliases -> the folder name used in the repo. Sem1/2 always live
# under "common" regardless of branch, so those aren't listed here.
BRANCH_MAP = {
    "cse": "cse", "computer science": "cse", "cs": "cse", "computer science and engineering": "cse",
    "cve": "cve", "civil": "cve", "civil engineering": "cve", "ce": "cve",
    "ece": "ece", "electronics": "ece", "electronics and communication": "ece", "electronics and communication engineering": "ece",
    "me": "me", "mechanical": "me", "mechanical engineering": "me",
    "ee": "ee", "electrical": "ee", "electrical engineering": "ee",
    "che": "che", "chemical": "che", "chemical engineering": "che",
    "bt": "bt", "biotech": "bt", "biotechnology": "bt",
    "mme": "mme", "metallurgy": "mme", "metallurgical": "mme", "metallurgical and materials engineering": "mme",
    "mac": "mac", "maths": "mac", "mathematics": "mac", "mathematics and computing": "mac", "mc": "mac", "mnc": "mac",
    "chem": "cy", "chemistry": "cy", "cy": "cy",
    "phy": "ph", "physics": "ph", "ph": "ph"
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


def _extract_semesters(text: str) -> list[int]:
    sems = []
    # explicit sem
    patterns = [
        r"\bsem(?:ester)?\s*[-_]?\s*([1-8])\b",
        r"\b([1-8])(?:st|nd|rd|th)?\s*sem(?:ester)?\b",
        r"\bs([1-8])\b",
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            sems.append(int(m.group(1)))
            
    # year to sems
    year_patterns = {
        r"\b(?:1st|first)\s+(?:yr|year)\b": [1, 2],
        r"\b(?:2nd|second)\s+(?:yr|year)\b": [3, 4],
        r"\b(?:3rd|third)\s+(?:yr|year)\b": [5, 6],
        r"\b(?:4th|fourth)\s+(?:yr|year)\b": [7, 8],
    }
    for pattern, year_sems in year_patterns.items():
        if re.search(pattern, text):
            sems.extend(year_sems)
            
    return sorted(list(set(sems)))


def _codes_for(branch: Optional[str], sems: list[int]) -> list[str]:
    return sorted(
        code for code, entry in MANIFEST.items()
        if (branch is None or entry["branch"] == branch)
        and (not sems or entry["sem"] in sems)
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
    text = message.lower().strip()

    if not any(re.search(p, text) for p in PYQ_TRIGGERS):
        return None

    code = _extract_subject_code(text)
    branch = _extract_branch(text)
    sems = _extract_semesters(text)

    # "give me all sem 3 cse pyqs" -> no single subject named, but "all" +
    # a known sem/branch means: return every subject in that sem/branch.
    if code is None and ALL_RE.search(text) and sems:
        candidates = _codes_for(branch, sems)
        if not candidates and branch is not None:
            candidates = _codes_for("common", sems)  # fall back to common core
        subjects = []
        for c in candidates:
            files = _build_files(c, text, latest_only=True)  # newest paper per subject, not full history
            if files:
                subjects.append({"code": c.upper(), "files": files})
        if subjects:
            sems_str = " & ".join(map(str, sems))
            where = f"{branch or 'common'}, sem {sems_str}" if branch else f"sem {sems_str}"
            return {
                "type": "results",
                "semester": sems[0] if len(sems) == 1 else None,
                "message": f"Latest PYQs for {where}",
                "subjects": subjects,
            }

    if code is None:
        if sems:
            candidates = _codes_for(branch, sems)
            if not candidates and branch is not None:
                candidates = _codes_for("common", sems)
            if candidates:
                sems_str = " & ".join(map(str, sems))
                where = f"{branch or 'common'}, sem {sems_str}" if branch else f"sem {sems_str}"
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