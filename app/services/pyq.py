import re
from typing import Optional

# Subject mapping dictionary (lowercase phrases -> short code)
SUBJECT_MAP = {
    # Computer Science / IT / Core CS
    "operating systems": "os",
    "operating system": "os",
    "os": "os",
    "data structures and algorithms": "dsa",
    "data structures": "dsa",
    "data structure": "dsa",
    "dsa": "dsa",
    "ds": "dsa",
    "database management systems": "dbms",
    "database management": "dbms",
    "dbms": "dbms",
    "computer networks": "cn",
    "networking": "cn",
    "cn": "cn",
    "design and analysis of algorithms": "daa",
    "algorithms": "daa",
    "algo": "daa",
    "daa": "daa",
    "object oriented programming": "oops",
    "oops": "oops",
    "oop": "oops",
    "cpp": "oops",
    "c++": "oops",
    "computer science": "cse",
    "cse": "cse",
    "cs": "cse",
    "information technology": "it",
    "it": "it",
    
    # Engineering Branches & Subjects
    "civil engineering": "ce",
    "concrete technology": "ce",
    "civil": "ce",
    "ce": "ce",
    "electronics and communication": "ece",
    "electronics": "ece",
    "ece": "ece",
    "ec": "ece",
    "electrical engineering": "ee",
    "electrical": "ee",
    "ee": "ee",
    "mechanical engineering": "me",
    "mechanical": "me",
    "me": "me",
    "chemical engineering": "che",
    "chemical": "che",
    "che": "che",
    "biotechnology": "bt",
    "biotech": "bt",
    "bt": "bt",
    "metallurgical and materials engineering": "mme",
    "metallurgy": "mme",
    "mme": "mme",
    
    # Basic Sciences
    "mathematics": "maths",
    "maths": "maths",
    "math": "maths",
    "physics": "physics",
    "phy": "physics",
    "chemistry": "chemistry",
    "chem": "chemistry",
}

def _get_ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    suffixes = {1: "st", 2: "nd", 3: "rd"}
    return f"{n}{suffixes.get(n % 10, 'th')}"

def _extract_semester(text: str) -> Optional[int]:
    # Match patterns like: sem 3, semester 3, 3rd sem, 3rd semester, sem-3, s3, etc.
    patterns = [
        r"\bsem(?:ester)?\s*[-_]?\s*([1-8])\b",
        r"\b([1-8])(?:st|nd|rd|th)?\s*sem(?:ester)?\b",
        r"\bs([1-8])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None

def _extract_subject_code(text: str) -> str:
    # 1. Match against known dictionary keys (longest match first)
    sorted_keys = sorted(SUBJECT_MAP.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if re.search(r"\b" + re.escape(key) + r"\b", text):
            return SUBJECT_MAP[key]
            
    # 2. Fallback: find alphanumeric words that aren't stop words/numbers/PYQ keywords
    ignored_words = {
        "pyq", "pyqs", "paper", "papers", "question", "questions", "exam", "exams",
        "past", "previous", "year", "years", "sem", "semester", "get", "can", "for",
        "the", "give", "me", "find", "show", "i", "need", "please", "download", "link"
    }
    tokens = re.findall(r"\b[a-z]{2,10}\b", text)
    for token in tokens:
        if token not in ignored_words and not token.isdigit():
            return token
            
    return "general"

def _extract_years(text: str) -> list[str]:
    years = re.findall(r"\b(20\d{2})\b", text)
    if years:
        # Deduplicate preserving order
        unique_years = list(dict.fromkeys(years))
        return sorted(unique_years, reverse=True)
    return ["2023", "2022"]

async def get_pyq_response(message: str) -> Optional[str]:
    """
    Checks if the user's message is asking for PYQs or past exam papers.
    If matched, dynamically constructs CDN links pointing to the GitHub PYQ repository.
    Otherwise, returns None.
    """
    text = message.lower().strip()
    
    # PYQ trigger patterns
    pyq_triggers = [
        r"\bpyqs?\b",
        r"\bprevious\s+year\s+questions?\b",
        r"\bprevious\s+year\s+papers?\b",
        r"\bquestion\s+papers?\b",
        r"\bexam\s+papers?\b",
        r"\bpast\s+papers?\b",
        r"\bquestion\s+banks?\b",
        r"\bpast\s+exams?\b",
    ]
    
    is_pyq_query = any(re.search(pattern, text) for pattern in pyq_triggers)
    if not is_pyq_query:
        return None
        
    sem = _extract_semester(text) or 3  # Default to 3rd sem if unspecified
    subject_code = _extract_subject_code(text)
    years = _extract_years(text)
    
    sem_str = _get_ordinal(sem)
    subj_title = subject_code.upper()
    
    links_markdown = []
    base_url = "https://cdn.jsdelivr.net/gh/Ankit-Gope007/pyq-dummy@main/pyqs"
    
    for year in years:
        url = f"{base_url}/sem{sem}/{subject_code}/{year}.pdf"
        links_markdown.append(f"- [{year} Question Paper]({url})")
        
    response = f"Here are the {sem_str} Semester {subj_title} PYQs:\n" + "\n".join(links_markdown)
    return response

