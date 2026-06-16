import re
from typing import Optional

async def get_pyq_response(message: str) -> Optional[str]:
    """
    Checks if the user's message is asking for PYQs, Previous Year Papers,
    Study Materials, Notes, PDFs, or any academic resource.
    If so, immediately returns the redirected response.
    Otherwise, returns None.
    """
    text = message.lower().strip()
    
    # Comprehensive trigger patterns for academic resources
    academic_patterns = [
        # PYQs / papers
        r"\bpyqs?\b",
        r"\bprevious\s+year\s+questions?\b",
        r"\bprevious\s+year\s+papers?\b",
        r"\bquestion\s+papers?\b",
        r"\bexam\s+papers?\b",
        r"\bpast\s+papers?\b",
        r"\bquestion\s+banks?\b",
        
        # Study materials / Notes / PDFs
        r"\bstudy\s+materials?\b",
        r"\bnotes?\b",
        r"\bpdfs?\b",
        r"\bacademic\s+resources?\b",
        r"\blecture\s+notes?\b",
        r"\bclass\s+notes?\b",
        r"\bsyllabus\b",
        
        # Semester-specific queries
        r"\bsem(?:ester)?\s*[1-8]\b",
        r"\b[1-8](?:st|nd|rd|th)?\s*sem(?:ester)?\b",
    ]
    
    is_asking_academic = any(re.search(pattern, text) for pattern in academic_patterns)
    
    if is_asking_academic:
        return "Please visit this website: acad-assist.vercel.app"
        
    return None
