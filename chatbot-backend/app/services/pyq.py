import re
from typing import Optional, Tuple
from app.services.drive import drive_service


# Database of PYQs pointing to local files served by the backend
PYQ_DATABASE = {
    "semesters": {
        "1": "https://nitdgp.ac.in/p/semester-question-papers",
        "2": "https://nitdgp.ac.in/p/semester-question-papers",
        "3": "https://nitdgp.ac.in/p/semester-question-papers",
        "4": "https://nitdgp.ac.in/p/semester-question-papers",
        "5": "https://nitdgp.ac.in/p/semester-question-papers",
        "6": "https://nitdgp.ac.in/p/semester-question-papers",
        "7": "https://nitdgp.ac.in/p/semester-question-papers",
        "8": "https://nitdgp.ac.in/p/semester-question-papers",
    },
    "subjects": {
        "dsa": {
            "name": "Data Structures & Algorithms (DSA)",
            "keywords": [r"\bdsa\b", r"\bdata\s+structures?\b", r"\balgorithms?\b"],
            "links": [
                {"year": "2023 (End-Sem)", "url": "/static/pyqs/dsa_2023.pdf"},
                {"year": "2022 (End-Sem)", "url": "/static/pyqs/dsa_2022.pdf"},
            ]
        },
        "os": {
            "name": "Operating Systems (OS)",
            "keywords": [r"\bos\b", r"\boperating\s+systems?\b"],
            "links": [
                {"year": "2023 (End-Sem)", "url": "/static/pyqs/os_2023.pdf"},
                {"year": "2022 (End-Sem)", "url": "/static/pyqs/os_2022.pdf"}
            ]
        },
        "dbms": {
            "name": "Database Management Systems (DBMS)",
            "keywords": [r"\bdbms\b", r"\bdatabases?\b", r"\bdatabase\s+management\b"],
            "links": [
                {"year": "2023 (End-Sem)", "url": "/static/pyqs/dbms_2023.pdf"},
                {"year": "2022 (End-Sem)", "url": "/static/pyqs/dbms_2022.pdf"}
            ]
        },
        "cn": {
            "name": "Computer Networks (CN)",
            "keywords": [r"\bcn\b", r"\bcomputer\s+networks?\b", r"\bnetworks?\b"],
            "links": [
                {"year": "2023 (End-Sem)", "url": "/static/pyqs/cn_2023.pdf"},
                {"year": "2022 (End-Sem)", "url": "/static/pyqs/cn_2022.pdf"}
            ]
        },
        "oop": {
            "name": "Object-Oriented Programming (OOP)",
            "keywords": [r"\boop\b", r"\bobject\s+oriented\b", r"\boops\b", r"\bjava\b", r"\bc\+\+\b"],
            "links": [
                {"year": "2023 (End-Sem)", "url": "/static/pyqs/oop_2023.pdf"},
                {"year": "2022 (End-Sem)", "url": "/static/pyqs/oop_2022.pdf"}
            ]
        },
        "toc": {
            "name": "Theory of Computation (TOC)",
            "keywords": [r"\btoc\b", r"\btheory\s+of\s+computation\b", r"\bautomata\b"],
            "links": [
                {"year": "2023 (End-Sem)", "url": "/static/pyqs/toc_2023.pdf"},
                {"year": "2022 (End-Sem)", "url": "/static/pyqs/toc_2022.pdf"}
            ]
        },
        "coa": {
            "name": "Computer Organization & Architecture (COA)",
            "keywords": [r"\bcoa\b", r"\bcomputer\s+organization\b", r"\barchitecture\b"],
            "links": [
                {"year": "2023 (End-Sem)", "url": "/static/pyqs/coa_2023.pdf"},
                {"year": "2022 (End-Sem)", "url": "/static/pyqs/coa_2022.pdf"}
            ]
        }
    }
}

async def get_pyq_response(message: str) -> Optional[Tuple[str, Optional[str]]]:
    """
    Checks if the user's message is requesting a PYQ.
    If so, parses details (semester or subject) and returns a tuple (text_response, file_url).
    If not requesting a PYQ, returns None.
    """
    text = message.lower().strip()
    
    # List of triggers indicating PYQ request
    pyq_triggers = ["pyq", "previous year", "question paper", "past paper", "exam paper", "question bank"]
    is_asking_pyq = any(trigger in text for trigger in pyq_triggers)
    
    if not is_asking_pyq:
        return None

    # 1. Check for specific Semester requests (e.g. "sem 3", "3rd sem", "semester 3")
    sem_patterns = [
        r"\bsem(?:ester)?\s*([1-8])\b",
        r"\b([1-8])(?:st|nd|rd|th)?\s*sem(?:ester)?\b"
    ]
    for pattern in sem_patterns:
        match = re.search(pattern, text)
        if match:
            sem_num = match.group(1)
            sem_url = PYQ_DATABASE["semesters"].get(sem_num)
            if sem_url:
                return (
                    f"Here is the official Semester Question Papers link for **Semester {sem_num}**:\n"
                    f"🔗 [{sem_num}st Semester PYQs]({sem_url})\n\n"
                    f"You can also browse all semesters directly on the NIT Durgapur portal.",
                    None
                )

    # 2. Check for specific Subject requests
    for sub_key, sub_data in PYQ_DATABASE["subjects"].items():
        for keyword in sub_data["keywords"]:
            if re.search(keyword, text):
                # Check if a specific year is requested (e.g. 2022 vs 2023)
                selected_item = sub_data["links"][0]  # Default to the most recent
                year_val = "2023"  # Default year value
                
                for item in sub_data["links"]:
                    year_match = re.search(r"\b(20\d{2})\b", item["year"])
                    if year_match:
                        y = year_match.group(1)
                        if y in text:
                            selected_item = item
                            year_val = y
                            break

                # Try fetching dynamically from Google Drive folder first
                drive_file_url = drive_service.fetch_pyq_file(sub_key, year_val)
                if not drive_file_url:
                    drive_file_url = drive_service.fetch_pyq_file(sub_data["name"], year_val)

                # If found on Drive, use it! Otherwise fall back to local database mapping
                final_url = drive_file_url if drive_file_url else selected_item["url"]
                source_info = "Google Drive" if drive_file_url else "local cache"

                links_text = "\n".join([f"- **{item['year']}:** [Download/View]({item['url']})" for item in sub_data["links"]])
                response_message = (
                    f"Here are the Previous Year Question Papers (PYQs) for **{sub_data['name']}**.\n"
                    f"I have delivered the **{year_val}** PDF file directly from our {source_info} to your chat window.\n\n"
                    f"All local files:\n{links_text}"
                )
                return response_message, final_url

    # 3. Default fallback: generic PYQ request
    subjects_list = "\n".join([f"- **{sub_data['name']}** (e.g. *'{sub_key}'*)" for sub_key, sub_data in PYQ_DATABASE["subjects"].items()])
    
    response_message = (
        "Sure! I can help you download Previous Year Questions (PYQs) directly.\n\n"
        "I have attached the **All PYQs Package (ZIP)** containing all available subjects.\n"
        "If you want a specific subject, please specify it in your message (e.g. *\"Give me DSA PYQs\"*).\n\n"
        "**Available Semesters:** Semesters 1 to 8\n\n"
        "**Common CSE Subjects:**\n"
        f"{subjects_list}\n\n"
        "You can also check the official institute question repository directly: https://nitdgp.ac.in"
    )
    return response_message, "/static/pyqs/all_pyqs.zip"
