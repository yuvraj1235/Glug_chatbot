import hashlib
import json
import os
import asyncio
import logging
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import SystemMessage, HumanMessage
from app.config import settings

logger = logging.getLogger("chatbot")

CACHE_FILE = "hf_enrichment_cache.json"

def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"[Enrichment Cache] Failed to load cache: {e}")
    return {}

def save_cache(cache: dict):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.error(f"[Enrichment Cache] Failed to save cache: {e}")

async def enrich_with_hf_summary(
    source: str,
    raw_text: str
) -> str:
    """
    Transforms raw API dumps into clean, uniform natural language paragraphs (prose)
    using AWS Bedrock to eliminate technical syntax boilerplate.
    """
    TABLE_DESCRIPTIONS = {
        "profiles": "Profiles of current students and core members of the club.",
        "alumni": "Profiles of graduated students (alumni).",
        "events": "Details about events, hackathons, and workshops conducted by the club.",
        "project": "Open-source projects or GitHub repositories created by the club.",
        "ctf": "Cybersecurity challenges, CTFs (Capture The Flag), or hacking events.",
        "techbytes": "Technical blogs, write-ups, or articles published by the club."
    }
    table_desc = TABLE_DESCRIPTIONS.get(source, "This data contains general records.")

    if source == "profiles":
        style_instruction = (
            "Convert the raw profile keys and fields into a highly cohesive biographical paragraph. "
            "Write in the third-person active voice. Highlight their full name, role (if visible), graduation or batch year, degree, and specific social profile handles. "
            "Extract any technical skills, tools, or languages mentioned and list them."
        )
    elif source == "events":
        style_instruction = (
            "Convert the raw event fields into an engaging, descriptive paragraph summarizing the technical event. "
            "Incorporate the title, key description facts, core technical domains, exact timelines/dates, prize tracks, and registration URLs. "
            "Extract the core technical domains/skills this event is related to."
        )
    else:
        style_instruction = (
            "Convert the raw data data fields into clear, natural language prose. Eliminate brackets, trailing braces, structural punctuation, and database table metadata keys. "
            "If any technical skills or domains are mentioned, note them."
        )

    system_prompt = (
        "You are an expert data-transformation engineer. "
        f"Your task is to rewrite raw technical text into clean, high-density natural language prose, AND extract skills.\n"
        f"CONTEXT: This data comes from the '{source}' table. {table_desc}\n"
        f"CRITICAL RULES:\n"
        f"1. Follow this style: {style_instruction}\n"
        f"2. Output ONLY a valid JSON object with exactly two keys: 'prose' (string) and 'skills' (list of strings). No conversational preambles, introductory lines, or markdown annotations.\n"
        f"3. Retain every unique personal handle, URL link, date, name, and metric asset precisely in the 'prose'."
    )

    try:
        llm = ChatBedrockConverse(
            model=settings.DEFAULT_MODEL,
            temperature=0.1,
            max_tokens=500,
            region_name=settings.AWS_REGION_NAME,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Raw text data payload to transform:\n{raw_text[:2500]}")
        ]

        response = await llm.ainvoke(messages)
        result_text = response.content.strip()

        # Strip markdown block formatting if the model returns it
        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()

        parsed_response = json.loads(result_text)
        
        return json.dumps({
            "source": source,
            "prose": parsed_response.get("prose", ""),
            "skills": parsed_response.get("skills", [])
        })

    except json.JSONDecodeError:
        logger.error("[Enrichment] Failed to parse JSON response from LLM")
        return json.dumps({
            "source": source, 
            "prose": f"Source Section: {source}\n\n{result_text}",
            "skills": []
        })
    except Exception as e:
        logger.error(f"[Enrichment] Exception for source={source}: {e}")
        return json.dumps({"source": source, "prose": f"Source Section: {source}\n\n{raw_text}", "skills": []})

async def enrich_cached(source: str, raw_text: str, cache: dict) -> str:
    key = hashlib.md5(raw_text.encode()).hexdigest()
    if key in cache:
        return cache[key]
    enriched = await enrich_with_hf_summary(source, raw_text)
    cache[key] = enriched
    return enriched