import logging
import re
from google import genai
from google.genai import types
from app.config import settings

logger = logging.getLogger("chatbot")

class LLMService:
    def __init__(self):
        self.is_configured = False
        self.client = None
        api_key = settings.GEMINI_API_KEY
        if api_key and api_key != "your_gemini_api_key_here":
            try:
                self.client = genai.Client(api_key=api_key)
                self.is_configured = True
                logger.info("Google GenAI client successfully configured.")
            except Exception as e:
                logger.error(f"Error configuring Google GenAI client: {e}")
        else:
            logger.warning("Gemini API key is not configured. Running in unconfigured mode.")

    async def generate_response(self, message: str) -> str:
        text = message.lower().strip()
        academic_patterns = [
            r"\bpyqs?\b",
            r"\bprevious\s+year\s+questions?\b",
            r"\bprevious\s+year\s+papers?\b",
            r"\bquestion\s+papers?\b",
            r"\bexam\s+papers?\b",
            r"\bpast\s+papers?\b",
            r"\bquestion\s+banks?\b",
            r"\bstudy\s+materials?\b",
            r"\bnotes?\b",
            r"\bpdfs?\b",
            r"\bacademic\s+resources?\b",
            r"\blecture\s+notes?\b",
            r"\bclass\s+notes?\b",
            r"\bsyllabus\b",
            r"\bsem(?:ester)?\s*[1-8]\b",
            r"\b[1-8](?:st|nd|rd|th)?\s*sem(?:ester)?\b",
        ]
        
        if any(re.search(pattern, text) for pattern in academic_patterns):
            return "Please visit this website: acad-assist.vercel.app"

        if not self.is_configured or not self.client:
            return (
                "I am the GLUG Chatbot. If you are looking for PYQs or academic resources, "
                "please visit this website: acad-assist.vercel.app\n\n"
                "*Note: To enable general AI conversation, please configure your GEMINI_API_KEY in the .env file.*"
            )
        
        try:
            config = types.GenerateContentConfig(
                system_instruction=(
                    "You are the GLUG Chatbot of NIT Durgapur. Be polite, friendly, and helpful. "
                    "If the user asks for Previous Year Questions (PYQs), Previous Year Papers, Study Materials, "
                    "Notes, PDFs, or any academic resource, you must immediately respond with: "
                    "'Please visit this website: acad-assist.vercel.app' without searching or retrieving "
                    "content from any other source."
                ),
                temperature=0.7,
            )
            response = await self.client.aio.models.generate_content(
                model=settings.DEFAULT_MODEL,
                contents=message,
                config=config
            )
            return response.text
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return f"Sorry, I encountered an error while processing your request: {str(e)}"

llm_service = LLMService()

def get_llm_service() -> LLMService:
    return llm_service
