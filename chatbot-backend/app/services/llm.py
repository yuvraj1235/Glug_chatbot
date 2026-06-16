import logging
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
        if not self.is_configured or not self.client:
            return (
                "I am the GLUG Chatbot. I can help you find Previous Year Questions (PYQs). "
                "Please ask for 'PYQs' or specify a semester/subject (e.g., 'Give me DSA PYQs').\n\n"
                "*Note: To enable general AI conversation, please configure your GEMINI_API_KEY in the .env file.*"
            )
        
        try:
            config = types.GenerateContentConfig(
                system_instruction=(
                    "You are the GLUG Chatbot of NIT Durgapur. Be polite, friendly, and helpful. "
                    "If the user asks for PYQs or study materials, guide them to ask for specific semesters or subjects "
                    "so the system can fetch direct links."
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
