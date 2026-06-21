# Glug Chatbot Backend

A modern, high-performance, and robust asynchronous FastAPI backend for the Glug Chatbot, powered by the new official Google GenAI SDK and Gemini models (e.g., `gemini-1.5-flash`).

## Features

- **Asynchronous Execution:** Built on FastAPI and leveraging the official `google-genai` asynchronous client (`client.aio`) for fast, non-blocking requests.
- **Real-Time Streaming:** Supports Server-Sent Events (SSE) streaming chunks of chatbot responses to the client.
- **Configurable Personas:** Allows custom system instructions per chat request to direct the chatbot's behavior.
- **Robust Settings Management:** Environment validation powered by Pydantic Settings.
- **Standardized Logging:** Real-time application logs.

---

## Directory Structure

```text
chatbot-backend/
├── app/
│   ├── main.py          # Application entry point, CORS, logging, and router inclusion
│   ├── routes/
│   │   └── chat.py      # /api/chat & /api/chat/models endpoints (supporting SSE streaming)
│   ├── services/
│   │   └── llm.py       # Asynchronous Gemini LLM service using google-genai
│   ├── schemas/
│   │   └── chat.py      # Pydantic request and response schemas
│   └── config.py        # Settings management & validation
├── requirements.txt     # Python dependencies
└── .env                 # Environment variables configuration
```

---

## Setup & Installation

### 1. Prerequisites
- Python 3.10 or higher
- A Gemini API Key (obtained from [Google AI Studio](https://aistudio.google.com/))

### 2. Install Dependencies
Clone the repository, then set up a virtual environment and install the required modules:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy the `.env` template or edit it directly:
```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
ENVIRONMENT=development

# Security & CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:5173,http://localhost:8000

# Gemini LLM Configuration
GEMINI_API_KEY=your_actual_gemini_api_key_here
DEFAULT_MODEL=gemini-1.5-flash
```

---

## Running the Server

Start the development server using Uvicorn (includes live-reload on changes):

```bash
uvicorn app.main:app --reload
```

The application will be accessible at:
- **API Server:** [http://localhost:8000](http://localhost:8000)
- **Interactive Documentation (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative Documentation (Redoc):** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## API Endpoints

### 1. `POST /api/chat`
Handles sending chat history to the chatbot. Supports both full JSON responses and Server-Sent Events (SSE) streaming.

#### Request Body
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hello! What is your name?"
    }
  ],
  "model": "gemini-1.5-flash",
  "temperature": 0.7,
  "system_instruction": "You are a friendly and helpful coding assistant.",
  "stream": false
}
```

*Set `"stream": true` to receive chunk-by-chunk event-stream responses.*

#### Standard Response
```json
{
  "role": "model",
  "content": "Hello! I don't have a personal name, but I am your AI assistant...",
  "model": "gemini-1.5-flash"
}
```

#### Streaming Response (SSE Format)
```text
data: {"content": "Hello"}

data: {"content": "! I"}

data: {"content": " am"}

...

data: [DONE]
```

### 2. `GET /api/chat/models`
Returns the default model and the list of supported Gemini models.

---

## License
MIT
