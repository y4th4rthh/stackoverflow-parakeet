from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from google.generativeai import configure, GenerativeModel
from typing import Optional
from gtts import gTTS
import speech_recognition as sr
import uuid
import tempfile
import os
import datetime
import httpx
import io
import time
import re
import random
from system_prompts import SYSTEM_PROMPT

load_dotenv()
app = FastAPI()

MONGO_URI = os.getenv("MONGO_URI")
mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client["neuraai"]
chats_collection = db["chats"]
users_collection = db["users"]

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://neura-ai.netlify.app", "http://localhost:3000", "http://localhost:5173", "https://neura-explore-ai.netlify.app/","https://neura-explore-ai.netlify.app",
                   "https://neura-share.netlify.app","https://dev-neura-ai.netlify.app" ,"https://admin-neura.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static/audio_responses", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


class TextRequest(BaseModel):
    text: str
    model: str = "neura.essence1.o"
    user_id: Optional[str] = None
    sessionId: Optional[str] = None
    incognito: bool


MODEL_CONFIG = {
    "neura.essence1.o": {
        "provider": "gemini",
        "model_name": "gemini-2.5-flash",
        "tts_speed": False,
        "max_tokens": 1000,
        "temperature": 0.7,
    },
    "neura.swift1.o": {
        "provider": "gemini",
        "model_name": "gemini-2.5-flash",
        "tts_speed": False,
        "max_tokens": 1000,
        "temperature": 0.7,
    },
}

# Configure Gemini
configure(api_key=os.getenv("GEMINI_API_KEY"))


# -------------------------------
# STEP 1: CLASSIFY QUERY
# -------------------------------
async def classify_query(query: str) -> str:
    """
    Classify if the query is TECHNICAL (coding, programming, errors, APIs, etc.)
    or GENERAL (non-technical).
    """
    prompt = f"""
Classify the following user query as either TECHNICAL or GENERAL.

Query: "{query}"

Answer only with one word:
- TECHNICAL → if the query is related to coding, programming, debugging, APIs, frameworks, software tools, etc.
- GENERAL → for anything else.
"""
    try:
        model = GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction="You are a classifier that determines if a query is technical or not.",
        )
        response = model.generate_content(prompt)
        answer = response.text.strip().upper()
        return "TECHNICAL" if "TECHNICAL" in answer else "GENERAL"
    except Exception as e:
        print(f"❌ Classification error: {e}")
        return "GENERAL"


# -------------------------------
# STEP 2: STACK OVERFLOW API LOGIC
# -------------------------------
async def stackoverflow_search(query: str, num_results: int = 5):
    """
    Search Stack Overflow for questions related to the given query.
    """
    url = "https://api.stackexchange.com/2.3/search/advanced"
    params = {
        "order": "desc",
        "sort": "relevance",
        "q": query,
        "site": "stackoverflow",
        "pagesize": num_results,
        "filter": "!9_bDDxJY5"  # include body & answers
    }

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        data = resp.json()

    results = []
    for item in data.get("items", []):
        answers_text = ""
        if item.get("answer_count", 0) > 0 and "answers" in item:
            for ans in item["answers"]:
                answers_text += f"- {ans.get('body_markdown', '')[:400]}...\n"

        results.append({
            "title": item.get("title"),
            "link": item.get("link"),
            "snippet": item.get("body_markdown", "")[:400],
            "answers": answers_text
        })

    return results


async def summarize_stackoverflow_results(query, search_data):
    if not search_data:
        return "No relevant Stack Overflow results found."

    text_block = "\n\n".join(
        f"Q: {d['title']}\nSnippet: {d['snippet']}\nAnswers:\n{d['answers']}\n"
        for d in search_data
    )

    prompt = f"""
Summarize the following Stack Overflow discussions for the query "{query}".
Focus on key insights, common solutions, and relevant code ideas.
Use emojis naturally where they help emphasize a point, make instructions clearer.

Use markdown to format responses properly:
- Always leave a blank line **above and below** code blocks to keep things tidy and readable
- Use proper code block formatting with tags (e.g., ```, ```)
- Use markdown tables for comparisons whenever the user asks for a difference, comparison, or versus-style question
- Use lists, headings, and emphasis where helpful
- Use **plain code blocks** with triple backticks only — do not add any language tag (like `python`, `javascript`, `html`, etc.)
Example:
```
# Your code here

```

- Never use: ```python or ```javascript (no language after the backticks)


Stack Overflow Data:
{text_block}
"""

    try:
        model = GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction="You are a helpful AI that summarizes Stack Overflow discussions into concise technical insights."
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"❌ Gemini summarization error: {e}")
        return "Summarization failed. Please try again."


async def stackoverflow_ai_answer(query):
    print(f"💻 Searching Stack Overflow for: {query}")
    search_data = await stackoverflow_search(query)
    summary = await summarize_stackoverflow_results(query, search_data)

    formatted_urls = "\n\n".join([d["link"] for d in search_data]) if search_data else "No links found."
    return f"🧠 **Stack Overflow Summary:**\n\n \n\n{summary}\n\n \n\n🔗 **Sources:**\n\n \n\n{formatted_urls}"

def create_tts_with_retry(text, filepath, max_retries=3):
    """Create TTS with retry logic and exponential backoff"""
    for attempt in range(max_retries):
        try:
            # Add jitter to prevent thundering herd
            if attempt > 0:
                delay = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)

            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(filepath)
            return True

        except Exception as e:
            print(f"TTS attempt {attempt + 1} failed: {e}")
            if "429" in str(e) or "Too Many Requests" in str(e):
                if attempt == max_retries - 1:
                    print("Max retries reached for TTS")
                    return False
                continue
            else:
                # For non-rate-limit errors, don't retry
                print(f"Non-rate-limit TTS error: {e}")
                return False

    return False



EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F" 
    "\U0001F300-\U0001F5FF"  
    "\U0001F680-\U0001F6FF"  
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

def sanitize_text_for_tts(text: str) -> str:
    """
    Remove Markdown symbols and code snippets from the text before TTS.
    """
    # Remove code blocks (```...```)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    # Remove inline code (`...`)
    text = re.sub(r"`[^`]+`", "", text)

    # Remove markdown links but keep the text part [text](url)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Remove bold/italic markers (**text**, __text__, *text*, _text_)
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)

    # Remove markdown headers (# Header)
    text = re.sub(r"#+\s*", "", text)

    # Remove remaining symbols like > or -
    text = re.sub(r"[>`\-]+", "", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    text = EMOJI_PATTERN.sub(" ", text)

    return text.strip()


# -------------------------------
# STEP 3: MAIN CHAT ENDPOINT
# -------------------------------
@app.post("/search")
async def chat(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="No text provided")

    if req.model not in MODEL_CONFIG:
        raise HTTPException(status_code=400, detail=f"Invalid model: {req.model}")

    try:
        session_id = req.sessionId or str(uuid.uuid4())
        userId = req.user_id
        config = MODEL_CONFIG[req.model]
        chatvalue = ""

        # Classify query as technical or general
        category = await classify_query(req.text)
        print(f"🧠 Query classified as: {category}")

        if category == "TECHNICAL":
            ai_response = await stackoverflow_ai_answer(query=req.text)
        else:
            # General query → handled by Gemini directly
            if req.sessionId:
                chatdatas = chats_collection.find({"session_id": req.sessionId})
                async for chatval in chatdatas:
                    chatvalue += chatval["user_text"] + chatval["ai_response"]

                sysPrompt = SYSTEM_PROMPT + f"\n\nUse prior chat context only if related.\nHistory:\n{chatvalue}\n"
            else:
                sysPrompt = SYSTEM_PROMPT

            model = GenerativeModel(
                model_name="gemini-2.5-flash",
                system_instruction=sysPrompt
            )
            response = model.generate_content(req.text)
            ai_response = response.text

        # Save chat if not incognito
        if not req.incognito:
            chat_doc = {
                "session_id": session_id,
                "timestamp": datetime.datetime.utcnow(),
                "user_text": req.text,
                "user_id": userId,
                "model": req.model,
                "ai_response": ai_response
            }
            await chats_collection.insert_one(chat_doc)

               # Generate unique filename
        filename = f"{uuid.uuid4()}.mp3"
        filepath = f"static/audio_responses/{filename}"

        # Create audio response with retry logic
        audio_url = None
        if req.model == "neura.swift1.o":
            print("AI RES", ai_response)
            sanitized_text = sanitize_text_for_tts(ai_response)
            tts_success = create_tts_with_retry(sanitized_text, filepath)

            if tts_success:
                audio_url = f"/static/audio_responses/{filename}"
            else:
                print("TTS failed after retries - returning text only")

        
        if audio_url:
            response_data = {"text": ai_response, "session_id": session_id, "audio_url": audio_url}
        else:
            response_data = {"text": ai_response, "session_id": session_id}


        return response_data

    except Exception as e:
        return {"text": "Your daily quota has expired. Please switch to another model or try later :(", "session_id": session_id}
        # raise HTTPException(status_code=500, detail=f"Chat processing error: {e}")

@app.get("/ping")
async def ping():
    """Keep-alive / health check endpoint"""
    return {"status": "ok"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)










