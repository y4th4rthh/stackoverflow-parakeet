from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from dotenv import load_dotenv
from google.generativeai import configure, GenerativeModel
from typing import Optional
import uuid
import os
import datetime
import httpx
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

    formatted_urls = "\n".join([d["link"] for d in search_data]) if search_data else "No links found."
    return f"🧠 **Stack Overflow Summary:**\n\n{summary}\n\n\n\n🔗 **Sources:**\n{formatted_urls}"


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

        return {"text": ai_response, "session_id": session_id}

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







