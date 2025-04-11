import os
import fitz  # PyMuPDF
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import uvicorn

# ========== Gemini API Setup ========== #
GEMINI_API_KEY = "AIzaSyBaYkOY_pT-mPTtsEy-MmdmqrkImtDKTds"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-pro")

# ========== FastAPI Setup ========== #
app = FastAPI(title="Legal Document Q&A API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Prompt Template ========== #
LEGAL_QA_PROMPT = """
You are an expert Indian legal researcher providing precise answers on Indian law.

Context:
{text}

Legal Question:
{question}

Provide a detailed and authoritative answer focused specifically on Indian law. Include references to relevant statutes, sections, case law with proper citations (AIR, SCC, etc.), and legal principles. Your response should reflect current Indian legal position with proper reasoning.
"""

# ========== Utilities ========== #
def extract_text_from_file(file_path: str) -> str:
    if file_path.lower().endswith(".pdf"):
        doc = fitz.open(file_path)
        return "\n".join([page.get_text() for page in doc])
    elif file_path.lower().endswith(".txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError("Unsupported file format. Only PDF and TXT are allowed.")

# ========== API Endpoint ========== #
@app.post("/ask-legal-question")
async def ask_legal_question(file: UploadFile = File(...), question: str = Form(...)):
    try:
        # Save uploaded file temporarily
        suffix = os.path.splitext(file.filename)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # Extract text
        full_text = extract_text_from_file(temp_file_path)
        os.unlink(temp_file_path)  # Clean up temp file

        # Prepare prompt and get Gemini response
        prompt = LEGAL_QA_PROMPT.format(text=full_text[:10000], question=question)
        response = model.generate_content(prompt)

        return JSONResponse(content={"answer": response.text})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
