"""
app.py — FastAPI Backend
=========================
Connects index.html to all 4 Python modules.

Run with:
    uvicorn app:app --reload

Then open index.html in your browser.
API docs available at: http://localhost:8000/docs
"""

import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional

# ── Import our 4 modules ───────────────────────────────────────────────────
from module1_preprocessing import preprocess, read_file
from module2_features       import extract_features
from module3_similarity     import compare_documents
from module4_reporting      import generate_html_report


# ══════════════════════════════════════════════════════════════════════════
# APP SETUP
# ══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "PlagScan API",
    description = "NLP-powered plagiarism detection using spaCy, TF-IDF, Cosine & Jaccard similarity.",
    version     = "1.0.0",
)

# Allow the HTML page (opened as a local file) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}

def get_text(file, text):
    """
    Extract raw text and a display name from either an uploaded file or plain text.
    File takes priority if both are provided.
    """
    if file and file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{ext}'. Use .txt, .pdf, or .docx")

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name

        raw_text = read_file(tmp_path)
        os.unlink(tmp_path)
        return raw_text, file.filename

    elif text and text.strip():
        return text.strip(), "Typed Text"

    raise ValueError("No content provided. Upload a file or paste text.")


# ══════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    """Health check — confirm the API is running."""
    return {"status": "ok", "message": "PlagScan API is running. Open index.html in your browser."}


@app.post("/analyze")
async def analyze(
    text1: Optional[str]        = Form(None),
    text2: Optional[str]        = Form(None),
    file1: Optional[UploadFile] = File(None),
    file2: Optional[UploadFile] = File(None),
):
    """
    Main endpoint — receives two documents and returns similarity results.

    Accepts either:
      - text1 / text2  : plain text strings (from textarea)
      - file1 / file2  : uploaded files (.txt, .pdf, .docx)

    Returns JSON with cosine, jaccard, overall score, verdict, and matching passages.
    """

    # Step 1: Extract raw text
    try:
        raw1, name1 = get_text(file1, text1)
        raw2, name2 = get_text(file2, text2)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    if len(raw1.split()) < 5 or len(raw2.split()) < 5:
        return JSONResponse(
            status_code=400,
            content={"error": "Documents are too short. Please provide at least a few sentences."}
        )

    # Step 2: Module 1 — Preprocess
    try:
        doc1 = preprocess(raw1)
        doc2 = preprocess(raw2)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Preprocessing failed: {str(e)}"})

    # Step 3: Module 2 — Extract features
    try:
        features = extract_features([doc1, doc2])
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Feature extraction failed: {str(e)}"})

    # Step 4: Module 3 — Compute similarity
    try:
        result = compare_documents(doc1, doc2, features, doc1_idx=0, doc2_idx=1)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Similarity computation failed: {str(e)}"})

    # Step 5: Module 4 — Generate HTML report (optional, don't fail if it errors)
    try:
        generate_html_report(result, name1, name2, "report.html")
    except Exception:
        pass

    # Return full result
    return {
        "doc1_name":           name1,
        "doc2_name":           name2,
        "cosine_similarity":   result["cosine_similarity"],
        "jaccard_similarity":  result["jaccard_similarity"],
        "overall_score":       result["overall_score"],
        "verdict":             result["verdict"],
        "matching_passages":   result["matching_passages"],
        "doc1_token_count":    len(doc1["tokens"]),
        "doc2_token_count":    len(doc2["tokens"]),
        "doc1_sentence_count": len(doc1["sentences"]),
        "doc2_sentence_count": len(doc2["sentences"]),
    }
