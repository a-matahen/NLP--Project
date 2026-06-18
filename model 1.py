import re
import os
import spacy


# ── Load spaCy model ───────────────────────────────────────────────────────
# Make sure you ran:  python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError("\n[!] spaCy model not found.\n")

file_name="test.txt"

# ══════════════════════════════════════════════════════════════════════════
# 1. FILE READING
# ══════════════════════════════════════════════════════════════════════════

def get_extension(filename: str) -> str:
    file_extension = os.path.splitext(filename)[1].lower()

    if file_extension == ".txt":
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()


    elif file_extension == ".pdf":
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(filename) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
        except ImportError:
            raise ImportError("Install pdfplumber: ")

    elif file_extension == ".docx":
        try:
            from docx import Document
            doc = Document(filename)
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            raise ImportError("Install python-docx:")

    else:
        raise ValueError(f"Unsupported file type: '{file_extension}'. Use .txt, .pdf, or .docx")


# ══════════════════════════════════════════════════════════════════════════
# 2. TEXT CLEANING  (Regular Expressions)
# ══════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:

    text = re.sub(r"<[^>]+>", " ", text)           # 1. strip HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # 2. strip URLs
    text = re.sub(r"[^a-zA-Z\s]", " ", text)       # 3. letters only
    text = re.sub(r"\s+", " ", text).strip()        # 4. collapse spaces
    text = text.lower()                             # 5. lowercase
    return text


# # ══════════════════════════════════════════════════════════════════════════
# 3. TOKENIZATION  (spaCy)
# ══════════════════════════════════════════════════════════════════════════

def tokenize_words(text: str, remove_stopwords: bool = True) -> list[str]:
    """
    Use spaCy to split text into word tokens.
    Filters out:
      - stopwords (the, is, a ...) if remove_stopwords=True
      - punctuation tokens
      - whitespace-only tokens
    Returns a list of word strings.
    """
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        if remove_stopwords and token.is_stop:
            continue
        tokens.append(token.lemma_) #here is this step i combin two steps which is tokenization and lemmatization insted of (token.text)
    return tokens


def tokenize_sentences(text: str) -> list[str]:

    doc = nlp(text)

    sentences = []

    for sent in doc.sents:

        cleaned_sentence = sent.text.strip()

        if cleaned_sentence != "":
            sentences.append(cleaned_sentence)

    return sentences


#==========================================================
#             ---------------------------
#==========================================================
def preprocess_pipline(text: str) -> dict:

    cleaned   = clean_text(text)
    sentences = tokenize_sentences(text)
    tokens_lemmatized    = tokenize_words(cleaned, remove_stopwords=True)


    return {
        "cleaned":   cleaned,
        "sentences": sentences,
        "tokens":    tokens_lemmatized,
    }

