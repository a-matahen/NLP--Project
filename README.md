# PlagScan — NLP Plagiarism Detector

A plagiarism detection built with spaCy and scikit-learn. Detects potential plagiarism by combining TF-IDF cosine similarity (topic-level) with Jaccard similarity (phrase-level) and highlights matching sentence pairs

---

## Requirements

- Python 3.8 or newer
- Internet connection (for first-time setup only)

---

## Installation

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

---

## Running the Project

### Step 1 — Open a terminal inside the project folder

```bash
cd path/to/plagiarism_detector
```

### Step 2 — Start the FastAPI server

```bash
python -m uvicorn app:app --reload
```

You should see:
```
INFO: Uvicorn running on http://127.0.0.1:8000
```

### Step 3 — Open the web interface

Double-click `index.html` to open it in your browser.
Paste two documents and click **Analyze Documents**.

### Step 4 — View API documentation (optional)

Open your browser and go to:
```
http://localhost:8000/docs
```

---

## Project Structure

```
plagiarism_detector/
├── module1_preprocessing.py   # Clean, tokenize, lemmatize (spaCy)
├── module2_features.py        # N-grams + TF-IDF vectors
├── module3_similarity.py      # Cosine, Jaccard, matching passages
├── module4_reporting.py       # Terminal report output
├── app.py                     # FastAPI backend
├── index.html                 # Web interface
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## How It Works

| Module | Job |
|--------|-----|
| Module 1 | Reads .txt/.pdf/.docx files, cleans text with regex, tokenizes and lemmatizes using spaCy |
| Module 2 | Generates N-grams (unigrams, bigrams, trigrams) and builds TF-IDF vectors |
| Module 3 | Computes cosine similarity (topic-level) and Jaccard similarity (phrase-level), finds matching passages |
| Module 4 | Prints a formatted similarity report to the terminal |

---

## Similarity Score Interpretation

| Score | Verdict |
|-------|---------|
| 75% – 100% | HIGH — Likely plagiarism |
| 40% – 74%  | MODERATE — Possible plagiarism |
| 15% – 39%  | LOW — Probably original |
| 0%  – 14%  | MINIMAL — Original content |

---

## Supported File Types

| Format | Library Required |
|--------|-----------------|
| .txt   | Built-in |
| .pdf   | pdfplumber |
| .docx  | python-docx |

---

## Dependencies

```
spacy>=3.0.0
scikit-learn>=1.0.0
numpy>=1.19.0
fastapi>=0.100.0
uvicorn>=0.20.0
python-multipart>=0.0.6
pdfplumber>=0.9.0
python-docx>=0.8.11
```
