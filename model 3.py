import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ══════════════════════════════════════════════════════════════════════════
# 1. COSINE SIMILARITY
# ══════════════════════════════════════════════════════════════════════════

def compute_cosine_similarity(tfidf_matrix, idx1: int, idx2: int) -> float:

    vec1 = tfidf_matrix[idx1]
    vec2 = tfidf_matrix[idx2]
    score = cosine_similarity(vec1, vec2)[0][0]
    return round(float(score), 4)


# ══════════════════════════════════════════════════════════════════════════
# 2. JACCARD SIMILARITY
# ══════════════════════════════════════════════════════════════════════════

def compute_jaccard_similarity(ngrams1: list, ngrams2: list) -> float:

    set1 = set(ngrams1)
    set2 = set(ngrams2)

    intersection = set1 & set2
    union        = set1 | set2

    if not union:
        return 0.0

    score = len(intersection) / len(union)
    return round(score, 4)


# ══════════════════════════════════════════════════════════════════════════
# 3. OVERALL SIMILARITY SCORE
# ══════════════════════════════════════════════════════════════════════════

def compute_overall_score(cosine: float, jaccard: float,
                          cosine_weight: float = 0.6,
                          jaccard_weight: float = 0.4) -> float:

    score = (cosine * cosine_weight) + (jaccard * jaccard_weight)
    return round(score * 100, 2)


# ══════════════════════════════════════════════════════════════════════════
# 4. MATCHING PASSAGE DETECTION
# ══════════════════════════════════════════════════════════════════════════

def find_matching_passages(sentences1: list[str], sentences2: list[str],
                           threshold: float = 0.5) -> list[dict]:

    from sklearn.feature_extraction.text import TfidfVectorizer

    matches = []

    if not sentences1 or not sentences2:
        return matches

    # Build a TF-IDF matrix across all sentences from both docs
    all_sentences = sentences1 + sentences2
    try:
        vectorizer = TfidfVectorizer().fit(all_sentences)
    except ValueError:
        return matches

    vecs1 = vectorizer.transform(sentences1)
    vecs2 = vectorizer.transform(sentences2)

    # Compare every sentence in doc1 against every sentence in doc2
    similarity_matrix = cosine_similarity(vecs1, vecs2)

    for i, row in enumerate(similarity_matrix):
        best_j     = int(np.argmax(row))   # index of best matching sentence in doc2
        best_score = float(row[best_j])

        if best_score >= threshold:
            matches.append({
                "sentence1":  sentences1[i],
                "sentence2":  sentences2[best_j],
                "similarity": round(best_score, 4),
            })

    # Sort by similarity descending so strongest matches appear first
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    return matches


# ══════════════════════════════════════════════════════════════════════════
# 5. FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def compare_documents(doc1_preprocessed: dict, doc2_preprocessed: dict,
                      features: dict,
                      doc1_idx: int = 0, doc2_idx: int = 1) -> dict:

    # 1. Cosine similarity from TF-IDF vectors
    cosine = compute_cosine_similarity(
        features["tfidf_matrix"], doc1_idx, doc2_idx
    )

    # 2. Jaccard similarity from bigrams (good balance of specificity)
    bigrams1 = features["ngrams"][doc1_idx]["bigrams"]
    bigrams2 = features["ngrams"][doc2_idx]["bigrams"]
    jaccard  = compute_jaccard_similarity(bigrams1, bigrams2)

    # 3. Combined overall score
    overall = compute_overall_score(cosine, jaccard)

    # 4. Verdict label based on score
    if overall >= 75:
        verdict = "HIGH SIMILARITY — Likely plagiarism"
    elif overall >= 40:
        verdict = "MODERATE SIMILARITY — Possible plagiarism"
    elif overall >= 15:
        verdict = "LOW SIMILARITY — Probably original"
    else:
        verdict = "MINIMAL SIMILARITY — Original content"

    # 5. Find matching passages
    matches = find_matching_passages(
        doc1_preprocessed["sentences"],
        doc2_preprocessed["sentences"],
        threshold=0.5,
    )

    return {
        "cosine_similarity":  cosine,
        "jaccard_similarity": jaccard,
        "overall_score":      overall,
        "verdict":            verdict,
        "matching_passages":  matches,
    }

