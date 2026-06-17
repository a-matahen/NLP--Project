from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import numpy as np


# ══════════════════════════════════════════════════════════════════════════
# 1. N-GRAM GENERATION
# ══════════════════════════════════════════════════════════════════════════

def generate_ngrams(tokens: list[str], n: int) -> list[tuple]:
    ngrams = []

    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n])

        ngrams.append(gram)

    return ngrams

def get_all_ngrams(tokens: list[str]) -> dict:

    return {
        "unigrams": generate_ngrams(tokens, 1),
        "bigrams":  generate_ngrams(tokens, 2),
        "trigrams": generate_ngrams(tokens, 3),
    }


def ngram_counts(tokens: list[str], n: int) -> dict:

    grams = generate_ngrams(tokens, n)

    counter = Counter(grams)

    return dict(counter)

# ══════════════════════════════════════════════════════════════════════════
# 2. TF-IDF VECTORIZATION
# ══════════════════════════════════════════════════════════════════════════

def build_tfidf_vectors(cleaned_texts: list[str]) -> tuple:

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),   # include unigrams AND bigrams
        min_df=1,             # include a term even if it appears in only 1 doc
        sublinear_tf=True,    # apply log to term frequency (reduces impact of very common words)
    )

    vectors = vectorizer.fit_transform(cleaned_texts)
    feature_names = vectorizer.get_feature_names_out()

    return vectors, vectorizer, feature_names


def get_top_terms(vector, feature_names, top_n: int = 10) -> list[tuple]:

    # Convert sparse row to a flat numpy array
    dense = np.asarray(vector.todense()).flatten()

    # Pair each term with its score, sort by score descending
    scores = sorted(
        zip(feature_names, dense),
        key=lambda x: x[1],
        reverse=True
    )
    return [(term, round(score, 4)) for term, score in scores[:top_n] if score > 0]


# ══════════════════════════════════════════════════════════════════════════
# 3. FULL PIPELINE  (combines all steps)
# ══════════════════════════════════════════════════════════════════════════

def extract_features(preprocessed_docs: list[dict]) -> dict:

    # Extract cleaned strings and token lists
    cleaned_texts = []

    for doc in preprocessed_docs:
        cleaned_texts.append(doc["cleaned"])

    token_lists = []

    for doc in preprocessed_docs:
        token_lists.append(doc["tokens"])


    # Generate N-grams for each document
    all_ngrams = []

    for tokens in token_lists:
        all_ngrams.append(
            get_all_ngrams(tokens)
        )

        
    # Build TF-IDF vectors across all documents
    tfidf_matrix, vectorizer, feature_names = build_tfidf_vectors(cleaned_texts)

    # Get top terms for each document
    top_terms = [
        get_top_terms(tfidf_matrix[i], feature_names)
        for i in range(len(preprocessed_docs))
    ]

    return {
        "ngrams":        all_ngrams,
        "tfidf_matrix":  tfidf_matrix,
        "vectorizer":    vectorizer,
        "feature_names": feature_names,
        "top_terms":     top_terms,
    }

