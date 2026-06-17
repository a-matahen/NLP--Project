

import os
from datetime import datetime

# Similarity thresholds
HIGH_SIMILARITY = 75
MODERATE_SIMILARITY = 40
LOW_SIMILARITY = 15


def print_report(result: dict,
                 doc1_name: str = "Document 1",
                 doc2_name: str = "Document 2") -> None:
    width = 60

    print("\n" + "=" * width)
    print("  PLAGIARISM DETECTION REPORT")
    print("=" * width)
    print(f"  Date    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Doc 1   : {doc1_name}")
    print(f"  Doc 2   : {doc2_name}")
    print("-" * width)

    cosine = result["cosine_similarity"]
    jaccard = result["jaccard_similarity"]
    overall = result["overall_score"]
    verdict = result["verdict"]

    print(f"\n  Cosine Similarity   : {cosine:.4f} ({cosine*100:.1f}%)")
    print(f"  Jaccard Similarity  : {jaccard:.4f} ({jaccard*100:.1f}%)")
    print(f"\n  Overall Score       : {overall:.2f}%")
    print(f"  Verdict             : {verdict}")

    bar_length = 40
    filled = int((overall / 100) * bar_length)
    bar = "#" * filled + "-" * (bar_length - filled)

    if overall >= HIGH_SIMILARITY:
        label = "HIGH"
    elif overall >= MODERATE_SIMILARITY:
        label = "MODERATE"
    elif overall >= LOW_SIMILARITY:
        label = "LOW"
    else:
        label = "MINIMAL"

    print(f"\n  [{bar}] {label}")
    print("\n" + "=" * width + "\n")


def _get_verdict_color(overall: float) -> str:
    if overall >= HIGH_SIMILARITY:
        return "#e74c3c"
    elif overall >= MODERATE_SIMILARITY:
        return "#e67e22"
    elif overall >= LOW_SIMILARITY:
        return "#f1c40f"
    return "#2ecc71"


def generate_report(result: dict,
                    doc1_name: str = "Document 1",
                    doc2_name: str = "Document 2") -> None:
    print_report(result, doc1_name, doc2_name)


