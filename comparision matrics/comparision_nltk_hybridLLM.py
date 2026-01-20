import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score

def jaccard_similarity_list(a, b):
    """Compute Jaccard similarity between 2 keyword lists."""
    if not a or not b:
        return 0.0
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(" | "))
    mat = vectorizer.fit_transform([" | ".join(a), " | ".join(b)]).toarray()
    return jaccard_score(mat[0], mat[1], average='macro')


# ======== Load Your Files ========
NLTK_FILE = r"preprocessing using_nltk\nltk_dblp_cleaned_1000_random_samples_keywords_cleaned.jsonl"
HYBRID_FILE = r"dblp_keywords_hybrid_parallel_retry.jsonl"

nltk_data = [json.loads(line) for line in open(NLTK_FILE, "r", encoding="utf-8")]
hybrid_data = [json.loads(line) for line in open(HYBRID_FILE, "r", encoding="utf-8")]

# Convert hybrid list -> dict for fast lookup
hybrid_dict = {d["paper_id"]: d for d in hybrid_data}


# ======== Collect Metrics ========
rows = []

for item in nltk_data:
    pid = item["id"]
    if pid not in hybrid_dict:
        continue

    orig = item["keywords"]
    clean = hybrid_dict[pid]["cleaned_keywords"].split(" | ")

    orig_count = len(orig)
    clean_count = len(clean)

    # Flatten words for avg length
    def avg_len(lst):
        toks = " ".join(lst).split()
        return sum(len(w) for w in toks) / max(1, len(toks))

    row = {
        "paper_id": pid,
        "orig_count": orig_count,
        "clean_count": clean_count,
        "reduction_ratio": round((orig_count - clean_count) / max(orig_count, 1), 4),
        "retention_ratio": round(clean_count / max(orig_count, 1), 4),
        "avg_length_original": round(avg_len(orig), 3),
        "avg_length_cleaned": round(avg_len(clean), 3),
        "jaccard_similarity": round(jaccard_similarity_list(orig, clean), 4),
        "compression_score": round(1 - (clean_count / max(orig_count, 1)), 4),
    }

    rows.append(row)

df = pd.DataFrame(rows)

print("\n===== METRICS SUMMARY =====\n")
print(df.head())

print("\n===== OVERALL AVERAGE METRICS =====\n")
print(df.mean(numeric_only=True))

# Save summary for publication
df.to_csv("keyword_processing_metrics_summary.csv", index=False)
print("\nSaved → keyword_processing_metrics_summary.csv")
