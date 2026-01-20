import json
import pandas as pd
from typing import List, Dict, Set
from nltk.tokenize import word_tokenize
import nltk
import os

# Ensure NLTK tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ========================== HELPERS ==========================

def clean_title(t: str) -> str:
    """Normalize titles for consistent matching."""
    if not t:
        return ""
    return (
        t.lower()
        .strip()
        .replace(".", "")
        .replace(",", "")
        .replace(":", "")
        .replace(";", "")
    )

def load_jsonl(file_path: str) -> List[Dict]:
    """Load a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Phrase-level Jaccard."""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def token_jaccard(list_a: List[str], list_b: List[str]) -> float:
    """Word-level Jaccard."""
    tokens_a = set(word_tokenize(" ".join(list_a).lower()))
    tokens_b = set(word_tokenize(" ".join(list_b).lower()))
    return jaccard_similarity(tokens_a, tokens_b)

def coverage_ratio(originals: List[str], cleaned: List[str]) -> float:
    """Measures how many original keywords appear in cleaned version."""
    if not originals:
        return 0.0
    covered = 0
    for orig in originals:
        orig_tokens = set(word_tokenize(orig.lower()))
        for clean in cleaned:
            clean_tokens = set(word_tokenize(clean.lower()))
            if orig_tokens & clean_tokens:
                covered += 1
                break
    return covered / len(originals)

def conciseness_score(originals: List[str], cleaned: List[str]) -> float:
    """Lower is better (less noise)."""
    return len(cleaned) / max(len(originals), 1)

def parse_keywords(raw):
    """Parse pipe-text, list, or JSON string robustly."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        if "|" in raw:
            return [x.strip() for x in raw.split("|") if x.strip()]
        else:
            return [raw.strip()]
    return []

# ========================== CONFIG PATHS ==========================

# Single prompt outputs
HYBRID_FILE = r"dblp_keywords_hybrid_parallel_retry.jsonl"

# Parallel prompt outputs
PARALLEL_LLM_FILES = {
    "gpt-4o-mini": r"llms_using_parallel_prompt\parallel_cleaned.gpt-4o-mini.jsonl",
    "deepseek-paid": r"llms_using_parallel_prompt\parallel_cleaned.deepseek-r1-qwen3-8b.jsonl",
    "qwen-2.5-72b": r"llms_using_parallel_prompt\parallel_cleaned.qwen-2.5-72b-instruct.jsonl",
    "grok-4.1-fast": r"llms_using_parallel_prompt\parallel_cleaned.grok-4.1-fast.jsonl",
    "mistral-7b": r"llms_using_parallel_prompt\parallel_cleaned.mistral-7b-instruct.jsonl",
}

FINAL_CLEANED_FIELD = "cleaned_keywords"

# ========================== LOAD SINGLE PROMPT (HYBRID) ==========================

hybrid_data = load_jsonl(HYBRID_FILE)
hybrid_dict = {clean_title(p.get("title", "")): p for p in hybrid_data}
print(f"Loaded single prompt (hybrid) papers: {len(hybrid_dict)}")

# ========================== LOAD PARALLEL PROMPT LLMs ==========================

llm_data = {}
for model, file_path in PARALLEL_LLM_FILES.items():
    full = os.path.abspath(file_path)
    if not os.path.exists(full):
        print(f"File missing: {model}")
        continue
    data = load_jsonl(full)
    llm_data[model] = {clean_title(p.get("title", "")): p for p in data}
    print(f"{model} loaded: {len(llm_data[model])} papers")

# ========================== FIND COMMON PAPERS ==========================

common_papers = set(hybrid_dict.keys())
for model_dict in llm_data.values():
    common_papers &= set(model_dict.keys())

print(f"\nCommon papers across all models: {len(common_papers)}")

if len(common_papers) == 0:
    print("ERROR: No matching titles. Your datasets must be aligned.")
    exit()

# ========================== METRIC COMPUTATION ==========================

results = []

for title in common_papers:
    hybrid_p = hybrid_dict[title]
    orig_keywords = parse_keywords(hybrid_p.get(FINAL_CLEANED_FIELD, []))  # single prompt keywords

    for model, model_dict in llm_data.items():
        llm_p = model_dict[title]
        llm_keywords = parse_keywords(llm_p.get(FINAL_CLEANED_FIELD, []))  # parallel prompt keywords

        results.append({
            "paper_title": title,
            "model": model,
            "jaccard_phrase": jaccard_similarity(set(orig_keywords), set(llm_keywords)),
            "jaccard_token": token_jaccard(orig_keywords, llm_keywords),
            "coverage": coverage_ratio(orig_keywords, llm_keywords),
            "conciseness": conciseness_score(orig_keywords, llm_keywords),
            "num_cleaned": len(llm_keywords)
        })

# ========================== SUMMARY ==========================

df = pd.DataFrame(results)

summary = df.groupby("model").agg({
    "jaccard_phrase": ["mean", "std"],
    "jaccard_token": ["mean", "std"],
    "coverage": "mean",
    "conciseness": "mean",
    "num_cleaned": "mean"
}).round(3)

print("\n=== SUMMARY METRICS (Single vs Parallel) ===")
print(summary)

# Average of phrase + token Jaccard for best model
df["avg_jaccard"] = (df["jaccard_phrase"] + df["jaccard_token"]) / 2
best_model = df.groupby("model")["avg_jaccard"].mean().idxmax()
print(f"\nBest model overall (Parallel vs Single): {best_model}")

# Save detailed comparison
output_csv = "hybrid_vs_Parallel_comparison_metrics.csv"
df.to_csv(output_csv, index=False)
print(f"\nSaved detailed results to {output_csv}")
