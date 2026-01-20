import json
import os
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk

# Ensure NLTK punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# ================== Helper Functions ==================
def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def parse_keywords(raw_input):
    if not raw_input:
        return []
    
    if isinstance(raw_input, str):
        if raw_input.strip().startswith('{'):
            try:
                parsed = json.loads(raw_input)
                if isinstance(parsed, dict) and "cleaned_keywords" in parsed:
                    raw_input = parsed["cleaned_keywords"]
                elif isinstance(parsed, list):
                    raw_input = parsed
            except:
                pass
        if isinstance(raw_input, str):
            return [kw.strip() for kw in raw_input.split('|') if kw.strip()]
    elif isinstance(raw_input, list):
        return [str(kw).strip() for kw in raw_input if kw.strip()]
    return []


def jaccard_similarity(set_a, set_b):
    set_a, set_b = set(set_a), set(set_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a.intersection(set_b)) / len(set_a.union(set_b))


def token_jaccard(list_a, list_b):
    tokens_a = set(word_tokenize(" ".join(list_a).lower()))
    tokens_b = set(word_tokenize(" ".join(list_b).lower()))
    return jaccard_similarity(tokens_a, tokens_b)


def coverage_ratio(originals, cleaned, threshold_words=1):
    orig_set = set(originals)
    covered = 0
    for orig in orig_set:
        orig_tokens = set(word_tokenize(orig.lower()))
        for clean in cleaned:
            clean_tokens = set(word_tokenize(clean.lower()))
            if len(orig_tokens.intersection(clean_tokens)) >= threshold_words:
                covered += 1
                break
    return covered / len(orig_set) if orig_set else 0.0


def conciseness_score(originals, cleaned):
    return len(cleaned) / max(len(originals), 1)


# ================== CONFIG ==================
HYBRID_FILE = r"dblp_keywords_hybrid_parallel_retry.jsonl"
LLM_FOLDER = r"llms"
CLEAN_FIELD = "cleaned_keywords"

# ================== LOAD FILES ==================
hybrid_data = load_jsonl(HYBRID_FILE)
hybrid_dict = {p.get("paper_id", p.get("id")): p for p in hybrid_data}
print(f"Loaded Hybrid papers: {len(hybrid_dict)}")

llm_files = [f for f in os.listdir(LLM_FOLDER) if f.endswith(".jsonl")]
llm_data = {}
for f in llm_files:
    path = os.path.join(LLM_FOLDER, f)
    data = load_jsonl(path)
    llm_data[f] = {p.get("paper_id", p.get("id")): p for p in data}
    print(f"{f} loaded: {len(data)} papers")

# ================== COMPUTE METRICS ==================
results = []

for paper_id, hybrid_paper in hybrid_dict.items():
    hybrid_cleaned = parse_keywords(hybrid_paper.get(CLEAN_FIELD, []))
    
    for llm_file, papers in llm_data.items():
        llm_paper = papers.get(paper_id)
        if not llm_paper:
            continue
        llm_cleaned = parse_keywords(llm_paper.get(CLEAN_FIELD, []))
        llm_orig = parse_keywords(llm_paper.get("original_keywords", []))
        
        results.append({
            "Model": llm_file,
            "Jaccard Phrase": round(jaccard_similarity(hybrid_cleaned, llm_cleaned), 3),
            "Jaccard Token": round(token_jaccard(hybrid_cleaned, llm_cleaned), 3),
            "Coverage": round(coverage_ratio(llm_orig, llm_cleaned), 3),
            "Conciseness": round(conciseness_score(llm_orig, llm_cleaned), 3),
            "# Cleaned Keywords": len(llm_cleaned)
        })

# ================== CREATE TABLE ==================
df = pd.DataFrame(results)
summary = df.groupby("Model").mean().reset_index()  # Average metrics per model
summary = summary[["Model", "Jaccard Phrase", "Jaccard Token", "Coverage", "Conciseness", "# Cleaned Keywords"]]

print("\n=== SUMMARY TABLE: Hybrid vs LLMs ===")
print(summary.to_string(index=False))

# Save CSV
summary.to_csv("Hybrid_vs_single_summary.csv", index=False)
print("\nSaved summary CSV as: Hybrid_vs_LLMs_summary.csv")
