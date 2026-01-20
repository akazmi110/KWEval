import json
import pandas as pd
from typing import List, Dict, Set
from nltk.tokenize import word_tokenize
import nltk
import os

# Ensure NLTK data is available (from your previous code)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file into list of dicts."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Jaccard: |A ∩ B| / |A ∪ B|."""
    if not set_a or not set_b:
        return 0.0
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    return len(intersection) / len(union) if union else 0.0

def token_jaccard(set_a: List[str], set_b: List[str]) -> float:
    """Word-level Jaccard: tokenize phrases, then Jaccard on word sets."""
    tokens_a = set(word_tokenize(' '.join(set_a).lower()))
    tokens_b = set(word_tokenize(' '.join(set_b).lower()))
    return jaccard_similarity(tokens_a, tokens_b)

def coverage_ratio(originals: List[str], cleaned: List[str], threshold_words: int = 1) -> float:
    """% of originals with >= threshold_words overlap in any cleaned keyword."""
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

def conciseness_score(originals: List[str], cleaned: List[str]) -> float:
    """Normalized length: len(cleaned) / max(len(originals), 1). Lower = more concise."""
    return len(cleaned) / max(len(originals), 1)

def parse_keywords(raw_input) -> List[str]:
    """
    Parse original_keywords into a list.
    Accepts either:
    - pipe-separated string
    - list of keywords
    """
    if not raw_input:
        return []
    
    if isinstance(raw_input, str):
        return [kw.strip() for kw in raw_input.split('|') if kw.strip()]
    elif isinstance(raw_input, list):
        return [kw.strip() for kw in raw_input if kw.strip()]
    else:
        return []

# ========================= CONFIGURATION =========================
# Update these paths
NLTK_FILE = r"preprocessing using_nltk\nltk_dblp_cleaned_1000_random_samples_keywords_cleaned.jsonl"
LLM_FILES = {
    "gpt-4o-mini": r"llms\dblp_cleaned_multi_free.gpt-4o-mini.jsonl",
    "deepseek-paid": r"llms\dblp_cleaned_multi_free.deepseek-r1-qwen3-8b.jsonl",
    "qwen-2.5-72b": r"llms\dblp_cleaned_multi_free.qwen-2.5-72b.jsonl",
    "grok-4.1-fast": r"llms\dblp_cleaned_multi_free.grok-4.1-fast.jsonl",
     "mistralai/mistral-7b-instruct": r"llms\dblp_cleaned_multi_free.mistral-7b.jsonl"
}

# Load NLTK data (baseline)
nltk_data = load_jsonl(NLTK_FILE)
nltk_dict = {paper['id']: paper for paper in nltk_data if paper.get('keywords')}

# Load LLM data
llm_data = {}
for model_name, file_path in LLM_FILES.items():
    if os.path.exists(file_path):
        data = load_jsonl(file_path)
        llm_data[model_name] = {paper['paper_id']: paper for paper in data if paper.get('original_keywords')}
    else:
        print(f"Warning: File not found for {model_name}: {file_path}")

# ========================= COMPUTE METRICS =========================
results = []
common_papers = set(nltk_dict.keys())
for d in llm_data.values():
    common_papers = common_papers.intersection(d.keys())  # Match by ID (paper_id == id)

print(f"Number of common papers across all datasets: {len(common_papers)}")

for paper_id in common_papers:
    paper_nltk = nltk_dict[paper_id]
    orig_keywords = parse_keywords(paper_nltk.get('keywords', []))  # Use NLTK's keywords as original
    
    nltk_cleaned = paper_nltk.get('keywords_cleaned', [])
    nltk_metrics = {
        'coverage': coverage_ratio(orig_keywords, nltk_cleaned),
        'conciseness': conciseness_score(orig_keywords, nltk_cleaned),
        'num_original': len(orig_keywords),
        'num_cleaned': len(nltk_cleaned)
    }
    
    for model_name, llm_dict in llm_data.items():
        paper_llm = llm_dict[paper_id]
        llm_cleaned_str = paper_llm.get('cleaned_keywords', '')
        
        # Fixed parsing: Try JSON first, then fallback to pipe-separated string
        llm_cleaned = []
        if llm_cleaned_str and llm_cleaned_str.strip():
            try:
                parsed_json = json.loads(llm_cleaned_str)
                if isinstance(parsed_json, dict) and 'cleaned_keywords' in parsed_json:
                    llm_cleaned = parsed_json['cleaned_keywords']
                elif isinstance(parsed_json, list):
                    llm_cleaned = parsed_json
            except json.JSONDecodeError:
                # Fallback: Treat as pipe-separated string
                llm_cleaned = parse_keywords(llm_cleaned_str)
        
        llm_orig = parse_keywords(paper_llm.get('original_keywords', ''))
        llm_metrics = {
            'coverage': coverage_ratio(llm_orig, llm_cleaned),
            'conciseness': conciseness_score(llm_orig, llm_cleaned),
            'num_original': len(llm_orig),
            'num_cleaned': len(llm_cleaned)
        }
        
        # Pairwise metrics
        jacc_phrase = jaccard_similarity(set(nltk_cleaned), set(llm_cleaned))
        jacc_token = token_jaccard(nltk_cleaned, llm_cleaned)
        
        results.append({
            'paper_id': paper_id,
            'model': model_name,
            'jaccard_phrase': jacc_phrase,
            'jaccard_token': jacc_token,
            'nltk_coverage': nltk_metrics['coverage'],
            'llm_coverage': llm_metrics['coverage'],
            'nltk_conciseness': nltk_metrics['conciseness'],
            'llm_conciseness': llm_metrics['conciseness'],
            'nltk_num_cleaned': nltk_metrics['num_cleaned'],
            'llm_num_cleaned': llm_metrics['num_cleaned']
        })

# ========================= SUMMARY TABLE =========================
if results:
    df = pd.DataFrame(results)
    summary = df.groupby('model').agg({
        'jaccard_phrase': ['mean', 'std'],
        'jaccard_token': ['mean', 'std'],
        'llm_coverage': 'mean',
        'llm_conciseness': 'mean',
        'llm_num_cleaned': 'mean'
    }).round(3)

    print("Summary Metrics (Higher Jaccard/Coverage better; Lower Conciseness better for noise removal)")
    print(summary)

    # Example: Rank by average Jaccard (phrase + token)
    df['avg_jaccard'] = (df['jaccard_phrase'] + df['jaccard_token']) / 2
    best_model = df.groupby('model')['avg_jaccard'].mean().idxmax()
    print(f"\nBest performing model (by avg Jaccard to NLTK): {best_model}")
    print(f"Overall avg Jaccard across all models: {df['avg_jaccard'].mean():.3f}")

    # Save detailed results
    df.to_csv(r"metrics_comparison_single_prompt_nltk.csv", index=False)
    print("\nDetailed results saved to: metrics_comparison.csv")
else:
    print("No common papers found. Check ID matching between NLTK ('id') and LLM ('paper_id') files.")