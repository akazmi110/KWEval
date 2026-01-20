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
    Parse input into a list.
    Accepts either:
    - pipe-separated string
    - list of keywords
    - JSON string like {"cleaned_keywords": [...]}
    """
    if not raw_input:
        return []
    
    # If it's a JSON string, parse it first
    if isinstance(raw_input, str) and raw_input.strip().startswith('{'):
        try:
            parsed = json.loads(raw_input)
            if isinstance(parsed, dict) and 'cleaned_keywords' in parsed:
                raw_input = parsed['cleaned_keywords']
            elif isinstance(parsed, list):
                raw_input = parsed
        except json.JSONDecodeError:
            pass  # Fall back to pipe-split
    
    if isinstance(raw_input, str):
        return [kw.strip() for kw in raw_input.split('|') if kw.strip()]
    elif isinstance(raw_input, list):
        return [str(kw).strip() for kw in raw_input if kw.strip()]
    else:
        return []

# ========================= CONFIGURATION =========================
# Update these paths for your parallel prompt files (use full paths if relative fails)
NLTK_FILE = r"preprocessing using_nltk\nltk_dblp_cleaned_1000_random_samples_keywords_cleaned.jsonl"

PARALLEL_LLM_FILES = {
    "gpt-4o-mini-parallel": r"llms_using_parallel_prompt\parallel_cleaned.gpt-4o-mini.jsonl",
    "deepseek-paid-parallel": r"llms_using_parallel_prompt\parallel_cleaned.deepseek-r1-qwen3-8b.jsonl",
    "qwen-2.5-72b-parallel": r"llms_using_parallel_prompt\parallel_cleaned.qwen-2.5-72b-instruct.jsonl",  # Fixed: removed extra 'l'
    "grok-4.1-fast-parallel": r"llms_using_parallel_prompt\parallel_cleaned.grok-4.1-fast.jsonl",
    "mistralai/mistral-7b-instruct":r"llms_using_parallel_prompt\parallel_cleaned.mistral-7b-instruct.jsonl"
}

# FIXED: For parallel outputs, use 'cleaned_keywords' (as in your sample)
FINAL_CLEANED_FIELD = 'cleaned_keywords'

# ========================= LOAD DATA WITH DEBUG =========================
# Load NLTK data (baseline)
nltk_data = load_jsonl(NLTK_FILE)
nltk_dict = {paper.get('id', paper.get('paper_id', 'unknown')): paper for paper in nltk_data if paper.get('keywords')}
print(f"NLTK loaded: {len(nltk_data)} papers")
print("Sample NLTK IDs:", list(nltk_dict.keys())[:3])

# Load Parallel LLM data
llm_data = {}
for model_name, file_path in PARALLEL_LLM_FILES.items():
    full_path = os.path.abspath(file_path)  # Convert relative to absolute
    if os.path.exists(full_path):
        data = load_jsonl(full_path)
        llm_dict = {paper.get('paper_id', paper.get('id', 'unknown')): paper for paper in data if paper.get('original_keywords', paper.get('keywords'))}
        llm_data[model_name] = llm_dict
        print(f"{model_name} loaded: {len(data)} papers (from {full_path})")
        print(f"  Sample {model_name} IDs:", list(llm_dict.keys())[:3])
        # Debug: Check a sample paper's cleaned field
        if data:
            sample_paper = data[0]
            sample_cleaned = sample_paper.get(FINAL_CLEANED_FIELD, 'MISSING')
            print(f"  Sample cleaned for {model_name}: {sample_cleaned[:1000]}...")  # Truncate for print
    else:
        print(f"Warning: File not found for {model_name}: {full_path}")

# Debug: Save samples to JSON for easy inspection
debug_samples = {
    'nltk_samples': list(nltk_dict.items())[:2],
    'llm_samples': {model: list(d.items())[:2] for model, d in llm_data.items()}
}
debug_file = r"nltk_parallel_comparision.json"
with open(debug_file, 'w') as f:
    json.dump(debug_samples, f, indent=2, default=str)
print(f"\nSample data saved to: {debug_file} (open to compare IDs/keys)")

# ========================= COMPUTE COMMON PAPERS =========================
common_papers = set(nltk_dict.keys())
for model_name, d in llm_data.items():
    common_papers = common_papers.intersection(d.keys())
    print(f"After intersecting with {model_name}: {len(common_papers)} common")

print(f"\nFinal number of common papers: {len(common_papers)}")

if len(common_papers) == 0:
    print("No matches! Tips:")
    print("- Check debug file: Compare NLTK 'id' vs. LLM 'paper_id' (e.g., are they the same strings?).")
    print("- If IDs match but no intersection, files may have different papers—re-run LLMs on NLTK samples.")
    print("- Test: Manually add a known ID from sample (e.g., '53e9a0e6b7602d97029cded1') to see.")
else:
    # ========================= COMPUTE METRICS =========================
    results = []

    for paper_id in common_papers:
        paper_nltk = nltk_dict[paper_id]
        orig_keywords = parse_keywords(paper_nltk.get('keywords', []))
        
        nltk_cleaned = paper_nltk.get('keywords_cleaned', [])
        nltk_metrics = {
            'coverage': coverage_ratio(orig_keywords, nltk_cleaned),
            'conciseness': conciseness_score(orig_keywords, nltk_cleaned),
            'num_original': len(orig_keywords),
            'num_cleaned': len(nltk_cleaned)
        }
        
        for model_name, llm_dict in llm_data.items():
            paper_llm = llm_dict[paper_id]
            llm_cleaned_str = paper_llm.get(FINAL_CLEANED_FIELD, '')
            
            # Parse the final output (handles JSON, pipe, or list)
            llm_cleaned = parse_keywords(llm_cleaned_str)
            
            llm_orig = parse_keywords(paper_llm.get('original_keywords', ''))
            llm_metrics = {
                'coverage': coverage_ratio(llm_orig, llm_cleaned),
                'conciseness': conciseness_score(llm_orig, llm_cleaned),
                'num_original': len(llm_orig),
                'num_cleaned': len(llm_cleaned)
            }
            
            # Pairwise metrics (NLTK vs LLM final)
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
    df = pd.DataFrame(results)
    summary = df.groupby('model').agg({
        'jaccard_phrase': ['mean', 'std'],
        'jaccard_token': ['mean', 'std'],
        'llm_coverage': 'mean',
        'llm_conciseness': 'mean',
        'llm_num_cleaned': 'mean'
    }).round(3)

    print("\nSummary Metrics for Parallel Prompts (Higher Jaccard/Coverage better; Lower Conciseness better for noise removal)")
    print(summary)

    # Rank by average Jaccard
    df['avg_jaccard'] = (df['jaccard_phrase'] + df['jaccard_token']) / 2
    best_model = df.groupby('model')['avg_jaccard'].mean().idxmax()
    print(f"\nBest performing model (by avg Jaccard to NLTK): {best_model}")
    print(f"Overall avg Jaccard across all models: {df['avg_jaccard'].mean():.3f}")

    # Save detailed results
    output_csv = r"metrics_nltk_parallel_comparison.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nDetailed results saved to: {output_csv}")