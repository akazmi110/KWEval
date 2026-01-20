import json
import pandas as pd
import os

# ========================= CONFIGURATION =========================
# Paths to the two CSV files generated from previous scripts
SINGLE_CSV = r"metrics_comparison_single_prompt_nltk.csv"
PARALLEL_CSV = r"metrics_nltk_parallel_comparison.csv"

# Base model names for matching (strip suffixes to align single vs parallel)
BASE_MODELS = {
    'gpt-4o-mini': 'gpt-4o-mini',
    'deepseek-paid': 'deepseek-paid',
    'qwen-2.5-72b': 'qwen-2.5-72b',
    'grok-4.1-fast': 'grok-4.1-fast',
    'mistral-7b-instruct': 'mistralai/mistral-7b-instruct' 
}

# ========================= LOAD DATA =========================
if not os.path.exists(SINGLE_CSV):
    print(f"Error: Single prompt CSV not found: {SINGLE_CSV}")
    exit(1)
if not os.path.exists(PARALLEL_CSV):
    print(f"Error: Parallel prompt CSV not found: {PARALLEL_CSV}")
    exit(1)

df_single = pd.read_csv(SINGLE_CSV)
df_parallel = pd.read_csv(PARALLEL_CSV)

print(f"Loaded single prompt: {len(df_single)} rows, {len(df_single['model'].unique())} models")
print(f"Loaded parallel prompt: {len(df_parallel)} rows, {len(df_parallel['model'].unique())} models")

# Normalize model names for comparison (strip '-parallel' suffix)
df_single['base_model'] = df_single['model'].str.replace('-parallel', '', regex=False)
df_parallel['base_model'] = df_parallel['model'].str.replace('-parallel', '', regex=False)

# ========================= AGGREGATE METRICS =========================
def aggregate_metrics(df, group_col='base_model'):
    """Compute mean/std for key metrics, grouped by base_model."""
    agg = df.groupby(group_col).agg({
        'jaccard_phrase': ['mean', 'std'],
        'jaccard_token': ['mean', 'std'],
        'llm_coverage': 'mean',
        'llm_conciseness': 'mean',
        'llm_num_cleaned': 'mean'
    }).round(3)
    agg.columns = ['_'.join(col).strip() if col[1] else col[0] for col in agg.columns.values]
    return agg.reset_index()

single_agg = aggregate_metrics(df_single, 'base_model')
parallel_agg = aggregate_metrics(df_parallel, 'base_model')

# Merge for side-by-side comparison (focus on common base_models)
common_bases = set(single_agg['base_model']).intersection(set(parallel_agg['base_model']))
print(f"\nCommon base models for comparison: {sorted(common_bases)}")

# Filter and merge
single_comp = single_agg[single_agg['base_model'].isin(common_bases)].set_index('base_model')
parallel_comp = parallel_agg[parallel_agg['base_model'].isin(common_bases)].set_index('base_model')

comparison = pd.concat([single_comp.add_suffix('_single'), parallel_comp.add_suffix('_parallel')], axis=1)

# Compute differences (parallel - single)
diff_cols = [col for col in comparison.columns if '_parallel' in col]
for col in diff_cols:
    base_col = col.replace('_parallel', '_single')
    if base_col in comparison.columns:
        comparison[f'{col.replace("_parallel", "_diff")}'] = comparison[col] - comparison[base_col]

print("\n=== COMPARISON TABLE: Single vs Parallel Prompts ===")
print("(Higher Jaccard/Coverage better; Lower Conciseness better)")
print(comparison)

# Overall averages
overall_single = df_single[['jaccard_phrase', 'jaccard_token', 'llm_coverage', 'llm_conciseness', 'llm_num_cleaned']].mean().round(3)
overall_parallel = df_parallel[['jaccard_phrase', 'jaccard_token', 'llm_coverage', 'llm_conciseness', 'llm_num_cleaned']].mean().round(3)
overall_df = pd.DataFrame({
    'Metric': overall_single.index,
    'Single Mean': overall_single.values,
    'Parallel Mean': overall_parallel.values,
    'Diff (Parallel - Single)': (overall_parallel - overall_single).round(3).values
})
print("\n=== OVERALL AVERAGES ===")
print(overall_df.to_string(index=False))

# Avg Jaccard improvement
single_avg_jacc = (df_single['jaccard_phrase'] + df_single['jaccard_token']) / 2
parallel_avg_jacc = (df_parallel['jaccard_phrase'] + df_parallel['jaccard_token']) / 2
print(f"\nOverall Avg Jaccard: Single = {single_avg_jacc.mean():.3f}, Parallel = {parallel_avg_jacc.mean():.3f}, Improvement = {parallel_avg_jacc.mean() - single_avg_jacc.mean():+.3f}")

# Save comparison
comparison.to_csv(r"single_vs_parallel_comparison.csv")
print(f"\nDetailed comparison saved to: single_vs_parallel_comparison.csv")

# Quick Insights
if parallel_avg_jacc.mean() > single_avg_jacc.mean():
    print("\nInsight: Parallel prompts improve agreement with NLTK (higher Jaccard).")
else:
    print("\nInsight: Single prompts perform similarly or better—check specifics.")
if overall_parallel['llm_conciseness'] < overall_single['llm_conciseness']:
    print("Insight: Parallel prompts are more concise (better noise reduction).")
print("Recommendation: Review per-model diffs; parallel may excel in consistency for complex rules.")