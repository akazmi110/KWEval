import json
import os
import glob
import pandas as pd

###############################################################################
# Helper functions
###############################################################################

def tokenize_keywords(text):
    if isinstance(text, list):
        return [t.lower().strip() for t in text]
    if not text:
        return []
    return [t.lower().strip() for t in text.split("|")]

def jaccard(list1, list2):
    set1, set2 = set(list1), set(list2)
    if not set1 and not set2:
        return 1
    if not set1 or not set2:
        return 0
    return len(set1 & set2) / len(set1 | set2)

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

###############################################################################
# Paths to your LLM & NLTK files
###############################################################################

paths = {
    "stopwords_llm": r"stopword_removal_using_llms/*.jsonl",
    "stopwords_nltk": r"preprocessing using_nltk/nltk_stopword_cleaned.jsonl",

    "acronyms_llm": r"acrnoyms_using_llms/*.jsonl",
    "acronyms_nltk": r"preprocessing using_nltk/dblp_keywords_acronyms_expanded.jsonl",

    "lemma_llm": r"lematization using llms/*.jsonl",
    "lemma_nltk": r"preprocessing using_nltk/dblp_keywords_lemmatized_stemmed.jsonl",
}

###############################################################################
# Master list for metrics
###############################################################################

metrics_rows = []

###############################################################################
# PROCESS STOPWORD REMOVAL
###############################################################################

print("Processing stopword removal...")

nltk_stop = {
    d["paper_id"]: tokenize_keywords(d["cleaned_keywords"])
    for d in load_jsonl(paths["stopwords_nltk"])
}

for file in glob.glob(paths["stopwords_llm"]):
    llm_results = load_jsonl(file)
    for item in llm_results:
        pid = item["paper_id"]
        llm_clean = tokenize_keywords(item["cleaned_keywords"])
        nltk_clean = nltk_stop.get(pid, [])

        metrics_rows.append({
            "task": "stopwords",
            "paper_id": pid,
            "model": item["model_display"],
            "jaccard": jaccard(llm_clean, nltk_clean),
            "coverage_llm": len(llm_clean),
            "coverage_nltk": len(nltk_clean),
            "conciseness_llm": len(llm_clean),
            "mismatch": len(set(llm_clean) ^ set(nltk_clean))
        })

###############################################################################
# PROCESS ACRONYM EXPANSION
###############################################################################

print("Processing acronym expansion...")

nltk_acr = {
    d["id"]: tokenize_keywords(" | ".join(d["expanded_acronyms_keywords"]))
    for d in load_jsonl(paths["acronyms_nltk"])
}

for file in glob.glob(paths["acronyms_llm"]):
    llm_results = load_jsonl(file)
    for item in llm_results:
        pid = item["paper_id"]
        llm_clean = tokenize_keywords(item["cleaned_keywords"])
        nltk_clean = nltk_acr.get(pid, [])

        metrics_rows.append({
            "task": "acronyms",
            "paper_id": pid,
            "model": item["model_display"],
            "jaccard": jaccard(llm_clean, nltk_clean),
            "coverage_llm": len(llm_clean),
            "coverage_nltk": len(nltk_clean),
            "conciseness_llm": len(llm_clean),
            "mismatch": len(set(llm_clean) ^ set(nltk_clean))
        })

###############################################################################
# PROCESS LEMMATIZATION
###############################################################################

print("Processing lemmatization...")

nltk_lemma = {
    d["id"]: tokenize_keywords(" | ".join(d["lemmatized_stemmed_keywords"]))
    for d in load_jsonl(paths["lemma_nltk"])
}

for file in glob.glob(paths["lemma_llm"]):
    llm_results = load_jsonl(file)
    for item in llm_results:
        pid = item["paper_id"]
        llm_clean = tokenize_keywords(item["cleaned_keywords"])
        nltk_clean = nltk_lemma.get(pid, [])

        metrics_rows.append({
            "task": "lemmatization",
            "paper_id": pid,
            "model": item["model_display"],
            "jaccard": jaccard(llm_clean, nltk_clean),
            "coverage_llm": len(llm_clean),
            "coverage_nltk": len(nltk_clean),
            "conciseness_llm": len(llm_clean),
            "mismatch": len(set(llm_clean) ^ set(nltk_clean))
        })

###############################################################################
# Save detailed per-paper metrics
###############################################################################

df = pd.DataFrame(metrics_rows)
df.to_csv("metrics_results.csv", index=False)
print("Saved: metrics_results.csv")

###############################################################################
# Create summary table (MEAN + STD)
###############################################################################

summary = (
    df.groupby(["task", "model"])
      .agg(
          jaccard_mean=("jaccard", "mean"),
          jaccard_std=("jaccard", "std"),

          coverage_llm_mean=("coverage_llm", "mean"),
          coverage_llm_std=("coverage_llm", "std"),

          coverage_nltk_mean=("coverage_nltk", "mean"),
          coverage_nltk_std=("coverage_nltk", "std"),

          conciseness_llm_mean=("conciseness_llm", "mean"),
          conciseness_llm_std=("conciseness_llm", "std"),

          mismatch_mean=("mismatch", "mean"),
          mismatch_std=("mismatch", "std"),
      )
      .reset_index()
)

# Round for readability
metric_cols = [c for c in summary.columns if c not in ["task", "model"]]
summary[metric_cols] = summary[metric_cols].round(3)

summary.to_csv("Tasked_based_metrics_summary_table.csv", index=False)
print("Saved: Tasked_based_metrics_summary_table.csv")

###############################################################################
# Print summary tables per task
###############################################################################

for task in summary["task"].unique():
    print(f"\n=== {task.upper()} ===")
    print(summary[summary["task"] == task].to_string(index=False))

###############################################################################
# Choose BEST model per task (using MEAN values)
###############################################################################

best_models = []

for task in summary["task"].unique():
    task_df = summary[summary["task"] == task].copy()

    # scoring: high mean jaccard + low mean mismatch
    task_df["score"] = task_df["jaccard_mean"] - 0.05 * task_df["mismatch_mean"]

    best_row = task_df.sort_values("score", ascending=False).iloc[0]

    best_models.append({
        "task": task,
        "best_model": best_row["model"],
        "avg_jaccard": best_row["jaccard_mean"],
        "std_jaccard": best_row["jaccard_std"],
        "avg_mismatch": best_row["mismatch_mean"],
        "std_mismatch": best_row["mismatch_std"],
    })

best_df = pd.DataFrame(best_models)
best_df.to_csv("best_models_per_task.csv", index=False)

print("\nSaved: best_models_per_task.csv")
print("\n=== BEST MODELS ===")
print(best_df.to_string(index=False))
