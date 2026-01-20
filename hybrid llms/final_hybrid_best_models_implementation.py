import json
import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# =======================
# Load API key
# =======================
load_dotenv()
api_key = os.getenv("keyword_processing")
if not api_key:
    raise ValueError("API key not found in env file under name 'keyword_processing'.")
base_url = "https://openrouter.ai/api/v1"

# =======================
# Initialize LLMs
# =======================
def create_llm(model_name):
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.1,
        max_tokens=200
    )

lowercase_llm = create_llm("openai/gpt-4o-mini")
acronyms_llm = create_llm("qwen/qwen-2.5-72b-instruct")
lemmatization_llm = create_llm("x-ai/grok-4.1-fast")
stopwords_llm = lemmatization_llm  # reuse

# =======================
# Prompt templates
# =======================
lowercase_prompt = ChatPromptTemplate.from_template(
    "Convert the following keywords to lowercase only. Preserve structure and separate by ' | '. Keywords: {keywords}"
)

acronyms_prompt = ChatPromptTemplate.from_template(
    """Expand acronyms in the keywords to their full form where meaningful, keeping the original if unclear. 
Output only the processed keywords separated by ' | '. Do not change other words. Keywords: {keywords}"""
)

lemmatization_prompt = ChatPromptTemplate.from_template(
    """Lemmatize the keywords (reduce to base form, e.g., 'running' -> 'run'). Preserve meaning and structure.
If lemmatization does not reduce a keyword, apply stemming.
Output only the final processed keywords separated by ' | '. Keywords: {keywords}"""
)

stopwords_prompt = ChatPromptTemplate.from_template(
    """Remove common stopwords (e.g., the, a, an, in, on) from the keywords, but keep domain-specific terms. 
Preserve structure. Output only the processed keywords separated by ' | '. Keywords: {keywords}"""
)

hybrid_prompt = ChatPromptTemplate.from_template(
    """Combine these processed keyword versions into one final cleaned set:
- Lowercased: {lowercased}
- No stopwords: {no_stopwords}
- Acronyms expanded: {acronyms_expanded}
- Lemmatized: {lemmatized}

Output only the final keywords separated by ' | '. Ensure high overlap with originals while cleaning."""
)

parser = StrOutputParser()

# =======================
# LLM Chains
# =======================
def make_chain(prompt, llm):
    return prompt | llm | parser

lowercase_chain = make_chain(lowercase_prompt, lowercase_llm)
acronyms_chain = make_chain(acronyms_prompt, acronyms_llm)
lemmatization_chain = make_chain(lemmatization_prompt, lemmatization_llm)
stopwords_chain = make_chain(stopwords_prompt, stopwords_llm)
hybrid_llm = lowercase_llm  # final merge chain

parallel_chain = RunnableParallel({
    "no_stopwords": stopwords_chain,
    "acronyms_expanded": acronyms_chain,
    "lemmatized": lemmatization_chain
})

# =======================
# Retry helper
# =======================
def invoke_with_retry(chain, input_dict, max_retries=3, delay=3):
    for attempt in range(1, max_retries + 1):
        try:
            return chain.invoke(input_dict)
        except Exception as e:
            print(f"Warning: Attempt {attempt} failed with error: {e}")
            if attempt == max_retries:
                print("Max retries reached. Returning empty string.")
                return ""
            else:
                time.sleep(delay * attempt)  # exponential backoff

# =======================
# Processing function
# =======================
def process_keywords(orig_keywords: str) -> str:
    # Step 1: Lowercase
    lowercased = invoke_with_retry(lowercase_chain, {"keywords": orig_keywords})

    # Step 2: Parallel processing
    parallel_out = invoke_with_retry(parallel_chain, {"keywords": lowercased})

    # Step 3: Hybrid merge
    hybrid_chain = hybrid_prompt | hybrid_llm | parser
    final = invoke_with_retry(hybrid_chain, {
        "lowercased": lowercased,
        "no_stopwords": parallel_out.get("no_stopwords", ""),
        "acronyms_expanded": parallel_out.get("acronyms_expanded", ""),
        "lemmatized": parallel_out.get("lemmatized", "")
    })

    if final.startswith('"') and final.endswith('"'):
        final = final[1:-1]

    return final

# =======================
# Input/output files
# =======================
input_file = "dblp_1000_random_samples.jsonl"
output_file = "dblp_keywords_hybrid_parallel_retry.jsonl"
os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

# =======================
# Resume logic
# =======================
processed_ids = set()
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
                processed_ids.add(row.get("paper_id"))
            except:
                continue

# =======================
# Main processing loop
# =======================
line_num = 0
with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "a", encoding="utf-8") as outfile:  # append mode

    for line in infile:
        if not line.strip():
            continue

        line_num += 1
        try:
            paper = json.loads(line.strip())
            paper_id = paper.get("id", f"paper_{line_num}")

            if paper_id in processed_ids:
                print(f"Skipping already processed paper {line_num} ({paper_id})")
                continue

            raw_kw = paper.get("keywords", [])
            if not raw_kw:
                cleaned = []
                orig_keywords_str = ""
            else:
                orig_keywords_str = " | ".join(map(str, raw_kw))
                final_keywords_str = process_keywords(orig_keywords_str)
                cleaned = [kw.strip() for kw in final_keywords_str.split(" | ") if kw.strip()]

            row = {
                "paper_id": paper_id,
                "title": str(paper.get("title", ""))[:200],
                "year": paper.get("year", ""),
                "venue": paper.get("venue", ""),
                "original_keywords": orig_keywords_str,
                "cleaned_keywords": " | ".join(cleaned),
                "num_original": len(raw_kw),
                "num_cleaned": len(cleaned),
                "model_display": "Hybrid Parallel LangChain with Retry",
                "full_model_name": "GPT-4o-mini (lowercase & hybrid) + Qwen 2.5 72B (acronyms) + Grok 4.1 Fast (lemmatization & stopwords)"
            }

            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")
            outfile.flush()
            print(f"Processed paper {line_num}: {len(raw_kw)} → {len(cleaned)} keywords")

            time.sleep(5)  # rate-limiting

        except json.JSONDecodeError:
            print(f"Skipping invalid JSON at line {line_num}")
            continue
        except Exception as e:
            print(f"Error processing paper {line_num}: {e}")
            continue

print(f"\nCompleted! Output saved to {output_file}")
