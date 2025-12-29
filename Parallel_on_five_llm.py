import json
import time
import os
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

load_dotenv()

# ============================
# API KEY
# ============================
api_key = os.getenv("keyword_processing")
if not api_key or len(api_key) < 40:
    raise ValueError("Invalid OpenRouter API key! Create new at: https://openrouter.ai/keys")

# ============================
# MODELS
# ============================
MODELS = [
    {"name": "openai/gpt-4o-mini", "short": "gpt-4o-mini", "display": "GPT-4o-mini", "use_structured": True},
    {"name": "deepseek/deepseek-r1-0528-qwen3-8b", "short": "deepseek-r1-qwen3-8b", "display": "DeepSeek-R1", "use_structured": False},
    {"name": "x-ai/grok-4.1-fast", "short": "grok-4.1-fast", "display": "Grok-4.1-Fast", "use_structured": False},
    {"name": "mistralai/mistral-7b-instruct", "short": "mistral-7b-instruct", "display": "mistral-7b-instructo", "use_structured": False},
    {"name": "qwen/qwen-2.5-72b-instruct", "short": "qwen-2.5-72b-instruct", "display": "qwen-2.5-72b-instruct", "use_structured": False},
]

class CleanedKeywords(BaseModel):
    cleaned_keywords: List[str] = Field(description="List of cleaned academic keywords")

# ============================
# SAFE PARSER (for non-structured models)
# ============================
def safe_parse_output(result) -> List[str]:
    try:
        if hasattr(result, "cleaned_keywords"):
            return result.cleaned_keywords
        if hasattr(result, "content"):
            text = result.content
        else:
            text = str(result)

        # Find JSON block
        json_match = re.search(r"\{.*?\}(?=\s*$|\n|$)", text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            if isinstance(data, dict) and "cleaned_keywords" in data:
                return data["cleaned_keywords"]
            if isinstance(data, list):
                return data
            return list(data.values()) if isinstance(data, dict) else []
        return []
    except:
        return []

# ============================
# BUILD CHAINS (Fixed for Azure JSON requirement)
# ============================
def build_chains(model_name: str, use_structured: bool):
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        max_tokens=500,
        timeout=180,
        max_retries=2,
    )

    # Base instruction that satisfies Azure's "json" requirement
    json_instruction = "You are a precise keyword processor. Always respond in valid JSON format with the key 'cleaned_keywords' containing a list of strings. Never add explanations."

    prompts = {
        "lowercase": ChatPromptTemplate.from_template(
            f"{json_instruction}\n"
            "Task: Convert ONLY these keywords to lowercase. Do nothing else.\n"
            "Input: {raw_keywords}\n"
            "Output ONLY valid JSON: {{\"cleaned_keywords\": [\"apple\", \"banana\"]}}"
        ),
        "no_generics": ChatPromptTemplate.from_template(
            f"{json_instruction}\n"
            "Task: Remove generic terms like 'analysis', 'study', 'approach', 'method', 'framework', 'system'. Keep domain-specific terms. Lowercase everything.\n"
            "Input: {raw_keywords}\n"
            "Output ONLY valid JSON: {{\"cleaned_keywords\": [...]}}"
        ),
        "no_punct": ChatPromptTemplate.from_template(
            f"{json_instruction}\n"
            "Task: Remove punctuation, symbols, slashes, hyphens, parentheses. Keep only letters, numbers, spaces. Then lowercase.\n"
            "Input: {raw_keywords}\n"
            "Output ONLY valid JSON: {{\"cleaned_keywords\": [...]}}"
        ),
        "lemmatize": ChatPromptTemplate.from_template(
            f"{json_instruction}\n"
            "Task: Lemmatize or stem words (e.g., networks → network, studies → study). Lowercase.\n"
            "Input: {raw_keywords}\n"
            "Output ONLY valid JSON: {{\"cleaned_keywords\": [...]}}"
        ),
        "expand": ChatPromptTemplate.from_template(
            f"{json_instruction}\n"
            "Task: Expand common academic acronyms (nlp → natural language processing, ml → machine learning, cnn → convolutional neural network, etc.). Lowercase.\n"
            "Input: {raw_keywords}\n"
            "Output ONLY valid JSON: {{\"cleaned_keywords\": [...]}}"
        ),
        "final": ChatPromptTemplate.from_template(
            f"{json_instruction}\n"
            "Refine into the BEST final list of maximum 15 technical, specific keywords.\n"
            "Remove duplicates, keep most important domain terms.\n"
            "Inputs:\n"
            "- Lowercased: {lowercase}\n"
            "- No generics: {no_generics}\n"
            "- No punctuation: {no_punct}\n"
            "- Lemmatized: {lemmatized}\n"
            "- Expanded: {expanded}\n"
            "Output ONLY valid JSON: {{\"cleaned_keywords\": [\"deep learning\", \"transformer\", ...]}}"
        ),
    }

    chains = {}
    parser = JsonOutputParser(pydantic_object=CleanedKeywords)

    for task, prompt in prompts.items():
        if use_structured:
            # This now works because prompt contains "JSON"
            chains[task] = prompt | llm.with_structured_output(CleanedKeywords, method="json_mode")
        else:
            chains[task] = prompt | llm
            chains[f"{task}_parser"] = parser

    return chains

# ============================
# PROCESS ONE PAPER
# ============================
def process_paper(args: Tuple[Dict, int, str, bool]) -> Tuple[int, Dict, List[str]]:
    chains, line_num, line, use_structured = args
    try:
        paper = json.loads(line)
        raw_kw = paper.get("keywords", [])
        if not raw_kw:
            return line_num, paper, []

        raw_text = ", ".join(map(str, raw_kw))

        # Run 5 intermediate tasks in parallel
        tasks = ["lowercase", "no_generics", "no_punct", "lemmatize", "expand"]
        with ThreadPoolExecutor(max_workers=5) as exec:
            futures = {exec.submit(chains[task].invoke, {"raw_keywords": raw_text}): task for task in tasks}
            intermediate_results = {}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    res = future.result()
                    if use_structured:
                        intermediate_results[task] = res.cleaned_keywords
                    else:
                        parser = chains[f"{task}_parser"]
                        parsed = parser.parse(res.content) if hasattr(res, "content") else safe_parse_output(res)
                        intermediate_results[task] = parsed.get("cleaned_keywords", []) if isinstance(parsed, dict) else parsed
                except:
                    intermediate_results[task] = []

        # Final refinement
        final_input = {
            "lowercase": " | ".join(intermediate_results.get("lowercase", [])),
            "no_generics": " | ".join(intermediate_results.get("no_generics", [])),
            "no_punct": " | ".join(intermediate_results.get("no_punct", [])),
            "lemmatized": " | ".join(intermediate_results.get("lemmatize", [])),
            "expanded": " | ".join(intermediate_results.get("expand", [])),
        }

        final_res = chains["final"].invoke(final_input)
        if use_structured:
            final_cleaned = final_res.cleaned_keywords
        else:
            final_cleaned = safe_parse_output(final_res)
            if isinstance(final_cleaned, dict):
                final_cleaned = final_cleaned.get("cleaned_keywords", [])

        # Dedupe & limit
        seen = set()
        unique = []
        for kw in final_cleaned:
            k = kw.strip().lower()
            if k and k not in seen and len(unique) < 20:
                unique.append(kw.strip())
                seen.add(k)
        final_cleaned = unique[:15]

        return line_num, paper, final_cleaned

    except Exception as e:
        print(f"ERROR line {line_num}: {e}")
        return line_num, {}, []

# ============================
# MAIN LOOP
# ============================
input_file = r"dblp_cleaned_fast_100_random_samples.jsonl"
output_dir = r"hybrid_llms"
os.makedirs(output_dir, exist_ok=True)

for model_info in MODELS:
    model_name = model_info["name"]
    short_name = model_info["short"]
    display_name = model_info["display"]
    use_structured = model_info["use_structured"]

    print(f"\n{'='*20} Starting {display_name} ({'Structured' if use_structured else 'Parser Mode'}) {'='*20}")

    chains = build_chains(model_name, use_structured)
    out_path = os.path.join(output_dir, f"parallel_cleaned.{short_name}.jsonl")
    prog_path = f"{out_path}.progress"

    start_line = 0
    if os.path.exists(prog_path):
        with open(prog_path, 'r') as f:
            start_line = int(f.read().strip())
        print(f"Resuming from line {start_line + 1}")

    with open(out_path, 'a', encoding='utf-8') as file_handle:
        pbar = tqdm(total=100, initial=start_line, desc=display_name, unit="paper")

        batch_size = 3 if use_structured else 1

        with open(input_file, 'r', encoding='utf-8') as f:
            for _ in range(start_line):
                f.readline()

            while True:
                batch = []
                for _ in range(batch_size):
                    line = f.readline()
                    if not line:
                        break
                    batch.append(line.strip())
                if not batch:
                    break

                args_list = [(chains, start_line + i, line, use_structured) for i, line in enumerate(batch)]

                with ThreadPoolExecutor(max_workers=len(batch)) as exec:
                    for future in as_completed([exec.submit(process_paper, arg) for arg in args_list]):
                        line_num, paper, cleaned = future.result()
                        if not paper:
                            continue

                        raw_kw = paper.get("keywords", [])
                        row = {
                            "paper_id": paper.get("id", f"paper_{line_num}"),
                            "title": str(paper.get("title", ""))[:200],
                            "year": paper.get("year", ""),
                            "venue": paper.get("venue", ""),
                            "original_keywords": " | ".join(map(str, raw_kw)),
                            "cleaned_keywords": " | ".join(cleaned),
                            "num_original": len(raw_kw),
                            "num_cleaned": len(cleaned),
                            "model": display_name,
                            "full_model_name": model_name,
                        }
                        file_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        file_handle.flush()

                        with open(prog_path, 'w') as prog:
                            prog.write(str(line_num + 1))

                        pbar.update(1)
                        time.sleep(0.7 if use_structured else 1.6)  # Rate limit

        pbar.close()

    if os.path.exists(prog_path):
        os.remove(prog_path)

    print(f"Finished {display_name} → {out_path}")

print("\nALL 4 MODELS COMPLETED SUCCESSFULLY!")
print("Output files:", output_dir)