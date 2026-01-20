import json
import time
import os
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import random
import re

load_dotenv()

# ============================

# ============================
PAID_MODELS = [
    "openai/gpt-4o-mini"
   "deepseek/deepseek-r1-0528-qwen3-8b",           
    "qwen/qwen-2.5-72b-instruct",                   
    "x-ai/grok-4.1-fast",                           
    "mistralai/mistral-7b-instruct",               
]

MODEL_SHORT_NAMES = [
    "openai-gpt"
   "deepseek-r1-qwen3-8b",
   "qwen-2.5-72b-instruct",
   "grok-4.1-fast",
   "mistral-7b",
]

MODEL_DISPLAY_NAMES = [
    "openai-gpt-4o-mini"
    "DeepSeek-R1 (Qwen3-8B)",
    "Qwen-2.5-72B",
    "Grok 4.1 Fast (xAI)",
    "Mistral-7B"
]

class CleanedKeywords(BaseModel):
    cleaned_keywords: List[str] = Field(description="List of cleaned academic keywords")

# Build chains with structured output (all paid models support json_mode perfectly)
chains = []
structured_flags = []

for i, model_name in enumerate(PAID_MODELS):
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.1,
        api_key=os.getenv("keyword_processing"),
        base_url="https://openrouter.ai/api/v1",
        max_tokens=400,
        timeout=90,
        max_retries=5,
    )
    try:
        structured_llm = llm.with_structured_output(CleanedKeywords, method="json_mode")
        structured_flags.append(True)
    except:
        structured_llm = llm
        structured_flags.append(False)
    
    prompt = ChatPromptTemplate.from_template("""
        You are an academic text-processing assistant.

        Your task: Clean and standardize academic keywords.

        Input keywords:
        {raw_keywords}

        Follow these rules strictly:
        1. Convert all keywords to lowercase.
        2. Remove stopwords(e.g., "analysis", "study", "approach", "method", "paper", "system", "based", etc.).
        3. Remove punctuation and symbols
        4. Lemmatize the following keywords. If lemmatization does not reduce a keyword, apply stemming.
        5. Expand acronyms when possible (e.g., "nlp" → "natural language processing").
        6. Return **only valid JSON**, no explanation.

        Output format (exactly this structure):
        {
        "cleaned_keywords": ["kw1", "kw2", "kw3"]
        }
        """)

    
    chain = prompt | structured_llm
    chains.append(chain)

# ============================
# Paths
# ============================
input_file = r"dblp_1000_random_samples.jsonl"
output_dir = r"llms"
os.makedirs(output_dir, exist_ok=True)
base_output = os.path.join(output_dir, "dblp_cleaned_multi_free")

# ============================
# Process EACH model on ALL 1000 papers
# ============================
for model_idx in range(len(PAID_MODELS)):
    short_name = MODEL_SHORT_NAMES[model_idx]
    display_name = MODEL_DISPLAY_NAMES[model_idx]
    full_model = PAID_MODELS[model_idx]
    chain = chains[model_idx]
    is_structured = structured_flags[model_idx]

    out_path = f"{base_output}.{short_name}.jsonl"
    prog_path = f"{out_path}.progress"
    
    start_line = 0
    if os.path.exists(prog_path):
        with open(prog_path, 'r') as f:
            start_line = int(f.read().strip())
    else:
        print(f"\nStarting {display_name} (1000 papers)")

    file_handle = open(out_path, 'a', encoding='utf-8')

    with open(input_file, 'r', encoding='utf-8') as f_in:
        for _ in range(start_line):
            f_in.readline()

        pbar = tqdm(total=1000, initial=start_line, desc=f"[{display_name}]", unit="paper")

        for line_num, line in enumerate(f_in, start=start_line):
            line = line.strip()
            if not line:
                pbar.update(1)
                continue

            try:
                paper = json.loads(line)
                raw_kw = paper.get("keywords", [])
                raw_text = ", ".join(map(str, raw_kw)) if raw_kw else "none"

                if not raw_kw:
                    cleaned = []
                else:
                    result = chain.invoke({"raw_keywords": raw_text})
                    if is_structured:
                        cleaned = result.cleaned_keywords
                    else:
                        text = getattr(result, 'content', str(result))
                        match = re.search(r'\{.*"cleaned_keywords".*?\}', text, re.DOTALL)
                        cleaned = json.loads(match.group(0)).get("cleaned_keywords", []) if match else []

                row = {
                    "paper_id": paper.get("id", f"paper_{line_num}"),
                    "title": str(paper.get("title", ""))[:200],
                    "year": paper.get("year", ""),
                    "venue": paper.get("venue", ""),
                    "original_keywords": " | ".join(map(str, raw_kw)),
                    "cleaned_keywords": " | ".join(cleaned),
                    "num_original": len(raw_kw),
                    "num_cleaned": len(cleaned),
                    "model_display": display_name,
                    "full_model_name": full_model,
                }

                file_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                file_handle.flush()

                with open(prog_path, 'w') as prog:
                    prog.write(str(line_num + 1))

                pbar.update(1)
                time.sleep(0.8 + random.uniform(0, 0.7))  # Gentle delay (paid models allow faster)

            except Exception as e:
                print(f"\nERROR line {line_num + 1} [{display_name}]: {e}")
                row = {
                    "paper_id": paper.get("id", f"paper_{line_num}"),
                    "title": str(paper.get("title", ""))[:200],
                    "year": paper.get("year", ""),
                    "venue": paper.get("venue", ""),
                    "original_keywords": " | ".join(map(str, raw_kw)),
                    "cleaned_keywords": "",
                    "num_original": len(raw_kw),
                    "num_cleaned": 0,
                    "model_display": display_name,
                    "full_model_name": full_model,
                }
                file_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                time.sleep(5)

        pbar.close()
        file_handle.close()
        if os.path.exists(prog_path):
            os.remove(prog_path)

    print(f"Finished {display_name} → {out_path}")

print("\nALL DONE!")
print("Check folder:")
print(output_dir)