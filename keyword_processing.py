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
# MODELS
# ============================
PAID_MODELS = [
    "openai/gpt-4o-mini"
]

MODEL_SHORT_NAMES = [
    "openai-gpt"
]

MODEL_DISPLAY_NAMES = [
    "openai-gpt-4o-mini"
]

class CleanedKeywords(BaseModel):
    cleaned_keywords: List[str] = Field(description="List of cleaned academic keywords")


# ============================
# PROMPT (SYSTEM + USER)
# ============================
PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    (
    "system",
    """
You are an academic keyword normalization assistant.

Your task is to clean, normalize, and standardize academic keywords from research papers in the computer science domain.

You must preserve the scientific meaning while removing noise, variants, and redundancies.

---
🔧 PROCESSING RULES

1. **Lowercasing**
   - Convert all keywords to lowercase (except known acronyms like "CNN", "AI", etc.).

2. **Trim and Normalize Whitespace/Punctuation**
   - Remove leading/trailing spaces.
   - Replace hyphens, underscores, and extra spaces with a single space.
   - Fix CamelCase or PascalCase (e.g., "DeepLearning" → "deep learning").

3. **Prefix and Suffix Trimming**
   - Remove generic leading/trailing words like: 
     "a", "the", "an", "study", "analysis", "method", "approach", 
     "paper", "system", "framework", "application", "model", "based".
   - E.g., "a study of blockchain-based systems" → "blockchain"

4. **Lemmatization or Stemming**
   - Lemmatize each token. If lemmatization fails, apply light stemming.
   - E.g., "systems" → "system", "approaches" → "approach"

5. **Stopword Removal**
   - Remove standalone stopwords or meaningless glue words:
     "and", "for", "of", "the", "in", "to", "with", "on", "by", "from", "via", etc.

6. **Invalid Keyword Filtering**
   - Remove any keyword that is:
     - Pure digits or standalone years (e.g., "2020")
     - Contains only symbols (e.g., "+", "~", "/")
     - Number ranges (e.g., "0–100", "1 to 300")
     - Units without context (e.g., "5 km", "10 ghz")
     - Broken fragments like "0 k", "3a", "2x", "z1"
     - Too short (length < 2) unless it's a valid acronym
     - Non-English or gibberish text
     - URLs or domains (e.g., "doi.org/...", "https://...")

7. **Acronym Expansion**
   - Expand known computer science acronyms:
     - "nlp" → "natural language processing"
     - "cnn" → "convolutional neural network"
     - "ai" → "artificial intelligence"
     - "ml" → "machine learning"
     - "rl" → "reinforcement learning"
     - "iot" → "internet of things"
   - Only expand if the acronym is **commonly known** and **unambiguous**
   - If both acronym and full form are present, keep only the full form

8. **Preserve Valid Numeric Terms**
   - Keep phrases where numbers are integral (e.g., "5g", "3d printing", "web 3.0")

9. **De-duplication**
   - After all cleaning, remove duplicate entries.

10. **Final Checks**
    - Ensure all keywords are clean, readable, semantically meaningful phrases.


---
  DEVELOPER INSTRUCTIONS

- The model **must return a valid JSON object** with a single key:
  `cleaned_keywords` (type: list of strings).

- Do not include:
  - Any Markdown formatting (e.g., no ```json).
  - Commentary, explanations, or extra headings.
  - Nulls, metadata, or extra fields.
  - Text outside the JSON structure.

- Ensure UTF-8 strings and JSON compliance.

  Output format (strict):

{
  "cleaned_keywords": ["kw1", "kw2", "kw3"]
}
"""
)



# ============================
# RETRY LOGIC FOR LLM CALL
# ============================
def call_llm_with_retry(chain, data, retries=5, delay=2):
    """Retry LLM request on failures such as network disconnect, rate limit, etc."""
    for attempt in range(1, retries + 1):
        try:
            return chain.invoke(data)
        except Exception as e:
            print(f"[LLM ERROR] Attempt {attempt}/{retries}: {e}")
            if attempt == retries:
                raise e
            time.sleep(delay + random.uniform(0, 1))


# ============================
# PATHS
# ============================
input_file = r"C:\Users\PMLS\Downloads\graphs using keyword processing\dblp_cyber_security_regex.jsonl"
output_dir = r"keyword_processing"
os.makedirs(output_dir, exist_ok=True)
base_output = os.path.join(output_dir, "dblp_cleaned_multi_free")


# ============================
# MAIN LOOP
# ============================
for model_idx in range(len(PAID_MODELS)):
    full_model = PAID_MODELS[model_idx]
    short_name = MODEL_SHORT_NAMES[model_idx]
    display_name = MODEL_DISPLAY_NAMES[model_idx]

    print(f"\nLoading model: {display_name}")

    llm = ChatOpenAI(
        model=full_model,
        temperature=0,
        api_key=os.getenv("keyword_processing"),
        base_url="https://openrouter.ai/api/v1",
        max_tokens=400,
        timeout=90,
        max_retries=5,
    )

    # Build structured chain
    try:
        structured_llm = llm.with_structured_output(CleanedKeywords, method="json_mode")
        is_structured = True
    except:
        structured_llm = llm
        is_structured = False

    chain = PROMPT_TEMPLATE | structured_llm

    out_path = f"{base_output}.{short_name}.jsonl"
    prog_path = f"{out_path}.progress"

    # ============================
    # Determine resume position
    # ============================
    start_line = 0
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            start_line = sum(1 for _ in f)

    print(f"Resuming from line {start_line}")

    f_out = open(out_path, "a", encoding="utf-8")

    # ============================
    # PROCESS INPUT FILE
    # ============================
    with open(input_file, "r", encoding="utf-8") as f_in:
        for _ in range(start_line):
            f_in.readline()

        pbar = tqdm(total=100, initial=start_line, desc=f"[{display_name}]", unit="paper")

        for line_num, line in enumerate(f_in, start=start_line):
            line = line.strip()
            if not line:
                pbar.update(1)
                continue

            try:
                paper = json.loads(line)
                raw_kw = paper.get("keywords", [])
                raw_text = ", ".join(raw_kw) if raw_kw else ""

                if len(raw_kw) == 0:
                    cleaned = []
                else:
                    result = call_llm_with_retry(chain, {"raw_keywords": raw_text})

                    if is_structured:
                        cleaned = result.cleaned_keywords
                    else:
                        text = getattr(result, "content", str(result))
                        match = re.search(r'\{.*"cleaned_keywords".*?\}', text, re.DOTALL)
                        cleaned = json.loads(match.group(0)).get("cleaned_keywords", []) if match else []

                # ============================
                # FINAL OUTPUT FORMAT
                # ============================
                row = {
                    "paper_id": paper.get("id", f"paper_{line_num}"),
                    "title": str(paper.get("title", ""))[:200],
                    "year": paper.get("year", ""),
                    "venue": paper.get("venue", ""),
                    "original_keywords": raw_kw,
                    "cleaned_keywords": cleaned,  # Store as list
                    "cleaned_keywords_str": " | ".join(cleaned),  # Optional human-readable
                    "num_original": len(raw_kw),
                    "num_cleaned": len(cleaned),
                    "model_display": display_name,
                    "full_model_name": full_model,
                }

                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                f_out.flush()

                # Save progress
                with open(prog_path, "w") as pf:
                    pf.write(str(line_num + 1))

                pbar.update(1)
                time.sleep(0.7 + random.uniform(0, 0.5))

            except Exception as e:
                print(f"\nERROR at line {line_num + 1}: {e}")
                time.sleep(3)

        pbar.close()
        f_out.close()

    if os.path.exists(prog_path):
        os.remove(prog_path)

print("\nAll models processed successfully.")
print("Output saved in:", output_dir)
