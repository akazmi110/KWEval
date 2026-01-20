import json
import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load env

load_dotenv()

# IMPORTANT: Key name must match your .env variable
API_KEY = os.getenv("keyword_processing")

if not API_KEY:
    raise ValueError("API key not found in env file under name 'keyword_processing'.")

# LangChain Chat Models (OpenRouter-compatible)
def get_llm(model_name):
    return ChatOpenAI(
        model=model_name,
        api_key=API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1,
        max_tokens=200,
    )

# Models
models = [
    {"full_model": "openai/gpt-4o-mini", "display_name": "GPT-4o Mini"},
    {"full_model": "deepseek/deepseek-r1-0528-qwen3-8b", "display_name": "DeepSeek R1"},
    {"full_model": "qwen/qwen-2.5-72b-instruct", "display_name": "Qwen 2.5 72B"},
    {"full_model": "x-ai/grok-4.1-fast", "display_name": "Grok 4.1 Fast"},
    {"full_model": "nvidia/nemotron-nano-12b-v2-vl", "display_name": "Nemotron Nano 12B"},
    {"full_model": "mistralai/mistral-7b-instruct", "display_name": "Mistral 7B Instruct"},
]

# Input file
input_file = "dblp_1000_random_samples.jsonl"

# Output folder
output_folder = r"stopword_removal_using_llms"
os.makedirs(output_folder, exist_ok=True)

# LangChain Prompt Template
prompt_template = ChatPromptTemplate.from_template("""
Remove stopwords from the following academic keywords.
Return ONLY cleaned keywords separated by " | ".
If nothing remains, return "NONE".

Keywords: {keywords}
""")


# Output parser
parser = StrOutputParser()

def remove_stopwords_with_llm(keywords_str, model_name):
    """Runs LangChain LLM pipeline."""
    llm = get_llm(model_name)

    chain = prompt_template | llm | parser

    try:
        cleaned = chain.invoke({"keywords": keywords_str}).strip()

        # Remove accidental quotes
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]

        return cleaned
    except Exception as e:
        print(f"Error with model {model_name}: {e}")
        return "ERROR"

# Loop through models
for model_info in models:
    full_model = model_info["full_model"]
    display_name = model_info["display_name"]

    output_path = os.path.join(
        output_folder,
        f"{full_model.replace('/', '_')}.jsonl"
    )

    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        line_num = 0

        for line in infile:
            if not line.strip():
                continue

            line_num += 1

            try:
                paper = json.loads(line.strip())
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON at line {line_num}")
                continue

            raw_kw = paper.get("keywords", [])
            num_original = len(raw_kw)

            # Format original keywords
            original_keywords = " | ".join([str(k).strip() for k in raw_kw if str(k).strip()])

            if num_original == 0:
                cleaned_kw_list = []
                cleaned_str = ""
                num_cleaned = 0
            else:
                # Run LLM through LangChain
                cleaned_keywords = remove_stopwords_with_llm(original_keywords, full_model)
                cleaned_kw_list = [
                    kw.strip() for kw in cleaned_keywords.split(" | ")
                    if kw.strip() and kw.strip() != "NONE"
                ]
                cleaned_str = " | ".join(cleaned_kw_list)
                num_cleaned = len(cleaned_kw_list)

            # Build output row
            row = {
                "paper_id": paper.get("id", f"paper_{line_num}"),
                "title": str(paper.get("title", ""))[:200],
                "year": paper.get("year", ""),
                "venue": paper.get("venue", ""),
                "original_keywords": original_keywords,
                "cleaned_keywords": cleaned_str,
                "num_original": num_original,
                "num_cleaned": num_cleaned,
                "model_display": display_name,
                "full_model_name": full_model,
            }

            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")

            print(f"Processed: {display_name} | Paper {line_num} | {num_original} → {num_cleaned}")

            time.sleep(5)

    print(f"Completed: {display_name} → {output_path}")

print("All models processed using LangChain.")
