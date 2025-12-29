import json
import os
import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download stopwords once
nltk.download('punkt')
nltk.download('stopwords')

# Input dataset path
INPUT_FILE = r"C:\Users\PMLS\Downloads\keyword processing\dblp_cleaned_fast_100_random_samples.jsonl"

# Output folder
OUTPUT_FOLDER = r"C:\Users\PMLS\Downloads\keyword processing\stopword_removal_using_llms_nltk"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load English stopwords
stop_words = set(stopwords.words("english"))

def clean_keyword(keyword):
    """Remove stopwords from a keyword string."""
    tokens = word_tokenize(keyword.lower())
    cleaned = [t for t in tokens if t not in stop_words]
    return " ".join(cleaned)

# Output file
output_path = os.path.join(OUTPUT_FOLDER, "nltk_stopword_cleaned.jsonl")

with open(INPUT_FILE, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:

    for line_num, line in enumerate(infile, start=1):

        paper = json.loads(line.strip())

        raw_kw = paper.get("keywords", [])

        # Clean each keyword
        cleaned_kw = [clean_keyword(k) for k in raw_kw if isinstance(k, str)]

        row = {
            "paper_id": paper.get("id", f"paper_{line_num}"),
            "title": str(paper.get("title", ""))[:200],
            "year": paper.get("year", ""),
            "venue": paper.get("venue", ""),
            "original_keywords": " | ".join(map(str, raw_kw)),
            "cleaned_keywords": " | ".join(cleaned_kw),
            "num_original": len(raw_kw),
            "num_cleaned": len(cleaned_kw),
            "model_display": "NLTK Stopword Removal",
            "full_model_name": "nltk-stopwords"
        }

        outfile.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"Processed: NLTK | Paper {line_num} | {len(raw_kw)} → {len(cleaned_kw)}")

        time.sleep(0.5)

print("\nCompleted! Output saved to:")
print(output_path)
