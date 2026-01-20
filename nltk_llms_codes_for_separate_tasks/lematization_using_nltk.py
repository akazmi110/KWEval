import json
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet

# Download required NLTK packages
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

input_file = "dblp_1000_random_samples.jsonl"
output_file = "dblp_keywords_lemmatized_stemmed.jsonl"

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def normalize_word(word):
    """Lemmatize, and if lemma = original, apply stemming."""
    original = word

    # Try different POS tags: noun, verb, adjective, adverb
    for pos in ['n', 'v', 'a', 'r']:
        lemma = lemmatizer.lemmatize(word, pos=pos)
        if lemma != word:
            return lemma
    
    # If still unchanged, apply stemming
    return stemmer.stem(original)


def normalize_keyword_phrase(phrase):
    """Normalize each word inside a multi-word keyword."""
    words = phrase.split()
    processed = [normalize_word(w.lower()) for w in words]
    return " ".join(processed)


# Process file
with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", encoding="utf-8") as outfile:

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

        original_keywords = paper.get("keywords", [])
        processed_keywords = []

        for kw in original_keywords:
            kw = kw.strip()
            if kw:
                processed_keywords.append(normalize_keyword_phrase(kw))

        # Add processed keywords to output
        paper["lemmatized_stemmed_keywords"] = processed_keywords

        # Write row
        outfile.write(json.dumps(paper, ensure_ascii=False) + "\n")

        print(f"Processed paper {line_num}: {len(original_keywords)} → {len(processed_keywords)}")


print("\nCompleted! Output saved to:", output_file)
