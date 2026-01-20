import json
import re
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# ========================= NLTK SETUP =========================
# Only download if not already present
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# ========================= FILE PATHS =========================
INPUT_FILE = r"keyword processing\dblp_1000_random_samples.jsonl"
OUTPUT_FILE = r"keyword processing\nltk_dblp_cleaned_1000_random_samples_keywords_cleaned.jsonl"

# ========================= STOPWORDS + GENERIC =========================
stop_words = set(stopwords.words("english"))
GENERIC_REMOVE = {
    "analysis", "study", "approach", "method", "methods", "technique", "techniques",
    "system", "systems", "model", "models", "algorithm", "algorithms", "framework",
    "based", "using", "application", "applications", "design", "development",
    "evaluation", "performance", "optimization", "problem", "problems", "research",
    "paper", "work", "proposed", "novel", "new", "improved", "efficient", "effective",
    "experimental", "theoretical", "simulation", "case", "review", "survey",
    "implementation", "tool", "tools", "software", "dataset", "results",
    "comparison", "investigation", "assessment", "prediction", "classification",
    "detection", "recognition", "estimation", "learning", "management", "control",
    "communication", "network", "networks", "processing", "information", "use",
    "used", "uses", "related", "various", "different", "multiple"
}
STOPWORDS = stop_words.union(GENERIC_REMOVE)

# ========================= ACRONYM MAP =========================
ACRONYM_MAP = {
    "nlp": "natural language processing",
    "ml": "machine learning",
    "dl": "deep learning",
    "cnn": "convolutional neural network",
    "rnn": "recurrent neural network",
    "lstm": "long short term memory",
    "gan": "generative adversarial network",
    "svm": "support vector machine",
    "pca": "principal component analysis",
    "iot": "internet of things",
    "iiot": "industrial internet of things",
    "ai": "artificial intelligence",
    "ann": "artificial neural network",
    "bci": "brain computer interface",
    "eeg": "electroencephalogram",
    "ecg": "electrocardiogram",
    "mri": "magnetic resonance imaging",
    "sdn": "software defined networking",
    "wlan": "wireless local area network",
    "wsn": "wireless sensor network",
    "qos": "quality of service",
    "vr": "virtual reality",
    "ar": "augmented reality",
    "hci": "human computer interaction",
    "ux": "user experience",
    "ui": "user interface",
    "gpu": "graphics processing unit",
    "ocr": "optical character recognition",
    "ner": "named entity recognition",
    "ir": "information retrieval",
    "cv": "computer vision",
    "rl": "reinforcement learning",
    "hpc": "high performance computing",
    "sbas-insar": "small baseline subset interferometric synthetic aperture radar",
    "in-memory": "in memory computation",
}

# ========================= TOOLS =========================
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def get_wordnet_pos(word: str):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def punctuation_to_space(text: str) -> str:
    trans = str.maketrans({p: " " for p in string.punctuation})
    text = text.translate(trans)
    return re.sub(r"\s+", " ", text).strip()

def expand_acronyms_text(text: str) -> str:
    raw_tokens = text.split()
    expanded_tokens = []

    for tok in raw_tokens:
        t = tok.lower()

        if t in ACRONYM_MAP:
            expanded_tokens.append(ACRONYM_MAP[t])
            continue

        if "-" in t:
            parts = [p for p in t.split("-") if p]
            part_expanded = []
            all_parts_expandable = True
            for p in parts:
                if p in ACRONYM_MAP:
                    part_expanded.append(ACRONYM_MAP[p])
                else:
                    all_parts_expandable = False
                    break
            if all_parts_expandable and part_expanded:
                expanded_tokens.append(" ".join(part_expanded))
                continue

        expanded_tokens.append(tok)

    return " ".join(expanded_tokens)

def clean_keywords(keyword_list):
    if not keyword_list:
        return []

    cleaned = []
    seen = set()

    for kw in keyword_list:
        kw = expand_acronyms_text(kw)
        kw = kw.lower()
        kw = punctuation_to_space(kw)
        tokens = word_tokenize(kw)
        filtered = [t for t in tokens if t not in STOPWORDS and len(t) > 2]

        processed = []
        for token in filtered:
            lemma = lemmatizer.lemmatize(token, get_wordnet_pos(token))
            if lemma != token:
                processed.append(lemma)
            else:
                processed.append(stemmer.stem(token))

        if processed:
            final_phrase = " ".join(processed)
            if final_phrase not in seen:
                seen.add(final_phrase)
                cleaned.append(final_phrase)

    return cleaned

# ========================= PROCESS JSONL FILE =========================
with open(INPUT_FILE, "r", encoding="utf-8") as f_in, open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue
        paper = json.loads(line)
        original_keywords = paper.get("keywords", [])
        paper["keywords_cleaned"] = clean_keywords(original_keywords)
        f_out.write(json.dumps(paper, ensure_ascii=False) + "\n")

print("Done! Cleaned keywords saved to:", OUTPUT_FILE)