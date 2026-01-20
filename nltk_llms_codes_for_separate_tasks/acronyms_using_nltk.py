import json

# Input/output files
input_file = "dblp_1000_random_samples.jsonl"
output_file = "dblp_keywords_acronyms_expanded.jsonl"

# Acronym dictionary
ACRONYM_MAP = {
    'nlp': 'natural language processing',
    'ml': 'machine learning',
    'dl': 'deep learning',
    'cnn': 'convolutional neural network',
    'rnn': 'recurrent neural network',
    'lstm': 'long short term memory',
    'gan': 'generative adversarial network',
    'svm': 'support vector machine',
    'pca': 'principal component analysis',
    'iot': 'internet of things',
    'ai': 'artificial intelligence',
    'ann': 'artificial neural network',
    'bci': 'brain computer interface',
    'eeg': 'electroencephalogram',
    'ecg': 'electrocardiogram',
    'mri': 'magnetic resonance imaging',
    'abr': 'adaptive bit rate',
    'sdn': 'software defined networking',
    'epon': 'ethernet passive optical network',
    'wlan': 'wireless local area network',
    'wsn': 'wireless sensor network',
    'qos': 'quality of service',
    'vr': 'virtual reality',
    'ar': 'augmented reality',
    'hci': 'human computer interaction',
    'ux': 'user experience',
    'ui': 'user interface',
    'gpu': 'graphics processing unit',
    'fpga': 'field programmable gate array',
    'asic': 'application specific integrated circuit',
    'ocr': 'optical character recognition',
    'ner': 'named entity recognition',
    'pos': 'part of speech',
    'ir': 'information retrieval',
    'cv': 'computer vision',
    'nlg': 'natural language generation',
    'nlu': 'natural language understanding',
    'asr': 'automatic speech recognition',
    'tts': 'text to speech',
    'rl': 'reinforcement learning',
    'dft': 'discrete fourier transform',
    'fft': 'fast fourier transform',
    'tfet': 'tunnel field effect transistor',
    'sic': 'silicon carbide',
    'hevc': 'high efficiency video coding',
    'vp9': 'vp9 video codec',
    'hpc': 'high performance computing',
    'in-memory': 'in memory computation',
    'sbas-insar': 'small baseline subset interferometric synthetic aperture radar',
    'sbvs': 'structure based virtual screening',
    'mmds': 'multidimensional scaling',
    'omux': 'output multiplexer',
    'ppq': 'preemptive priority queue',
    'mcp': 'multi point control protocol',
    'di-owim': 'double inverter open winding induction machine',
    'di-wrim': 'double inverter wound rotor induction machine',
}

def expand_acronym(word):
    """Expand acronym if found in map; otherwise return as-is."""
    w = word.lower()
    return ACRONYM_MAP.get(w, word)

def expand_phrase(phrase):
    """Expand full keyword (single word or multi-word)."""
    words = phrase.split()
    expanded = [expand_acronym(w) for w in words]
    return " ".join(expanded)

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
        expanded_keywords = []

        for kw in original_keywords:
            kw = kw.strip()
            if kw:
                expanded_keywords.append(expand_phrase(kw))

        # Add new field
        paper["expanded_acronyms_keywords"] = expanded_keywords

        # Save row
        outfile.write(json.dumps(paper, ensure_ascii=False) + "\n")

        print(f"Processed paper {line_num}: {len(original_keywords)} keywords")

print("\nCompleted! Acronym-expanded file saved to:", output_file)
