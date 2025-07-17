
from transformers import pipeline, logging
import re
import json 
import regex
import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")

logging.set_verbosity_error()

def extract_isnad(hadith_text: str) -> list[str]:
    """
    Extracts the chain of narrators (isnad) from a Hadith text using a pre-trained NER model.

    Args:
        hadith_text: A string containing the full Hadith text in English.

    Returns:
        A list of strings, where each string is a narrator's name in sequential order.
        Returns an empty list if no narrators are identified.
    """
    print(f"--- Processing Hadith ---")
    
    # 1. PRE-PROCESSING: Isolate the Isnad
    # The isnad typically ends where the main report (matn) begins. We use regex to find common
    # phrases that introduce the matn, such as "that the Prophet said" or "reported:".
    matn_starters = [
        r'that the Messenger of Allah \(ﷺ\) said',
        r'that the Prophet \(ﷺ\) said',
        r'that the Prophet said',
        r'the Prophet \(ﷺ\) said',
        r'he said, "The Prophet',
        r'said:',
        r'reported:'
    ]
    matn_pattern = re.compile(r'(?i)' + r'|'.join(matn_starters))
    match = matn_pattern.search(hadith_text)
    
    isnad_text = hadith_text
    if match:
        isnad_text = hadith_text[:match.start()]
        print(f"Isnad segment identified and isolated.")
    else:
        print("Warning: Could not definitively separate isnad from matn. Processing entire text.")

    isnad_text = " ".join(isnad_text.split())

    if not isnad_text:
        print("Isnad text is empty after pre-processing. Aborting.")
        return []

    # 2. NER EXTRACTION: Identify Persons
    # Model: 'dslim/bert-base-NER' is a robust choice for general-purpose NER.
    # aggregation_strategy='simple' groups word pieces (e.g., "Abdur", "-", "Rahman")
    # into a single entity ("Abdur-Rahman").

    try:
        ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple" , token=HF_TOKEN )
        print("NER pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading NER pipeline: {e}")
        return []

    ner_results = ner_pipeline(isnad_text)
    print(f"NER model identified {len(ner_results)} potential entities.")

    narrator_chain = []
    
    exclusion_list = [
        "prophet", "messenger of allah", "allah's messenger", "allah", "ibn shihab",
    ]

    for entity in ner_results:
        if entity['entity_group'] == 'PER':
            name = entity['word'].strip()
            
            # Check against the exclusion list (case-insensitive).
            is_excluded = any(excluded_term in name.lower() for excluded_term in exclusion_list)
            
            if not is_excluded:
                # Avoid adding consecutive duplicates.
                if not narrator_chain or narrator_chain[-1] != name:
                    narrator_chain.append(name)
                    print(f"  [+] Added narrator: {name} (Confidence: {entity['score']:.2f})")
                else:
                    print(f"  [-] Skipped duplicate: {name}")
            else:
                print(f"  [-] Filtered out excluded term: {name}")

    print(f"--- Extraction Complete ---")
    return narrator_chain




try:
    NER_PIPELINE = pipeline(
        "ner",
        model="CAMeL-Lab/bert-base-arabic-camelbert-msa-ner",
        grouped_entities=True,
        token=os.environ.get("HF_TOKEN")
    )


except Exception as e:
    print(f"FATAL: Could not load NER model. Error: {e}")
    NER_PIPELINE = None

NARRATOR_CONNECTORS = [
    'حَدَّثَنَا', 'حَدَّثَنِي', 'حَدَّثَتْنَا', 'حَدَّثَتْنِي',
    'أَخْبَرَنَا', 'أَخْبَرَنِي',
    'سَمِعْتُ', 'سَمِعَ',
    'قَالَ', 'قَالَتْ',
    'عَنْ'
]
CONNECTORS_PATTERN = regex.compile(f"({'|'.join(NARRATOR_CONNECTORS)})")

MATN_STARTERS = [
    'أَنَّ رَسُولَ اللَّهِ',
    'أَنَّ النَّبِيَّ',
    'يَقُولُ',
    'قَالَ رَسُولُ اللَّهِ'
]

# Example function to merge subword tokens correctly
def merge_tokens(ner_results):
    names = []
    current_name = []
    for entity in ner_results:
        if entity['entity_group'] != 'PERS':
            continue
        word = entity['word']
        if word.startswith('##'):
            if current_name:
                current_name[-1] += word[2:]
            else:
                current_name.append(word[2:])
        else:
            if current_name:
                names.append(' '.join(current_name))
                current_name = []
            current_name.append(word)
    if current_name:
        names.append(' '.join(current_name))
    return names


def extract_narrator_chain(hadith_text: str) -> list[str]:
    if not hadith_text or not isinstance(hadith_text, str):
        return []
    
    if not NER_PIPELINE:
        raise RuntimeError("NER model is not available.")
    
    # Extract isnad part
    isnad_text = hadith_text
    for starter in MATN_STARTERS:
        if starter in hadith_text:
            isnad_text = hadith_text.split(starter, 1)[0]
            break

    # Split using connectors
    segments = CONNECTORS_PATTERN.split(isnad_text)
    
    narrator_phrases = []
    for i in range(1, len(segments), 2):
        if i + 1 < len(segments):
            phrase = segments[i] + segments[i+1]
            narrator_phrases.append(phrase.strip())
    
    if not narrator_phrases and isnad_text.strip():
        narrator_phrases.append(isnad_text.strip())

    narrator_chain = []
    for phrase in narrator_phrases:
        if not phrase:
            continue

        ner_results = NER_PIPELINE(phrase)
        person_entities = merge_tokens(ner_results)
        
        narrator_chain.extend(person_entities)  
    
    return narrator_chain
