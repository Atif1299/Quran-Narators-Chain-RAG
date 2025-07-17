from ..models.query import QueryResponse
from ..utils.query_ayahs import get_embedding, search_ayahs
from ..utils.get_hadith import extract_narrators_chain_with_llm
from ..rag.ayah_filter import filter_relevant_ayahs
from ..rag.hadith_validaiton import check_relationship
from ..rag.final_validation import get_hadith_verdict_from_llm
import os

def validate_hadith(query: str):
    narrators , query = extract_narrators_chain_with_llm(query)
    query_vector = get_embedding(query)

    ayahs = search_ayahs(query_vector=query_vector, limit=15)

    filtered_ayahs = filter_relevant_ayahs(ayahs=ayahs, hadith_text=query)

    hadith_result = {
        "hadith": query,
        "supported": [],
        "contradicted": []
    }

    for ayah in filtered_ayahs:
      ayah_text = f"{ayah.english_translation} (Surah: {ayah.surah_name_english}, Ayah: {ayah.aya_number})"
      label = check_relationship(query, ayah_text)
      if label == "Supported":
          hadith_result["supported"].append(ayah)
      elif label == "Contradicted":
          hadith_result["contradicted"].append(ayah)

    result = get_hadith_verdict_from_llm(hadith_result)
    hadith_result['verdict'] = result.verdict
    hadith_result['summary'] = result.summary
    hadith_result['confidence'] = result.confidence

    return QueryResponse(results=hadith_result)
