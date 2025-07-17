from fastapi import APIRouter
from ..rag.open_source_models import extract_isnad , extract_narrator_chain
from ..rag.closed_source_models import extract_narrators_chain_with_llm
from pydantic import BaseModel
from typing import List, Optional
import spacy
from spacy.pipeline import EntityRuler
from typing import List, Tuple
import json
import os
import re

router = APIRouter()

class HadithInput(BaseModel):
    hadith_text: str
    language: Optional[str] = "english"

@router.post('/extract_narrators_llm')
def extract_narrators_llm(input: HadithInput):
    if input.language.lower() == 'english' or input.language == 'arabic':
        narrators, content = extract_narrators_chain_with_llm(input.hadith_text)
    else:
        narrators = [] 
        content= ""
        
    data = {
        'hadith_text' : input.hadith_text,
        'language' : input.language, 
        'narrators_chain' : narrators,
        'hadith_content': content
    }

    file_path = "closed_source_models_results.json"
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                all_data = json.load(f)
        else:
            all_data = []
        all_data.append(data)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        return {"error": f"File write error: {str(e)}"}
    return data

@router.post("/extract_narrators_ner_dslim")
def extract_narrators(input: HadithInput):
    if input.language.lower() == "english":
        narrators = extract_isnad(input.hadith_text)
    data = {
        "hadith_text": input.hadith_text,
        "language": input.language,
        "narrators_chain": narrators
    }

    file_path = "open_source_models_results.json"
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                all_data = json.load(f)
        else:
            all_data = []
        all_data.append(data)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        return {"error": f"File write error: {str(e)}"}
    return data

@router.post("/extract_narrators_ner_CAMel_Lab")
def extract_narrators(input: HadithInput):
    if input.language == "arabic":
        narrators = extract_narrator_chain(input.hadith_text)
    data = {
        "hadith_text": input.hadith_text,
        "language": input.language,
        "narrators_chain": narrators
    }

    file_path = "open_source_models_results.json"
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                all_data = json.load(f)
        else:
            all_data = []
        all_data.append(data)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        return {"error": f"File write error: {str(e)}"}
    return data

@router.get("/all_narrators_from_llm")
def get_all_narrators_llm():
    file_path = "closed_source_models_results.json"
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                all_data = json.load(f)
            return all_data
        else:
            return [] 
    except Exception as e:
        return {"error": f"File read error: {str(e)}"}
    
@router.get("/all_narrators_from_ner")
def get_all_narrators_ner():
    file_path = "open_source_models_results.json"
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                all_data = json.load(f)
            return all_data
        else:
            return [] 
    except Exception as e:
        return {"error": f"File read error: {str(e)}"} 