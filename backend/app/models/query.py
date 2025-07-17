from pydantic import BaseModel
from typing import List

class AyahResult(BaseModel):
    score: float
    english_translation: str
    surah_name_english: str
    aya_number: int
    arabic_diacritics: str


class HadithResult(BaseModel):
    hadith: str
    verdict: str
    summary: str
    confidence: float
    supported: List[AyahResult]
    contradicted: List[AyahResult]

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    results: HadithResult
