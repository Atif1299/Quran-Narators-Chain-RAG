from fastapi import APIRouter
from ..models.query import QueryRequest, QueryResponse
from ..services.quran_services import validate_hadith

router = APIRouter()

@router.post("/search", response_model=QueryResponse)
async def search_ayahs(request: QueryRequest):
    return validate_hadith(request.query)