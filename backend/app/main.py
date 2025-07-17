from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes.quran import router as quran_router
from .routes.extraction import router as extraction_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(quran_router, prefix="/api/quran", tags=["Quran Search"])
app.include_router(extraction_router, prefix="/api/extraction", tags=["Hadith & Narators Extraction"])


from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})