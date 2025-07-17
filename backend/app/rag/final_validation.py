from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from ..models.query import AyahResult

load_dotenv()

class HadithVerdict(BaseModel):
    confidence: float = Field(description="Confidence that you have correctly validate the hadith.")
    verdict: str = Field(description="Valid, Invalid, or Unknown")
    summary: str = Field(description="Short explanation of why it is valid/invalid")


parser = PydanticOutputParser(pydantic_object=HadithVerdict)


template = """
You are an expert in Islamic scholarship. Your task is to first **analyze** the following Hadith and then **validate** it based **only on the provided Quranic ayahs**.

Follow these strict rules for validation:

- If the Hadith is clearly **supported** by one or more ayahs, mark it as **Valid**.
- If the Hadith is clearly **contradicted** by one or more ayahs, mark it as **Invalid**.
- If the ayahs are **insufficient**, **ambiguous**, or **not clearly related**, mark it as **Unknown**.

Do **not** rely on any external knowledge or assumptions. Your decision must be based strictly on the ayahs given.

Hadith:
"{hadith}"

Supported Ayahs:
{supported_text}

Contradicted Ayahs:
{contradicted_text}

Respond with:
{format_instructions}
"""

prompt = PromptTemplate(
    input_variables=["hadith", "supported_text", "contradicted_text"],
    partial_variables={"format_instructions": '''
Respond in **only** this JSON format:
{
  "confidence": 0.85,
  "verdict": "Valid",
  "summary": "The ayahs clearly support the message of the hadith."
}
'''},
    template=template,
)


def format_ayahs(ayahs: List[AyahResult]) -> str:
    if not ayahs:
        return "None"
    return "\n\n".join([
        f"""→ Score: {a.score:.2f}
- English: "{a.english_translation}"
- Arabic: {a.arabic_diacritics}
- Surah: {a.surah_name_english}, Ayah: {a.aya_number}"""
        for a in ayahs
    ])


def get_hadith_verdict_from_llm(hadith_result: dict, llm=None) -> HadithVerdict:
    if llm is None:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    chain = prompt | llm | parser

    supported_text = format_ayahs(hadith_result.get("supported", []))
    contradicted_text = format_ayahs(hadith_result.get("contradicted", []))

    return chain.invoke({
        "hadith": hadith_result["hadith"],
        "supported_text": supported_text,
        "contradicted_text": contradicted_text
    })
