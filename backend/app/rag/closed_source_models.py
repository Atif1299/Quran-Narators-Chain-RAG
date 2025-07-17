from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel , Field
from typing import List, Optional
import json 
import os
import re

load_dotenv()

class HadithOuput(BaseModel):
    hadith_content : str = Field(description = 'Hadith Content extracted from the text')
    narators_chain : List[str] = Field(description = 'List of narrators in the order they appear in the perfect sequence')

template = PromptTemplate(
    template="""
        Extract the chain of narrators from the following hadith text:
        {hadith_text}
        Return Hadith Content separately and the list of narrators in the order they appear in the perfect sequence separately. Nothing else.
        """,

    input_variables=['hadith_text']
)
def extract_narrators_chain_with_llm(hadith_text: str) -> List[str]:
    llm = ChatGoogleGenerativeAI(model='gemini-2.5-pro',
                                 google_api_key=os.getenv("GOOGLE_API_KEY"))

    prompt = template.invoke({
        'hadith_text': hadith_text
    })

    try:
        structured_output = llm.with_structured_output(HadithOuput)
        response = structured_output.invoke(prompt)
        content = response.hadith_content
        narators = response.narators_chain

        if isinstance(narators, str):
            try:
                narrators = json.loads(narators)
                if isinstance(narrators, list):
                    return narrators, content
            except Exception:
                narrators = [n.strip(' ",') for n in re.split(r'\n|,', narators) if n.strip()]
                return narrators, content

        elif isinstance(narators, list):
            return narators, content
        else:
            return [f"Unexpected narrators format: {type(narators)}"], content

    except Exception as e:
        return [f"LLM extraction error: {str(e)}"], ""
