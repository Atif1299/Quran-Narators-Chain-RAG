from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()

class RelationshipOutput(BaseModel):
    classification: str = Field(description="One of: Supported, Weak Support, Contradicted")

def check_relationship(hadith_text, ayah_text, llm=None):
    if llm is None:
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    parser = PydanticOutputParser(pydantic_object=RelationshipOutput)

    prompt = PromptTemplate(
        template="""
You are a scholar analyzing the relationship between a Hadith and a specific Quranic ayah.

Your task is to carefully examine **all possible cases** and then assign the relationship to exactly **one** of the following categories:

- **Supported** – The ayah clearly confirms or directly aligns with the message of the Hadith.
- **Weak Support** – The ayah is somewhat related but does not directly or strongly support the Hadith.
- **Contradicted** – The ayah clearly opposes, denies, or invalidates the message of the Hadith.

Base your classification **strictly on the content** of the given ayah and hadith. Do not rely on external sources or assumptions.

Hadith:
"{hadith}"

Quranic Ayah:
"{ayah}"

Respond with:
{format_instructions}
""",
        input_variables=["hadith", "ayah"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    try:
        result = chain.invoke({"hadith": hadith_text, "ayah": ayah_text})
        return result.classification
    except Exception as e:
        print("Error during classification:", e)
        return "Weak Support"
