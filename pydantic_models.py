import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq


load_dotenv()


class ClassificationResult(BaseModel):
    reason: Literal[
        "Billing",
        "Technical Issue",
        "Cancellation",
        "Refund",
        "Complaint",
        "Other"
    ] = Field(description="Choose only one allowed reason")


parser = PydanticOutputParser(pydantic_object=ClassificationResult)

prompt = PromptTemplate(
    template="""
You are a customer message classifier.

Allowed reasons:
- Billing
- Technical Issue
- Cancellation
- Refund
- Complaint
- Other

Examples:

Message: "I was charged twice."
Output: {{"reason": "Billing"}}

Message: "I cannot log in to my account."
Output: {{"reason": "Technical Issue"}}

Message: "I want to cancel my subscription."
Output: {{"reason": "Cancellation"}}

Now classify this message:

Message: {message}

{format_instructions}
""",
    input_variables=["message"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0
)

chain = prompt | model | parser

result = chain.invoke({
    "message": "My refund has not arrived yet."
})

print(result.reason)