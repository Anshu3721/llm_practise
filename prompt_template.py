from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

# Creating a prompt template with a variable
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions briefly.Use a simple example to explain the concept."),
    ("human", "{user_question}")
])

# Creating a chain - connecting prompt to llm
chain = prompt | llm | StrOutputParser()

# Invoking with actual value
# Invoking with actual value
response = chain.invoke({"user_question": "What is machine learning?"})

# Clean string extraction
output = response if isinstance(response, str) else response.content
print(output)
print(type(output))