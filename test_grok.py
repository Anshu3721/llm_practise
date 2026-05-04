from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

# Your code → LangChain → Groq API → LLaMA 3.3 model → response back → response.content prints it
response = llm.invoke("Say hello in one line")
print(response.content)