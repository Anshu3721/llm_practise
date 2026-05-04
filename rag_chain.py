from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Step 1 - Load and split document
loader = TextLoader("knowledge_base.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 20
)
chunks = splitter.split_documents(documents)

# Step 2 - Create vector store
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)
retriver = vectorstore.as_retriever(search_kwargs={"k": 2})

# Step 3 - Setup LLM
llm = ChatGroq(
    api_key = os.getenv("GROQ_API_KEY"),
    model_name = "llama-3.3-70b-versatile"
)

# Step 4 - Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful customer support assistant.
     Answer the question using only the context provided below.
     If the answer is not in the context, say 'I don't have information on that.'
     
     Context: {context}"""),
    ("human", "{question}")
])

# Step 5 - Full RAG chain
def format_docs(docs):
    return "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

def rag_answer(question):
    # retrive relevant chunks
    relevant_docs = retriver.invoke(question)
    context = format_docs(relevant_docs)

    # Generate answer
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response.content


# Test the RAG chain
questions = [
    "What is the refund policy?",
    "How long does shipping take?",
    "How do I reset my password?",
    "What is the warranty period?",
    "What is the capital of France?"
]

for question in questions:
    print(f"Q: {question}")
    print(f"A: {rag_answer(question)}")
    print("-" * 50)