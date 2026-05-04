from dotenv import load_dotenv
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


# Sample knowledge base
documents = [
    "Python is a popular programming language for data science and machine learning",
    "Machine learning helps computers learn from data without being explicitly programmed",
    "FAISS is a library developed by Facebook for efficient similarity search",
    "LangChain helps developers build applications powered by large language models",
    "Neural networks are computing systems inspired by biological neural networks in brains",
    "Deep learning is a subset of machine learning using multi-layered neural networks",
    "Natural language processing helps computers understand and generate human language",
    "Vector databases store embeddings and enable fast semantic search at scale"
]


# Step 1: create  embeddings and FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents).astype('float32')
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Step 2 - Setup LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)


# Step 3 - Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. 
     Answer the question using only the context provided below. Try to understand the context first and IF you find relevant answers then answer it.
     If the answer is not in the context, say 'I don't have information on that.'
     
     Context: {context}"""),
    ("human", "{question}")
])

chain = prompt | llm

# Step 4 - Search and answer
def search_and_answer(query):
    # search FAISS for relevant documents
    query_embedding  = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, k=2)

    # Add these debug lines
    print(f"Retrieved docs for '{query}':")
    for i, idx in enumerate(indices[0]):
        print(f"  {i+1}. {documents[idx]}")

    # Get relevant Documents
    relevant_docs = [documents[i] for i in indices[0]]
    context = "\n".join(relevant_docs)

    # Invoke LLM with context and question
    response = chain.invoke({
        "context": context,
        "question": query
    })
    return response.content

# Test it
queries = [
    # "What is machine learning?",
    "How does FAISS work?",
    "What is the capital of France?"  # not in knowledge base
]

for query in queries:
    print(f"\nQuestion: {query}")
    print(f"Answer: {search_and_answer(query)}")
    print("-" * 50)