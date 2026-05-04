from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


# Step 1 - Load document
loader = TextLoader("knowledge_base.txt")
documents = loader.load()

# Step 2 - Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 20
)

chunks = splitter.split_documents(documents)

# Step 3 - Create embeddings model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 4 - Create FAISS vector store in one line
vectorstore = FAISS.from_documents(chunks, embedding_model)
print(f"Vector store created with {vectorstore.index.ntotal} chunks")

# Step 5 - Search
query = "what is the refund policy?"
results = vectorstore.similarity_search(query, k=2)

print(f"\nQuery: {query}")
print(f"\nResults:")
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(doc.page_content)
