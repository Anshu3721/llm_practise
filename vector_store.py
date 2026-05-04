from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Sample documents
documents = [
    "Python is a popular programming language for data science",
    "Machine learning helps computers learn from data",
    "FAISS is a library for efficient similarity search",
    "LangChain helps build applications with LLMs",
    "Neural networks are inspired by the human brain"
]

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert documents to embeddings
embeddings = model.encode(documents)
print("Embeddings shape:", embeddings.shape)   #Embeddings shape: (5, 384)
print("Sample embedding vector:", embeddings[0][:5])  # Sample embedding vector: [-0.05738574 -0.00512589 -0.01381379  0.04639213 -0.03430237]

embeddings = np.array(embeddings).astype('float32')

# Create FAISS index
dimension = embeddings.shape[1] # size of embedding vector--> Embedding dimension: 384
print("Embedding dimension:", dimension)

index = faiss.IndexFlatL2(dimension)  # L2 = Euclidean distance
print("Is index trained?", index.is_trained) # True for IndexFlatL2, False for some other types of indexes that require training

index.add(embeddings)  # Add embeddings to index

print(f"Total documents in vector store: {index.ntotal}") #Total documents in vector store: 5

# Now Search
query = "how do computer learn?"
query_embedding = model.encode([query]).astype('float32')
print("Query embedding shape:", query_embedding.shape) # Query embedding shape: (1, 384)
print("Sample query embedding vector:", query_embedding[0][:5])  # Sample query embedding vector: [-0.032  0.012 -0.015  0.028 -0.022]

# search top 2 most similar documents
distances, indices = index.search(query_embedding, k=2)
print("Distances:", distances) #Distances: [[0.85 1.2]]
print("Indices:", indices)  #Indices: [[1 4]] 

print(f"\nQuery: {query}")
print(f"Most relevant documents:")
for i, idx in enumerate(indices[0]):
    print(f"{i+1}. {documents[idx]}")