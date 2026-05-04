from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

# Two similar sentences and one different
sentences = [
    "I love playing football",
    "I enjoy playing soccer",
    "Machine learning is  fascinating"
]

# convert sentences to embeddings
embeddings = model.encode(sentences)

# check Similarity
sim_matrix = cosine_similarity(embeddings)

print("Simialrity between sentences 1 and 2:", round(sim_matrix[0][1], 4)) 
print("Simialrity between sentences 1 and 3:", round(sim_matrix[0][2], 4))   