import faiss
import numpy as np

def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search(index, query, k=3):
    query = query.reshape(1, -1)
    distances, indices = index.search(query, k)
    return distances, indices