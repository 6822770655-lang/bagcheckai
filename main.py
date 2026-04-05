import numpy as np
import os
import faiss
from src.embedding import get_embedding
from src.search import build_index, search
from PIL import Image


# load dataset
folder = "data/original"
images = [os.path.join(folder, f) for f in os.listdir(folder)]

# build embeddings
embeddings = [get_embedding(img) for img in images]
embeddings = np.vstack(embeddings)

# build index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# query
query = input("Enter image path: ")
query_emb = get_embedding(query)

# search
D, I = index.search(query_emb, k=3)

# results
print("Top matches:")
for i in I[0]:
    print(images[i])
'''

images = [
    "data/original/bag1.jpg",
    "data/dupe/bag2.jpg"
]

# Create embeddings
embeddings = []
for img in images:
    embeddings.append(get_embedding(img))

embeddings = np.array(embeddings).astype('float32')

# Build FAISS index
index = build_index(embeddings)

# Query with first image
query = get_embedding("data/original/bag1.jpg").astype('float32')

distances, indices = search(index, query, k=2)

print("\n=== RESULTS ===")

for i, idx in enumerate(indices[0]):
    print(f"Image: {images[idx]}")
    print(f"Distance: {distances[0][i]}\n")
    Image.open(images[idx]).show()

# Simple decision
if distances[0][1] < 50:
    print("VERY SIMILAR (most likely copy)")
elif distances[0][1] < 100:
    print("SOME SIMILARITY")
else:
    print("DIFFERENT DESIGN")
'''
