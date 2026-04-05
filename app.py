from flask import Flask, request, jsonify
import os
import numpy as np
from PIL import Image
from src.embedding import get_embedding
from src.search import build_index, search

app = Flask(__name__)

# 🔹 YOUR ORIGINAL DATASET CODE
images = [
    "data/original/bag1.jpg",
    "data/dupe/bag2.jpg"
]

embeddings = []
for img in images:
    embeddings.append(get_embedding(img))

embeddings = np.array(embeddings).astype('float32')
index = build_index(embeddings)


# 🔹 API ROUTE
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    # save uploaded image
    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    # 🔥 THIS replaces your "query = ..."
    query = get_embedding(filepath).astype('float32')

    distances, indices = search(index, query, k=2)

    results = []

    for i, idx in enumerate(indices[0]):
        results.append({
            "image": images[idx],
            "distance": float(distances[0][i])
        })

    # 🔥 Your decision logic (converted)
    if distances[0][1] < 50:
        decision = "VERY SIMILAR"
    elif distances[0][1] < 100:
        decision = "SOME SIMILARITY"
    else:
        decision = "DIFFERENT DESIGN"

    return jsonify({
        "results": results,
        "decision": decision
    })


# 🔹 RUN SERVER
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)