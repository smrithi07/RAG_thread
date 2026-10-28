import os
import json
import time

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import config

TEST_QUERIES = [
    "What is the capital of Kerala?",
    "literacy rate of Tamil Nadu",
    "districts of Karnataka",
    "Which state is known as God's Own Country?",
    "official animal of Andaman and Nicobar",
]

model = None


def load_chunks():
    with open(config.CHUNKS_FILE, encoding="utf-8") as f:
        return json.load(f)


def get_model():
    global model
    if model is None:
        model = SentenceTransformer(config.EMBED_MODEL)
    return model


def build_index():
    chunks = load_chunks()
    texts = []
    for i in range(len(chunks)):
        texts.append(chunks[i]["text"])

    embeddings = get_model().encode(texts, batch_size=64, normalize_embeddings=True,
                                    show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    os.makedirs(config.INDEX_DIR, exist_ok=True)
    np.save(config.EMBED_FILE, embeddings)
    meta = {"model": config.EMBED_MODEL, "num_chunks": len(chunks), "dim": int(embeddings.shape[1])}
    with open(config.META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return chunks, embeddings


def load_index():
    chunks = load_chunks()
    embeddings = np.load(config.EMBED_FILE)
    if embeddings.shape[0] != len(chunks):
        raise ValueError("embeddings.npy has " + str(embeddings.shape[0]) + " rows but chunks.json has "
                         + str(len(chunks)) + " chunks. Rebuild with: python -m src.index")
    return chunks, embeddings


def embed_query(query):
    return get_model().encode(config.QUERY_PREFIX + query, normalize_embeddings=True)


def dense_search(query, embeddings, top_k):
    query_vec = embed_query(query)
    scores = cosine_similarity(query_vec.reshape(1, -1), embeddings)[0]
    order = np.argsort(-scores)[:top_k]
    results = []
    for i in range(len(order)):
        idx = int(order[i])
        results.append((idx, float(scores[idx])))
    return results


def print_results(chunks, results):
    for i in range(len(results)):
        idx, score = results[i]
        c = chunks[idx]
        print("  " + format(score, ".3f") + "  " + c["page"] + " | " + c["section"] + " | " + c["body"][:100])


def main():
    start = time.time()
    chunks, embeddings = build_index()
    print("Chunks:", len(chunks))
    print("Dim:", embeddings.shape[1])
    print("Seconds:", round(time.time() - start, 1))

    for q in TEST_QUERIES:
        print("\nQuery:", q)
        print_results(chunks, dense_search(q, embeddings, 5))


if __name__ == "__main__":
    main()
