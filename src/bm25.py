import re
import math
import time

import config
from src.index import load_chunks, print_results, TEST_QUERIES

STOPWORDS = set([
    "the", "a", "an", "of", "in", "is", "are", "was", "were", "be", "what", "which", "who", "whom",
    "how", "when", "where", "why", "and", "or", "to", "for", "on", "at", "by", "with", "from", "as",
    "it", "its", "this", "that", "these", "those", "do", "does", "did", "has", "have", "had", "s",
])


def tokenize(text):
    tokens = []
    words = re.findall(r"[a-z0-9]+", text.lower())
    for i in range(len(words)):
        if words[i] not in STOPWORDS:
            tokens.append(words[i])
    return tokens


def build_bm25(texts):
    doc_len = []
    postings = {}
    for doc in range(len(texts)):
        tokens = tokenize(texts[doc])
        doc_len.append(len(tokens))

        counts = {}
        for i in range(len(tokens)):
            t = tokens[i]
            if t not in counts:
                counts[t] = 0
            counts[t] += 1

        for t in counts:
            if t not in postings:
                postings[t] = []
            postings[t].append([doc, counts[t]])

    N = len(texts)
    avgdl = 0
    if N > 0:
        avgdl = sum(doc_len) / N

    idf = {}
    for t in postings:
        n = len(postings[t])
        idf[t] = math.log((N - n + 0.5) / (n + 0.5) + 1)

    return {"N": N, "doc_len": doc_len, "avgdl": avgdl, "postings": postings, "idf": idf}


def bm25_search(query, model, top_k, k1, b):
    scores = {}
    terms = tokenize(query)
    for i in range(len(terms)):
        t = terms[i]
        if t not in model["postings"]:
            continue
        idf = model["idf"][t]
        plist = model["postings"][t]
        for j in range(len(plist)):
            doc, tf = plist[j]
            norm = k1 * (1 - b + b * model["doc_len"][doc] / model["avgdl"])
            if doc not in scores:
                scores[doc] = 0
            scores[doc] += idf * tf * (k1 + 1) / (tf + norm)

    results = []
    for doc in scores:
        results.append((doc, scores[doc]))
    results.sort(key=score_of, reverse=True)
    return results[:top_k]


def score_of(pair):
    return pair[1]


def main():
    chunks = load_chunks()
    texts = []
    for i in range(len(chunks)):
        texts.append(chunks[i]["text"])

    start = time.time()
    model = build_bm25(texts)
    print("Docs:", model["N"])
    print("Vocabulary:", len(model["postings"]))
    print("avgdl:", round(model["avgdl"], 1))
    print("Seconds:", round(time.time() - start, 2))

    for q in TEST_QUERIES:
        print("\nQuery:", q)
        results = bm25_search(q, model, 5, config.BM25_K1, config.BM25_B)
        print_results(chunks, results)


if __name__ == "__main__":
    main()
