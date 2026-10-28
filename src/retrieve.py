import re
import math
import time

from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

import config
from src.index import load_index, embed_query, dense_search, TEST_QUERIES
from src.bm25 import build_bm25, bm25_search

chunks = None
embeddings = None
bm25_model = None
reranker = None
entity_names = None


def score_of(pair):
    return pair[1]


def load_entities():
    names = []
    with open(config.PAGES_FILE, encoding="utf-8") as f:
        lines = f.read().split("\n")
    for i in range(len(lines)):
        title = lines[i].strip()
        if title == "" or title == config.SUMMARY_PAGE:
            continue
        title = re.sub(r"\(.*?\)", "", title).strip()
        if title.endswith(", India"):
            title = title[:-len(", India")]
        names.append(title.lower())
    # longest first so "jammu and kashmir" is tried before shorter names
    names.sort(key=len, reverse=True)
    return names


def load_resources():
    global chunks, embeddings, bm25_model, reranker, entity_names
    if chunks is not None:
        return
    chunks, embeddings = load_index()
    texts = []
    for i in range(len(chunks)):
        texts.append(chunks[i]["text"])
    bm25_model = build_bm25(texts)
    reranker = CrossEncoder(config.RERANK_MODEL)
    entity_names = load_entities()


def find_entities(query):
    q = query.lower()
    found = []
    for i in range(len(entity_names)):
        if re.search(r"\b" + re.escape(entity_names[i]) + r"\b", q):
            found.append(entity_names[i])
    return found


def allowed(chunk, entities):
    if len(entities) == 0:
        return True
    if chunk["page"] == config.SUMMARY_PAGE:
        return True
    page = chunk["page"].lower()
    for i in range(len(entities)):
        if entities[i] in page:
            return True
    return False


def filter_ranked(ranked, entities, limit):
    kept = []
    for i in range(len(ranked)):
        if len(kept) >= limit:
            break
        if allowed(chunks[ranked[i][0]], entities):
            kept.append(ranked[i])
    return kept


def rrf(rank_lists, k):
    scores = {}
    for r in range(len(rank_lists)):
        ids = rank_lists[r]
        for i in range(len(ids)):
            rank = i + 1
            if ids[i] not in scores:
                scores[ids[i]] = 0
            scores[ids[i]] += 1 / (k + rank)
    results = []
    for idx in scores:
        results.append((idx, scores[idx]))
    results.sort(key=score_of, reverse=True)
    return results


def mmr(query_vec, candidate_ids, top_n, lam):
    if len(candidate_ids) == 0:
        return []
    cand_vecs = embeddings[candidate_ids]
    sim_q = cosine_similarity(query_vec.reshape(1, -1), cand_vecs)[0]
    sim_dd = cosine_similarity(cand_vecs)

    selected = []
    results = []
    while len(selected) < top_n and len(selected) < len(candidate_ids):
        best_i = -1
        best_score = 0
        for i in range(len(candidate_ids)):
            if i in selected:
                continue
            max_sim = 0
            for j in range(len(selected)):
                if sim_dd[i][selected[j]] > max_sim:
                    max_sim = sim_dd[i][selected[j]]
            score = lam * sim_q[i] - (1 - lam) * max_sim
            if best_i == -1 or score > best_score:
                best_i = i
                best_score = score
        selected.append(best_i)
        results.append((candidate_ids[best_i], float(best_score)))
    return results


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def rerank(query, ids):
    if len(ids) == 0:
        return []
    pairs = []
    for i in range(len(ids)):
        pairs.append([query, chunks[ids[i]]["text"]])
    logits = reranker.predict(pairs)
    results = []
    for i in range(len(ids)):
        results.append((ids[i], sigmoid(float(logits[i]))))
    results.sort(key=score_of, reverse=True)
    return results[:config.FINAL_TOP]


def ids_of(ranked):
    ids = []
    for i in range(len(ranked)):
        ids.append(ranked[i][0])
    return ids


def retrieve(query):
    load_resources()
    entities = find_entities(query)

    dense = dense_search(query, embeddings, len(chunks))
    dense = filter_ranked(dense, entities, config.DENSE_TOP_K)

    bm25 = bm25_search(query, bm25_model, len(chunks), config.BM25_K1, config.BM25_B)
    bm25 = filter_ranked(bm25, entities, config.BM25_TOP_K)

    fused = rrf([ids_of(dense), ids_of(bm25)], config.RRF_K)[:config.RRF_TOP]
    diverse = mmr(embed_query(query), ids_of(fused), config.MMR_TOP, config.MMR_LAMBDA)
    final = rerank(query, ids_of(diverse))

    best = 0
    if len(final) > 0:
        best = final[0][1]
    return {"query": query, "entities": entities, "dense": dense, "bm25": bm25, "rrf": fused,
            "mmr": diverse, "final": final, "best_score": best,
            "found": best >= config.NOT_FOUND_THRESHOLD}


def print_stage(name, ranked, n):
    print("  " + name + ":")
    for i in range(min(n, len(ranked))):
        idx, score = ranked[i]
        c = chunks[idx]
        print("    " + format(score, ".4f") + "  " + c["page"] + " | " + c["section"] + " | " + c["body"][:90])


def main():
    start = time.time()
    load_resources()
    print("Resources loaded in", round(time.time() - start, 1), "s")

    queries = TEST_QUERIES + ["Compare the literacy rate of Kerala and Bihar",
                              "Who won the FIFA World Cup in 2022?"]
    for q in queries:
        start = time.time()
        result = retrieve(q)
        print("\nQuery:", q)
        print("  Entities:", result["entities"])
        print_stage("Dense", result["dense"], 3)
        print_stage("BM25", result["bm25"], 3)
        print_stage("RRF", result["rrf"], 3)
        print_stage("MMR", result["mmr"], 3)
        print_stage("Final (reranked)", result["final"], config.FINAL_TOP)
        if result["found"]:
            status = "FOUND"
        else:
            status = "NOT FOUND"
        print("  Best score:", round(result["best_score"], 4), "->", status)
        print("  Time:", round(time.time() - start, 2), "s")


if __name__ == "__main__":
    main()
