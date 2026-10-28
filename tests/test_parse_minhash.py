from src.minhash import get_shingles, make_hash_functions, get_signature, jaccard, signature_similarity
from src.parse_thread import parse_thread

a = "kerala has the highest literacy rate among indian states according to the 2011 census of india"
b = "kerala has the highest literacy rate among all indian states according to the 2011 census"
c = "the capital of tamil nadu is chennai which is located on the coromandel coast"

funcs = make_hash_functions(128)
sa = get_shingles(a)
sb = get_shingles(b)
sc = get_shingles(c)
sig_a = get_signature(sa, funcs)
sig_b = get_signature(sb, funcs)
sig_c = get_signature(sc, funcs)

print("a vs b  jaccard:", round(jaccard(sa, sb), 3), " minhash:", round(signature_similarity(sig_a, sig_b), 3))
print("a vs c  jaccard:", round(jaccard(sa, sc), 3), " minhash:", round(signature_similarity(sig_a, sig_c), 3))
assert abs(jaccard(sa, sb) - signature_similarity(sig_a, sig_b)) < 0.15
assert signature_similarity(sig_a, sig_c) < 0.1

with open("data/sample_threads/thread1.txt", encoding="utf-8") as f:
    parsed = parse_thread(f.read())
print("messages:", len(parsed["messages"]), " stats:", parsed["stats"])
assert len(parsed["messages"]) == 3
assert parsed["stats"]["duplicate_paragraphs"] >= 1
print("all tests passed")