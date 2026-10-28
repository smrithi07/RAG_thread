import re
import random
import hashlib

PRIME = (1 << 61) - 1


def get_shingles(text, k=3):
    words = re.findall(r"[a-z0-9]+", text.lower())
    shingles = set()
    if len(words) == 0:
        return shingles
    if len(words) < k:
        shingles.add(" ".join(words))
        return shingles
    for i in range(len(words) - k + 1):
        shingles.add(" ".join(words[i:i + k]))
    return shingles


def shingle_to_int(shingle):
    digest = hashlib.md5(shingle.encode("utf-8")).hexdigest()
    return int(digest[:15], 16)


def make_hash_functions(num_hashes, seed=42):
    rng = random.Random(seed)
    funcs = []
    for i in range(num_hashes):
        a = rng.randint(1, PRIME - 1)
        b = rng.randint(0, PRIME - 1)
        funcs.append((a, b))
    return funcs


def get_signature(shingles, hash_funcs):
    ids = []
    for s in shingles:
        ids.append(shingle_to_int(s))

    signature = []
    for a, b in hash_funcs:
        minimum = PRIME
        for x in ids:
            h = (a * x + b) % PRIME
            if h < minimum:
                minimum = h
        signature.append(minimum)
    return signature


def jaccard(set1, set2):
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    common = 0
    for s in set1:
        if s in set2:
            common += 1
    union = len(set1) + len(set2) - common
    return common / union


def signature_similarity(sig1, sig2):
    same = 0
    for i in range(len(sig1)):
        if sig1[i] == sig2[i]:
            same += 1
    return same / len(sig1)


def candidate_pairs(signatures, bands):
    rows = len(signatures[0]) // bands
    buckets = {}
    for doc in range(len(signatures)):
        sig = signatures[doc]
        for band in range(bands):
            key = (band, tuple(sig[band * rows:(band + 1) * rows]))
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(doc)

    pairs = set()
    for docs in buckets.values():
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                pairs.add((docs[i], docs[j]))
    return pairs


def find_duplicates(texts, k=3, num_hashes=64, bands=8, threshold=0.8):
    hash_funcs = make_hash_functions(num_hashes)
    shingle_sets = []
    signatures = []
    for t in texts:
        s = get_shingles(t, k)
        shingle_sets.append(s)
        signatures.append(get_signature(s, hash_funcs))

    # keep the first occurrence, drop later near-copies
    drop = set()
    for i, j in sorted(candidate_pairs(signatures, bands)):
        if j in drop:
            continue
        if len(shingle_sets[i]) == 0 or len(shingle_sets[j]) == 0:
            continue
        if signature_similarity(signatures[i], signatures[j]) >= threshold:
            drop.add(j)
    return drop