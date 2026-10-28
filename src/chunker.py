import os
import glob
import json

import config
from src.minhash import find_duplicates


def split_text(text, max_words, overlap):
    chunks = []
    current = []
    paragraphs = text.split("\n")
    for i in range(len(paragraphs)):
        words = paragraphs[i].split()
        if len(words) == 0:
            continue

        if len(words) > max_words:
            if len(current) > 0:
                chunks.append(" ".join(current))
                current = []
            step = max_words - overlap
            start = 0
            while start < len(words):
                chunks.append(" ".join(words[start:start + max_words]))
                if start + max_words >= len(words):
                    break
                start += step
            continue

        if len(current) + len(words) > max_words:
            chunks.append(" ".join(current))
            current = []
        current = current + words

    if len(current) > 0:
        chunks.append(" ".join(current))
    return chunks


def row_to_sentence(columns, row):
    pairs = []
    for i in range(len(row)):
        value = row[i].strip()
        if i < len(columns):
            col = columns[i].strip()
        else:
            col = ""
        if value == "" or value == col:
            continue
        if col == "Map" or col.startswith("Unnamed"):
            continue
        if col == "":
            pairs.append(value)
        else:
            pairs.append(col + ": " + value)
    return ", ".join(pairs)


def make_chunk(chunk_id, page, section, ctype, body, caption=""):
    prefix = page["title"] + " | " + section
    if ctype == "table" and caption != "":
        prefix = prefix + " | " + caption
    return {"id": chunk_id, "page": page["title"], "url": page["url"], "section": section,
            "type": ctype, "body": body, "text": prefix + " | " + body}


def chunk_page(page):
    chunks = []
    for section in page["sections"]:
        heading = section["heading"]

        pieces = split_text(section.get("text", ""), config.CHUNK_WORDS, config.CHUNK_OVERLAP)
        for i in range(len(pieces)):
            chunks.append(make_chunk(None, page, heading, "text", pieces[i]))

        tables = section.get("tables", [])
        for t in range(len(tables)):
            table = tables[t]
            caption = table.get("caption", "").strip()
            for r in range(len(table["rows"])):
                row = table["rows"][r]
                # infobox rows are [field, value], so the field acts as the column name
                if table["type"] == "infobox" and len(row) == 2:
                    sentence = row_to_sentence([row[0]], [row[1]])
                else:
                    sentence = row_to_sentence(table["columns"], row)
                if sentence != "":
                    chunks.append(make_chunk(None, page, heading, "table", sentence, caption))

        captions = section.get("captions", [])
        if len(captions) > 0:
            chunks.append(make_chunk(None, page, heading, "caption", "; ".join(captions)))
    return chunks


def remove_duplicates(chunks):
    # positions of text/caption chunks inside the full list
    positions = []
    bodies = []
    for i in range(len(chunks)):
        if chunks[i]["type"] != "table":
            positions.append(i)
            bodies.append(chunks[i]["body"])

    drop = find_duplicates(bodies, config.SHINGLE_SIZE, config.NUM_HASHES,
                           config.LSH_BANDS, config.DUP_THRESHOLD)
    drop_positions = set()
    for j in drop:
        drop_positions.add(positions[j])

    kept = []
    seen_rows = set()
    exact = 0
    for i in range(len(chunks)):
        if i in drop_positions:
            continue
        if chunks[i]["type"] == "table":
            if chunks[i]["text"] in seen_rows:
                exact += 1
                continue
            seen_rows.add(chunks[i]["text"])
        kept.append(chunks[i])
    return kept, len(drop_positions), exact


def main():
    files = sorted(glob.glob(os.path.join(config.RAW_DIR, "*.json")))
    chunks = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            page = json.load(f)
        chunks = chunks + chunk_page(page)
    total = len(chunks)

    chunks, minhash_dropped, exact_dropped = remove_duplicates(chunks)
    for i in range(len(chunks)):
        chunks[i]["id"] = i

    os.makedirs(config.INDEX_DIR, exist_ok=True)
    with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=1)

    counts = {"text": 0, "table": 0, "caption": 0}
    words = {"text": 0, "table": 0, "caption": 0}
    examples = {"text": [], "table": [], "caption": []}
    for c in chunks:
        counts[c["type"]] += 1
        words[c["type"]] += len(c["body"].split())
        if len(examples[c["type"]]) < 2:
            examples[c["type"]].append(c["text"])

    print("Pages:", len(files))
    print("Chunks before dedup:", total)
    print("Chunks saved:", len(chunks))
    print("MinHash dropped:", minhash_dropped)
    print("Exact dropped:", exact_dropped)
    for ctype in ["text", "table", "caption"]:
        avg = 0
        if counts[ctype] > 0:
            avg = round(words[ctype] / counts[ctype], 1)
        print(ctype, "chunks:", counts[ctype], "| avg words:", avg)
    for ctype in ["text", "table", "caption"]:
        print("\nExamples of", ctype + ":")
        for e in examples[ctype]:
            print(" -", e[:300])
    print("\nSaved to", config.CHUNKS_FILE)


if __name__ == "__main__":
    main()
