import re
import sys
import json

import config
from src.minhash import find_duplicates

REPLY_MARKERS = [
    r"^On .+ wrote:$",
    r"^-+\s*Original Message\s*-+$",
    r"^-+\s*Forwarded message\s*-+$",
    r"^From:\s*.+",
]
HEADER = r"^(From|To|Cc|Sent|Date|Subject):"
SIGNOFF = r"^(thanks|thank you|thanks and regards|regards|best|best regards|kind regards|warm regards|cheers|sincerely)[,.!]?$"
GREETING = r"^(hi|hello|hey|dear|good morning|good afternoon)\b[^.?!]{0,30}[,!]?$"
DISCLAIMER = r"^(disclaimer|confidentiality notice|this e-?mail and any attachments)"


def strip_quote(line):
    depth = 0
    s = line.strip()
    while s.startswith(">"):
        s = s[1:].strip()
        depth += 1
    return s, depth


def is_marker(line):
    for p in REPLY_MARKERS:
        if re.match(p, line, re.IGNORECASE):
            return True
    return False


def get_sender(line):
    m = re.match(r"^From:\s*(.+)", line, re.IGNORECASE)
    if m:
        name = m.group(1)
    else:
        m = re.match(r"^On (.+) wrote:", line, re.IGNORECASE)
        if not m:
            return ""
        name = m.group(1).split(",")[-1]
    name = re.sub(r"<.*?>", "", name)
    return name.strip().strip('"')


def has_content(lines):
    for line in lines:
        if line != "" and not re.match(HEADER, line, re.IGNORECASE):
            return True
    return False


def split_messages(text, stats):
    messages = []
    current = {"sender": "", "lines": []}
    for raw in text.splitlines():
        line, depth = strip_quote(raw)
        if depth > 0:
            stats["quoted_lines"] += 1

        if is_marker(line):
            if has_content(current["lines"]):
                messages.append(current)
                current = {"sender": "", "lines": []}
            sender = get_sender(line)
            if sender:
                current["sender"] = sender
            continue
        current["lines"].append(line)

    if has_content(current["lines"]):
        messages.append(current)
    return messages


def clean_message(lines, stats):
    kept = []
    subject = ""
    signer = ""
    for i in range(len(lines)):
        line = lines[i]
        if re.match(HEADER, line, re.IGNORECASE):
            stats["header_lines"] += 1
            if line.lower().startswith("subject:"):
                subject = line[8:].strip()
            continue

        low = line.lower()
        if re.match(GREETING, low):
            continue
        if line == "--" or re.match(SIGNOFF, low) or re.match(DISCLAIMER, low) or low.startswith("sent from my"):
            for rest in lines[i:]:
                if rest != "":
                    stats["signature_lines"] += 1
            if re.match(SIGNOFF, low):
                for rest in lines[i + 1:]:
                    if rest != "":
                        signer = rest
                        break
            break
        kept.append(line)

    return "\n".join(kept).strip(), subject, signer


def remove_repeated(messages, stats):
    paras = []
    for i in range(len(messages)):
        for p in re.split(r"\n\s*\n", messages[i]["text"]):
            p = p.strip()
            if p:
                paras.append([i, p])

    long_ids = []
    long_texts = []
    for idx in range(len(paras)):
        if len(paras[idx][1].split()) >= config.MIN_PARA_WORDS:
            long_ids.append(idx)
            long_texts.append(paras[idx][1])

    drop = set()
    if len(long_texts) > 1:
        dups = find_duplicates(long_texts, config.SHINGLE_SIZE, config.NUM_HASHES,
                               config.LSH_BANDS, config.DUP_THRESHOLD)
        for j in dups:
            drop.add(long_ids[j])
    stats["duplicate_paragraphs"] = len(drop)

    new_texts = []
    for i in range(len(messages)):
        new_texts.append([])
    for idx in range(len(paras)):
        if idx not in drop:
            new_texts[paras[idx][0]].append(paras[idx][1])

    result = []
    for i in range(len(messages)):
        if len(new_texts[i]) > 0:
            result.append({"sender": messages[i]["sender"], "text": "\n\n".join(new_texts[i])})
    return result


def parse_thread(text):
    stats = {"quoted_lines": 0, "header_lines": 0, "signature_lines": 0, "duplicate_paragraphs": 0}
    raw_messages = split_messages(text, stats)

    subject = ""
    messages = []
    for m in raw_messages:
        body, subj, signer = clean_message(m["lines"], stats)
        if subj and not subject:
            subject = subj
        sender = m["sender"] or signer or "Unknown"
        if body:
            messages.append({"sender": sender, "text": body})

    # pasted threads are newest-first, we want oldest-first
    messages.reverse()
    messages = remove_repeated(messages, stats)
    subject = re.sub(r"^((re|fw|fwd):\s*)+", "", subject, flags=re.IGNORECASE)

    return {"subject": subject, "messages": messages, "stats": stats}


def thread_to_text(parsed):
    out = "Subject: " + parsed["subject"] + "\n\n"
    for i in range(len(parsed["messages"])):
        m = parsed["messages"][i]
        out += "Message " + str(i + 1) + " (" + m["sender"] + "):\n" + m["text"] + "\n\n"
    return out.strip()


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as f:
        parsed = parse_thread(f.read())
    print(json.dumps(parsed, indent=2))
    print("\n" + thread_to_text(parsed))