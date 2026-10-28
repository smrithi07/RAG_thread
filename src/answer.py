import re

import config
from src.llm import call_gemini
from src.rewrite import read_prompt

NOT_FOUND_TEXT = "Not found in the Wikipedia pages on Indian states and union territories."


def not_found():
    return {"answer": NOT_FOUND_TEXT, "found": False, "sources": [], "cited": []}


def build_context(final, chunks):
    lines = []
    sources = []
    for i in range(len(final)):
        idx, score = final[i]
        c = chunks[idx]
        n = i + 1
        lines.append("[" + str(n) + "] (" + c["page"] + " § " + c["section"] + ", " + c["type"] + ") " + c["body"])
        sources.append({"n": n, "page": c["page"], "section": c["section"], "type": c["type"],
                        "url": c["url"], "score": score})
    return "\n\n".join(lines), sources


def answer_question(question, result, chunks):
    if not result["found"]:
        return not_found()

    context, sources = build_context(result["final"], chunks)
    prompt = read_prompt(config.ANSWER_PROMPT).replace("{question}", question).replace("{context}", context)
    text = call_gemini(prompt).strip()
    if text == "NOT_FOUND":
        return not_found()

    cited = []
    invalid = 0
    numbers = re.findall(r"\[(\d+)\]", text)
    for i in range(len(numbers)):
        n = int(numbers[i])
        if n < 1 or n > len(sources):
            invalid += 1
        elif n not in cited:
            cited.append(n)
    return {"answer": text, "found": True, "sources": sources, "cited": cited, "invalid_citations": invalid}
