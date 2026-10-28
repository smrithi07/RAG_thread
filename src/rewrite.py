import json

import config
from src.parse_thread import thread_to_text
from src.llm import call_gemini


def read_prompt(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def rewrite_thread(parsed):
    thread_text = thread_to_text(parsed)
    prompt = read_prompt(config.REWRITE_PROMPT).replace("{thread}", thread_text)

    try:
        data = json.loads(call_gemini(prompt, json_mode=True))
        questions = []
        for q in data["questions"]:
            if isinstance(q, str) and q.strip() != "":
                questions.append(q.strip())
        if len(questions) == 0 and len(parsed["messages"]) > 0:
            last = parsed["messages"][-1]["text"].strip()
            if "?" in last:
                return {"questions": [last], "method": "fallback-empty", "thread_text": thread_text}
        return {"questions": questions, "method": "gemini", "thread_text": thread_text}
    except Exception as e:
        print("  Rewrite failed, using last message:", e)

    questions = []
    if len(parsed["messages"]) > 0:
        last = parsed["messages"][-1]["text"].strip()
        if last != "":
            questions.append(last)
    return {"questions": questions, "method": "fallback", "thread_text": thread_text}
