import time

from src.parse_thread import parse_thread
from src.rewrite import rewrite_thread
from src import retrieve
from src.answer import answer_question


def run_pipeline(text):
    start = time.time()
    parsed = parse_thread(text)
    rewrite = rewrite_thread(parsed)

    retrieve.load_resources()
    results = []
    questions = rewrite["questions"]
    for i in range(len(questions)):
        result = retrieve.retrieve(questions[i])
        answer = answer_question(questions[i], result, retrieve.chunks)
        results.append({"question": questions[i], "retrieval": result, "answer": answer})

    return {"parsed": parsed, "rewrite": rewrite, "results": results, "seconds": time.time() - start}
