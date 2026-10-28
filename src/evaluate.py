import os
import re
import sys
import json
import math
import time

import pandas as pd

import config
from src.index import load_chunks

STAGES = ["dense", "bm25", "rrf", "mmr", "final"]
TAUS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
RESULTS_FILE = os.path.join(config.EVAL_DIR, "run_results.json")
METRICS_FILE = os.path.join(config.EVAL_DIR, "metrics.json")
CSV_FILE = os.path.join(config.EVAL_DIR, "per_case.csv")


def average(total, count):
    if count == 0:
        return 0
    return total / count


def ids_only(ranked):
    ids = []
    for i in range(len(ranked)):
        ids.append(ranked[i][0])
    return ids


def run_all(cases):
    # imported here so --reuse does not load the models
    from src import retrieve
    from src.pipeline import run_pipeline
    retrieve.load_resources()

    runs = []
    for i in range(len(cases)):
        case = cases[i]
        run = {"id": case["id"], "error": "", "method": "", "questions": []}
        start = time.time()
        try:
            out = run_pipeline(case["thread"])
            run["method"] = out["rewrite"]["method"]
            for r in out["results"]:
                ret = r["retrieval"]
                ans = r["answer"]
                q = {"question": r["question"], "entities": ret["entities"], "best_score": ret["best_score"],
                     "found": ret["found"]}
                for stage in STAGES:
                    q[stage] = ids_only(ret[stage])
                q["answer"] = ans["answer"]
                q["answer_found"] = ans["found"]
                q["cited"] = ans["cited"]
                q["invalid_citations"] = ans.get("invalid_citations", 0)
                run["questions"].append(q)
        except Exception as e:
            run["error"] = str(e)

        status = str(len(run["questions"])) + " question(s)"
        if run["error"] != "":
            status = "ERROR " + run["error"][:80]
        print("[" + str(i + 1) + "/" + str(len(cases)) + "] " + case["id"] + " " + case["category"] + ": "
              + status + ", " + str(round(time.time() - start, 1)) + " s")
        runs.append(run)
        if i < len(cases) - 1:
            time.sleep(config.EVAL_SLEEP)

    os.makedirs(config.EVAL_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(runs, f, ensure_ascii=False, indent=1)
    return runs


def is_relevant(chunk, gold_item):
    if chunk["page"] not in gold_item["pages"]:
        return False
    body = chunk["body"].lower()
    for kw in gold_item["keywords"]:
        if kw.lower() not in body:
            return False
    return True


def count_relevant(cases, chunks):
    counts = {}
    for case in cases:
        counts[case["id"]] = []
        for g in range(len(case["gold"])):
            n = 0
            for c in chunks:
                if is_relevant(c, case["gold"][g]):
                    n += 1
            counts[case["id"]].append(n)
            if n == 0:
                print("WARNING: " + case["id"] + " gold " + str(g) + " has no relevant chunk in the corpus:",
                      case["gold"][g])
    return counts


def first_rank(ids, chunks, gold_item):
    for i in range(len(ids)):
        if is_relevant(chunks[ids[i]], gold_item):
            return i + 1
    return None


def ndcg(ids, chunks, gold_item, R, k):
    dcg = 0
    for i in range(min(k, len(ids))):
        if is_relevant(chunks[ids[i]], gold_item):
            dcg += 1 / math.log2(i + 2)
    idcg = 0
    for i in range(min(R, k)):
        idcg += 1 / math.log2(i + 2)
    if idcg == 0:
        return 0
    return dcg / idcg


def retrieval_metrics(cases, runs, chunks, R):
    metrics = {}
    by_category = {}
    for stage in STAGES:
        recall_sum = 0
        mrr_sum = 0
        ndcg_sum = 0
        count = 0
        for case in cases:
            if not case["answerable"]:
                continue
            questions = runs[case["id"]]["questions"]
            for g in range(len(case["gold"])):
                gold = case["gold"][g]
                best_rank = None
                best_ndcg = 0
                for q in questions:
                    rank = first_rank(q[stage], chunks, gold)
                    if rank is not None and (best_rank is None or rank < best_rank):
                        best_rank = rank
                    n = ndcg(q[stage], chunks, gold, R[case["id"]][g], config.EVAL_K)
                    if n > best_ndcg:
                        best_ndcg = n

                recall = 0
                if best_rank is not None and best_rank <= config.EVAL_K:
                    recall = 1
                mrr = 0
                if best_rank is not None and best_rank <= config.EVAL_MRR_K:
                    mrr = 1 / best_rank
                recall_sum += recall
                mrr_sum += mrr
                ndcg_sum += best_ndcg
                count += 1

                if stage == "final":
                    cat = case["category"]
                    if cat not in by_category:
                        by_category[cat] = [0, 0]
                    by_category[cat][0] += recall
                    by_category[cat][1] += 1
        metrics[stage] = {"recall@" + str(config.EVAL_K): average(recall_sum, count),
                          "mrr@" + str(config.EVAL_MRR_K): average(mrr_sum, count),
                          "ndcg@" + str(config.EVAL_K): average(ndcg_sum, count), "gold_items": count}

    category_recall = {}
    for cat in by_category:
        category_recall[cat] = average(by_category[cat][0], by_category[cat][1])
    return metrics, category_recall


def case_score(run):
    best = 0
    for q in run["questions"]:
        if q["best_score"] > best:
            best = q["best_score"]
    return best


def case_answered(run):
    for q in run["questions"]:
        if q["answer_found"]:
            return True
    return False


def decision_metrics(cases, runs):
    counts = {"answerable_answered": 0, "answerable_missed": 0, "unanswerable_rejected": 0,
              "unanswerable_answered": 0}
    for case in cases:
        answered = case_answered(runs[case["id"]])
        if case["answerable"] and answered:
            counts["answerable_answered"] += 1
        elif case["answerable"]:
            counts["answerable_missed"] += 1
        elif answered:
            counts["unanswerable_answered"] += 1
        else:
            counts["unanswerable_rejected"] += 1
    correct = counts["answerable_answered"] + counts["unanswerable_rejected"]

    sweep = {}
    for tau in TAUS:
        right = 0
        for case in cases:
            predicted = case_score(runs[case["id"]]) >= tau
            if predicted == case["answerable"]:
                right += 1
        sweep[str(tau)] = average(right, len(cases))
    return {"accuracy": average(correct, len(cases)), "counts": counts, "threshold_sweep": sweep}


def joined_answers(run):
    texts = []
    for q in run["questions"]:
        texts.append(q["answer"])
    return " ".join(texts).lower()


def has_keywords(case, run):
    text = joined_answers(run)
    for kw in case["answer_keywords"]:
        if kw.lower() not in text:
            return False
    return True


def clean_number(n):
    return n.replace(",", "").rstrip(".")


def answer_metrics(cases, runs, chunks):
    cases_checked = 0
    cases_correct = 0
    answered_questions = 0
    with_citation = 0
    invalid = 0
    numbers_total = 0
    numbers_grounded = 0
    for case in cases:
        run = runs[case["id"]]
        if not case["answerable"] or not case_answered(run):
            continue
        cases_checked += 1
        if has_keywords(case, run):
            cases_correct += 1

        for q in run["questions"]:
            if not q["answer_found"]:
                continue
            answered_questions += 1
            if len(q["cited"]) > 0:
                with_citation += 1
            invalid += q["invalid_citations"]

            text = re.sub(r"\[\d+\]", "", q["answer"])
            numbers = re.findall(r"\d[\d,]*\.?\d*", text)
            if len(numbers) == 0:
                continue
            bodies = []
            for n in q["cited"]:
                if n - 1 < len(q["final"]):
                    bodies.append(chunks[q["final"][n - 1]]["body"])
            evidence = " ".join(bodies).replace(",", "")
            for num in numbers:
                numbers_total += 1
                if clean_number(num) in evidence:
                    numbers_grounded += 1

    return {"cases_checked": cases_checked, "answer_accuracy": average(cases_correct, cases_checked),
            "citation_rate": average(with_citation, answered_questions), "invalid_citations": invalid,
            "numbers_grounded": numbers_grounded, "numbers_total": numbers_total,
            "number_grounding": average(numbers_grounded, numbers_total)}


def save_csv(cases, runs):
    rows = []
    for case in cases:
        run = runs[case["id"]]
        questions = []
        for q in run["questions"]:
            questions.append(q["question"])
        answered = case_answered(run)
        answer_correct = ""
        if case["answerable"]:
            answer_correct = answered and has_keywords(case, run)
        rows.append({"id": case["id"], "category": case["category"], "answerable": case["answerable"],
                     "n_questions": len(questions), "questions": " || ".join(questions),
                     "best_score": round(case_score(run), 4), "answered": answered,
                     "correct_decision": answered == case["answerable"], "answer_correct": answer_correct,
                     "answer": joined_answers(run)[:150]})
    pd.DataFrame(rows).to_csv(CSV_FILE, index=False)


def main():
    with open(config.TEST_FILE, encoding="utf-8") as f:
        cases = json.load(f)
    chunks = load_chunks()
    R = count_relevant(cases, chunks)

    if "--reuse" in sys.argv:
        with open(RESULTS_FILE, encoding="utf-8") as f:
            run_list = json.load(f)
        print("Reusing", RESULTS_FILE)
    else:
        run_list = run_all(cases)

    runs = {}
    for case in cases:
        runs[case["id"]] = {"id": case["id"], "error": "missing", "questions": []}
    for run in run_list:
        runs[run["id"]] = run

    retrieval, category_recall = retrieval_metrics(cases, runs, chunks, R)
    decision = decision_metrics(cases, runs)
    quality = answer_metrics(cases, runs, chunks)

    k = str(config.EVAL_K)
    mk = str(config.EVAL_MRR_K)
    print("\nRetrieval (" + str(retrieval["final"]["gold_items"]) + " gold items, answerable cases)")
    print("stage  | recall@" + k + " | MRR@" + mk + " | nDCG@" + k)
    for stage in STAGES:
        m = retrieval[stage]
        print(stage.ljust(6) + " | " + format(m["recall@" + k], ".3f").ljust(8) + " | "
              + format(m["mrr@" + mk], ".3f").ljust(6) + " | " + format(m["ndcg@" + k], ".3f"))
    print("\nFinal recall@" + k + " per category:")
    for cat in category_recall:
        print("  " + cat.ljust(16) + format(category_recall[cat], ".3f"))

    print("\nNot-found decision: accuracy", format(decision["accuracy"], ".3f"))
    for name in decision["counts"]:
        print("  " + name + ":", decision["counts"][name])
    print("Threshold sweep (case score >= tau means answerable):")
    for tau in decision["threshold_sweep"]:
        print("  tau " + tau.ljust(5) + " accuracy " + format(decision["threshold_sweep"][tau], ".3f"))

    print("\nAnswer quality (" + str(quality["cases_checked"]) + " answered answerable cases)")
    print("  answer_accuracy:", format(quality["answer_accuracy"], ".3f"))
    print("  citation_rate:", format(quality["citation_rate"], ".3f"))
    print("  invalid_citations:", quality["invalid_citations"])
    print("  number grounding:", str(quality["numbers_grounded"]) + "/" + str(quality["numbers_total"]),
          "(" + format(quality["number_grounding"], ".3f") + ")")

    errors = 0
    for case in cases:
        if runs[case["id"]]["error"] != "":
            errors += 1
    os.makedirs(config.EVAL_DIR, exist_ok=True)
    metrics = {"retrieval": retrieval, "final_recall_by_category": category_recall, "decision": decision,
               "answer_quality": quality, "cases": len(cases), "errors": errors}
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=1)
    save_csv(cases, runs)
    print("\nSaved", METRICS_FILE, "and", CSV_FILE)


if __name__ == "__main__":
    main()
