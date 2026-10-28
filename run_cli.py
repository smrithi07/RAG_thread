import sys

from src.parse_thread import thread_to_text
from src.pipeline import run_pipeline
from src import retrieve

USAGE = 'Usage: python run_cli.py <thread_file> [--debug]\n   or: python run_cli.py --text "your question" [--debug]'


def print_ranked(ranked, n):
    for i in range(min(n, len(ranked))):
        idx, score = ranked[i]
        c = retrieve.chunks[idx]
        print("    " + format(score, ".4f") + "  " + c["page"] + " § " + c["section"] + " | " + c["body"][:80])


def main():
    args = sys.argv[1:]
    debug = "--debug" in args
    if debug:
        args.remove("--debug")

    if len(args) == 2 and args[0] == "--text":
        text = args[1]
    elif len(args) == 1 and args[0] != "--text":
        with open(args[0], encoding="utf-8") as f:
            text = f.read()
    else:
        print(USAGE)
        sys.exit(1)

    try:
        out = run_pipeline(text)
    except Exception as e:
        print("Pipeline failed: " + str(e)[:300])
        sys.exit(1)

    print("=== Cleaned thread ===")
    print(thread_to_text(out["parsed"]))
    print("\n=== Parser stats ===")
    print(out["parsed"]["stats"])
    print("\n=== Rewrite (" + out["rewrite"]["method"] + ") ===")
    questions = out["rewrite"]["questions"]
    if len(questions) == 0:
        print("  No questions found.")
    for i in range(len(questions)):
        print("  " + str(i + 1) + ". " + questions[i])

    for i in range(len(out["results"])):
        r = out["results"][i]
        ret = r["retrieval"]
        ans = r["answer"]
        print("\n=== Question " + str(i + 1) + ": " + r["question"] + " ===")
        print("  Entities:", ret["entities"])
        if debug:
            stages = ["dense", "bm25", "rrf", "mmr"]
            for s in range(len(stages)):
                print("  " + stages[s] + " top 3:")
                print_ranked(ret[stages[s]], 3)
        print("  Final top 5:")
        print_ranked(ret["final"], 5)
        print("  Best score:", round(ret["best_score"], 4))
        print("\n  Answer:", ans["answer"])
        if len(ans["sources"]) > 0:
            print("  Sources:")
            for j in range(len(ans["sources"])):
                src = ans["sources"][j]
                # nothing cited (e.g. LLM unavailable): show all retrieved sources
                if len(ans["cited"]) == 0 or src["n"] in ans["cited"]:
                    print("    [" + str(src["n"]) + "] " + src["page"] + " § " + src["section"] + " - " + src["url"])
        if ans.get("invalid_citations", 0) > 0:
            print("  Invalid citations:", ans["invalid_citations"])

    print("\nTotal seconds:", round(out["seconds"], 1))


if __name__ == "__main__":
    main()
