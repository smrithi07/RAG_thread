import os

import pandas as pd
import streamlit as st

import config
from src import retrieve
from src.pipeline import run_pipeline

SAMPLE_DIR = os.path.join(config.DATA_DIR, "sample_threads")
STAGES = ["dense", "bm25", "rrf", "mmr", "final"]
STAT_NAMES = ["quoted_lines", "header_lines", "signature_lines", "duplicate_paragraphs"]


@st.cache_resource(show_spinner="Loading models and index...")
def load_models():
    retrieve.load_resources()
    return True


def stage_table(ranked):
    rows = []
    for i in range(min(5, len(ranked))):
        idx, score = ranked[i]
        c = retrieve.chunks[idx]
        rows.append({"rank": i + 1, "score": round(score, 4), "page": c["page"], "section": c["section"],
                     "type": c["type"], "text": c["body"][:100]})
    return pd.DataFrame(rows)


def show_sources(answer):
    shown = []
    for n in answer["cited"]:
        for s in answer["sources"]:
            if s["n"] == n:
                shown.append(s)
    if len(shown) == 0:
        shown = answer["sources"]
    for s in shown:
        st.markdown("\\[" + str(s["n"]) + "\\] [" + s["page"] + " § " + s["section"] + "](" + s["url"] + ")")


st.set_page_config(page_title="RAG_thread", layout="wide")
st.title("RAG_thread: Email Thread QA over Wikipedia (Indian States & UTs)")
st.caption("parse thread -> Gemini rewrite -> dense + BM25 -> RRF -> MMR -> cross-encoder -> grounded answer")
load_models()

st.sidebar.header("Input")
mode = st.sidebar.radio("Input mode", ["Sample thread", "Paste text", "Upload .txt"])
text = ""
if mode == "Sample thread":
    names = []
    for name in sorted(os.listdir(SAMPLE_DIR)):
        if name.endswith(".txt"):
            names.append(name)
    choice = st.sidebar.selectbox("Sample", names)
    with open(os.path.join(SAMPLE_DIR, choice), encoding="utf-8") as f:
        text = f.read()
    st.sidebar.text_area("Thread", text, height=300, disabled=True)
elif mode == "Paste text":
    text = st.sidebar.text_area("Paste a thread or a question", height=300)
else:
    upload = st.sidebar.file_uploader("Upload a thread", type=["txt"])
    if upload is not None:
        text = upload.getvalue().decode("utf-8")

st.sidebar.divider()
st.sidebar.caption("Gemini model: " + config.GEMINI_MODEL)
st.sidebar.caption("Fallback model: " + getattr(config, "GEMINI_FALLBACK_MODEL", "none"))
st.sidebar.caption("Not-found threshold: " + str(config.NOT_FOUND_THRESHOLD))
show_steps = st.sidebar.checkbox("Show retrieval steps", value=True)

if st.button("Run", type="primary"):
    if text.strip() == "":
        st.warning("Please choose, paste or upload a thread first.")
        st.stop()
    with st.spinner("Running pipeline..."):
        try:
            # kept in session_state so results survive reruns (e.g. toggling the checkbox)
            st.session_state["out"] = run_pipeline(text)
        except Exception as e:
            st.error("Pipeline failed: " + str(e)[:300])
            st.stop()

if "out" not in st.session_state:
    st.stop()
out = st.session_state["out"]

st.subheader("1. Cleaned thread")
left, right = st.columns([3, 1])
left.code(out["rewrite"]["thread_text"], language=None, wrap_lines=True)
for name in STAT_NAMES:
    right.metric(name, out["parsed"]["stats"][name])

st.subheader("2. Standalone questions")
questions = out["rewrite"]["questions"]
if len(questions) == 0:
    st.info("No questions found in the thread.")
lines = []
for i in range(len(questions)):
    lines.append(str(i + 1) + ". " + questions[i])
st.markdown("\n".join(lines))
st.caption("method: " + out["rewrite"]["method"])

results = out["results"]
for i in range(len(results)):
    ret = results[i]["retrieval"]
    ans = results[i]["answer"]
    st.subheader("Q" + str(i + 1) + ": " + results[i]["question"])
    if len(ret["entities"]) > 0:
        st.caption("Entities: " + ", ".join(ret["entities"]))
    else:
        st.caption("Entities: none")

    # escape $ so amounts like "US$ 10" are not rendered as LaTeX
    answer_text = ans["answer"].replace("$", "\\$")
    if ans["found"]:
        st.success(answer_text)
    else:
        st.warning(answer_text)
    st.caption("Best re-rank score: " + format(ret["best_score"], ".4f"))
    show_sources(ans)

    if show_steps:
        with st.expander("Retrieval steps"):
            for stage in STAGES:
                st.markdown("**" + stage + "**")
                st.dataframe(stage_table(ret[stage]), hide_index=True)

st.caption("Total seconds: " + str(round(out["seconds"], 1)))
