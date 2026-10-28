import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
INDEX_DIR = os.path.join(DATA_DIR, "index")
PROMPT_DIR = os.path.join(BASE_DIR, "prompts")
PAGES_FILE = os.path.join(DATA_DIR, "pages.txt")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

SHINGLE_SIZE = 3
NUM_HASHES = 64
LSH_BANDS = 8
DUP_THRESHOLD = 0.8
MIN_PARA_WORDS = 5

CHUNK_WORDS = 200
CHUNK_OVERLAP = 30
CHUNKS_FILE = os.path.join(INDEX_DIR, "chunks.json")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBED_FILE = os.path.join(INDEX_DIR, "embeddings.npy")
META_FILE = os.path.join(INDEX_DIR, "index_meta.json")
BM25_K1 = 1.5
BM25_B = 0.75
DENSE_TOP_K = 50
BM25_TOP_K = 50

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60
RRF_TOP = 30
MMR_LAMBDA = 0.7
MMR_TOP = 15
FINAL_TOP = 5
NOT_FOUND_THRESHOLD = 0.2
SUMMARY_PAGE = "States and union territories of India"
ALIASES_FILE = os.path.join(DATA_DIR, "aliases.json")

REWRITE_PROMPT = os.path.join(PROMPT_DIR, "rewrite.txt")
ANSWER_PROMPT = os.path.join(PROMPT_DIR, "answer.txt")
GEMINI_TEMPERATURE = 0.0
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "")

EVAL_DIR = os.path.join(DATA_DIR, "eval")
TEST_FILE = os.path.join(DATA_DIR, "test_threads.json")
EVAL_K = 5
EVAL_MRR_K = 10
EVAL_SLEEP = 4
