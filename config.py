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
