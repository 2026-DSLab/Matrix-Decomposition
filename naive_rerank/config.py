import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
load_dotenv(BASE_DIR / ".env")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

RATINGS_PATH = PROJECT_ROOT / "preprocessed" / "ratings_min300_500users.xlsx"
MOVIES_PATH = PROJECT_ROOT / "raw_data" / "ml-10M100K" / "movies.dat"

N_NEGATIVES = 19  # candidates = 1 test label + N_NEGATIVES
TOP_K = 5
N_EVAL_USERS = 20
RANDOM_SEED = 42