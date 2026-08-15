import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")  # 저장소 공용 키
load_dotenv(BASE_DIR / ".env", override=True)  # 이 실험만 다른 키/모델을 쓸 때

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
# 연구실 네트워크가 TLS를 가로채므로(사설 루트 CA: kookmin.ac.kr/KMU) 인증서 검증을 끈다.
# 검증이 되는 망에서는 .env에 VERIFY_SSL=true를 넣으면 정상 검증으로 돌아감.
VERIFY_SSL = os.environ.get("VERIFY_SSL", "false").lower() == "true"

RATINGS_PATH = PROJECT_ROOT / "preprocessed" / "ratings_min300_500users.xlsx"
MOVIES_PATH = PROJECT_ROOT / "raw_data" / "ml-10M100K" / "movies.dat"
# 유저별 후보 20개(정답 1 + negative 19)와 정답을 고정한 공용 라벨 파일.
# build_label_file.py가 만든 이 파일을 그대로 읽어야 다른 실험과 후보가 일치한다.
LABELS_PATH = PROJECT_ROOT / "preprocessed" / "labels_500users.xlsx"

N_NEGATIVES = 19  # candidates = 1 test label + N_NEGATIVES
TOP_K = 5
N_EVAL_USERS = 20
RANDOM_SEED = 42