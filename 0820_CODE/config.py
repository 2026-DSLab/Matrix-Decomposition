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

# ml-latest-small: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
# ml-latest (전체): https://files.grouplens.org/datasets/movielens/ml-latest.zip
DATA_DIR = "data/ml10m-500users"  # 공유 전처리 데이터(MovieLens 10M, interaction 300+ / 500 users) -- prepare_0820_data.py로 생성
MIN_HISTORY_LEN = 300  # 이 값 미만으로 평점을 남긴 유저는 제외 (긴 이력 유저만 평가)
MAX_POOL_USERS = 1500  # heavy user 중 행렬 계산에 실제로 쓸 유저 수 상한 (dense 행렬 크기 제어)

FACTOR_METHODS = ["svd", "nmf", "fa"]  # mf.FACTORIZERS 중 비교할 것들
# FACTOR_METHODS = ["svd", "nmf", "fa", "genre"]  # genre 포함 버전 -- 다시 쓰려면 이 줄로 바꾸고
# main.py의 item_genre_matrix / genre_kwargs 주석도 함께 해제할 것
N_FACTORS = 5  # 잠재 축 개수 (CLI --n-factors로 오버라이드 가능)
MAX_RETRIES = 5  # LLM 호출 실패 시 최대 시도 횟수 (지수 백오프: 2, 4, 8, 16초)
RETRY_BACKOFF_SEC = 2
AXIS_PERSONA_N = 25  # 축 페르소나 생성 시 양 끝에서 각각 뽑을 영화 수 (25+25=50편을 LLM에 보여줌)
# 축 엔지니어링은 기본 비활성(0). 켜면 기법마다 채택 개수가 달라져(svd 4개 / nmf 1개 / fa 1개)
# 축이 비대칭으로 바뀐다 -- 실측에서 svd는 엔지니어링 전후 축 상관이 [0.99, 0.14, 0.15, 0.17, 0.08]로
# 축 2~5가 사실상 다른 축이 된 반면 nmf/fa는 0.99~1.00으로 그대로였다. 게다가 svd에서 채택된 4개 중
# 3개는 잔차를 오히려 늘렸다(+0.6%, +0.7%, +0.2%). 기법 비교가 목적이므로 세 기법 모두 순수 분해
# 축을 쓰도록 끄고, 엔지니어링 효과를 보고 싶으면 이 값을 올려 별도 조건으로 돌린다.
MAX_ENGINEERED_AXES = 0  # 축 엔지니어링 라운드 상한(안전장치) -- 언제 멈출지는 LLM이 CONTINUE/STOP으로 결정하며, 이 값은 LLM이 계속 CONTINUE를 골라도 과다 라운드/API 비용으로 새지 않게 막는 안전망
N_NEGATIVES = 19
TOP_K = 5
N_EVAL_USERS = 500  # Inference User N -- 공유 데이터의 500명 전원 평가
RANDOM_SEED = 42
