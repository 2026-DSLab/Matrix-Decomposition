import json

import pandas as pd

from config import LABELS_PATH, MOVIES_PATH, RATINGS_PATH


def load_ratings() -> pd.DataFrame:
    return pd.read_excel(RATINGS_PATH)


def load_labels():
    """공용 라벨 파일에서 유저별 정답과 후보 20개를 읽는다.

    반환: (answer, candidates) -- 각각 {userId: 정답 movieId}, {userId: [movieId 20개]}.
    후보는 파일에 저장된 순서를 그대로 쓴다(생성 시점에 이미 셔플되어 있어 다시 섞으면
    다른 실험과 순서가 어긋난다)."""
    df = pd.read_excel(LABELS_PATH)
    answer = {int(r.userId): int(r.answer_movieId) for r in df.itertuples()}
    candidates = {int(r.userId): json.loads(r.candidate_movieIds) for r in df.itertuples()}
    return answer, candidates


def split_by_labels(ratings: pd.DataFrame, answer: dict) -> pd.DataFrame:
    """라벨 파일이 지정한 정답 1개를 유저마다 빼고 나머지를 이력으로 쓴다.

    정답이 이력에 남아 있으면 프롬프트에 그대로 노출되어 누출이 되므로 반드시 제거한다."""
    ans = pd.DataFrame({"userId": list(answer), "movieId": list(answer.values()), "_ans": 1})
    merged = ratings.merge(ans, on=["userId", "movieId"], how="left")
    return merged[merged["_ans"].isna()].drop(columns=["_ans"]).reset_index(drop=True)


def load_movies() -> pd.DataFrame:
    return pd.read_csv(
        MOVIES_PATH, sep="::", engine="python",
        names=["movieId", "title", "genres"], encoding="latin-1",
    )


def leave_one_out_split(ratings: pd.DataFrame, seed: int = 42):
    """유저별 가장 최근 상호작용 1개를 test label로 분리."""
    test_rows = ratings.sort_values("timestamp").groupby("userId").tail(1)
    train = ratings.drop(test_rows.index)
    return train.reset_index(drop=True), test_rows.reset_index(drop=True)