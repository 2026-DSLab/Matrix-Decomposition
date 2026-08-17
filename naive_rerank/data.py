import json

import pandas as pd

from config import LABELS_PATH, MOVIES_PATH, RATINGS_PATH


def load_ratings() -> pd.DataFrame:
    return pd.read_excel(RATINGS_PATH)


def load_movies() -> pd.DataFrame:
    return pd.read_csv(
        MOVIES_PATH, sep="::", engine="python",
        names=["movieId", "title", "genres"], encoding="latin-1",
    )


def load_labels() -> pd.DataFrame:
    """preprocess_code/build_label_file.py가 만든 고정 후보(20개) 라벨 파일 로드."""
    df = pd.read_excel(LABELS_PATH)
    df["candidate_movieIds"] = df["candidate_movieIds"].apply(json.loads)
    df["candidate_titles"] = df["candidate_titles"].apply(json.loads)
    return df


def leave_one_out_split(ratings: pd.DataFrame, seed: int = 42):
    """유저별 가장 최근 상호작용 1개를 test label로 분리."""
    test_rows = ratings.sort_values("timestamp").groupby("userId").tail(1)
    train = ratings.drop(test_rows.index)
    return train.reset_index(drop=True), test_rows.reset_index(drop=True)