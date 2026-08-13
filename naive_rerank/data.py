import pandas as pd

from config import MOVIES_PATH, RATINGS_PATH


def load_ratings() -> pd.DataFrame:
    return pd.read_excel(RATINGS_PATH)


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