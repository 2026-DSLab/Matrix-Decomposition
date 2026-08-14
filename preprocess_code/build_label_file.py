import argparse
import json
import os

import numpy as np
import pandas as pd

DEFAULT_RATINGS_PATH = "preprocessed/ratings_min300_500users.xlsx"
DEFAULT_RAW_DIR = "raw_data/ml-10M100K"
DEFAULT_OUT_PATH = "preprocessed/labels_500users.xlsx"

N_NEGATIVES = 19
RANDOM_SEED = 42


def load_movies(raw_dir: str) -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(raw_dir, "movies.dat"), sep="::", engine="python",
        names=["movieId", "title", "genres"], encoding="latin-1",
    )


def leave_one_out_split(ratings: pd.DataFrame):
    """유저별 가장 최근 상호작용 1개를 정답 레이블로 분리."""
    test_rows = ratings.sort_values("timestamp").groupby("userId").tail(1)
    train = ratings.drop(test_rows.index)
    return train.reset_index(drop=True), test_rows.reset_index(drop=True)


def build_label_file(ratings_path: str, raw_dir: str, n_negatives: int, seed: int) -> pd.DataFrame:
    ratings = pd.read_excel(ratings_path)
    movies = load_movies(raw_dir)
    title_map = dict(zip(movies.movieId, movies.title))
    genre_map = dict(zip(movies.movieId, movies.genres))

    def title_with_genre(mid: int) -> str:
        title = title_map.get(mid, "")
        genres = genre_map.get(mid, "").replace("|", ", ")
        return f"{title} ({genres})"

    train, test = leave_one_out_split(ratings)
    item_ids = sorted(ratings["movieId"].unique())

    rng = np.random.default_rng(seed)

    rows = []
    for uid in sorted(ratings["userId"].unique()):
        u_train = train[train["userId"] == uid]
        target_row = test[test["userId"] == uid]
        if target_row.empty:
            continue
        target_id = int(target_row.iloc[0]["movieId"])

        rated_ids = set(u_train["movieId"]) | {target_id}
        neg_pool = [m for m in item_ids if m not in rated_ids]
        negatives = rng.choice(neg_pool, size=min(n_negatives, len(neg_pool)), replace=False)

        candidates = [target_id] + [int(m) for m in negatives]
        rng.shuffle(candidates)
        answer_position = candidates.index(target_id) + 1

        row = {
            "userId": uid,
            "candidate_titles": json.dumps([title_with_genre(mid) for mid in candidates], ensure_ascii=False),
            "candidate_movieIds": json.dumps(candidates),
            "answer_movieId": target_id,
            "answer_title": title_map.get(target_id, ""),
            "answer_position": answer_position,
        }
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="유저별 20개 후보(정답 1 + negative 19, 셔플) 라벨링 파일 생성")
    parser.add_argument("--ratings-path", type=str, default=DEFAULT_RATINGS_PATH)
    parser.add_argument("--raw-dir", type=str, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT_PATH)
    parser.add_argument("--n-negatives", type=int, default=N_NEGATIVES)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    df = build_label_file(args.ratings_path, args.raw_dir, args.n_negatives, args.seed)
    df.to_excel(args.out, index=False, engine="openpyxl")
    print(f"{args.out}: users={len(df)}, candidates per user={args.n_negatives + 1}")