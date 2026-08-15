"""선배와 공유하는 전처리 데이터(interaction 300+ / 500 users)를 axis_rerank가 읽는
ml-latest 형식(ratings.csv, movies.csv)으로 변환한다.

axis_rerank/data.py의 load_movielens()는 DATA_DIR 아래의 ratings.csv/movies.csv만 읽으므로,
같은 컬럼명으로 맞춰주면 코드 수정 없이 MovieLens 10M 기반 공유 데이터를 쓸 수 있다.
"""
import argparse
import os

import pandas as pd


def main(ratings_xlsx: str, movies_dat: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    ratings = pd.read_excel(ratings_xlsx)  # userId, movieId, rating, timestamp
    ratings.to_csv(os.path.join(out_dir, "ratings.csv"), index=False)
    print(
        f"ratings.csv: users={ratings.userId.nunique()}, movies={ratings.movieId.nunique()}, "
        f"rows={len(ratings)}"
    )

    movies = pd.read_csv(
        movies_dat, sep="::", engine="python",
        names=["movieId", "title", "genres"], encoding="latin-1",
    )
    movies.to_csv(os.path.join(out_dir, "movies.csv"), index=False)
    print(f"movies.csv: rows={len(movies)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="공유 전처리 데이터 -> axis_rerank 입력 형식 변환")
    parser.add_argument("--ratings-xlsx", default="preprocessed/ratings_min300_500users.xlsx")
    parser.add_argument("--movies-dat", default="raw_data/ml-10M100K/movies.dat")
    parser.add_argument("--out-dir", default="axis_rerank/data/ml10m-500users")
    args = parser.parse_args()
    main(args.ratings_xlsx, args.movies_dat, args.out_dir)
