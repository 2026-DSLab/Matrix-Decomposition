import argparse
import os

import pandas as pd


def load_ratings(raw_dir: str) -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(raw_dir, "ratings.dat"), sep="::", engine="python",
        names=["userId", "movieId", "rating", "timestamp"], encoding="latin-1",
    )


def filter_by_min_interactions(ratings: pd.DataFrame, min_interactions: int, max_interactions: int | None = None) -> pd.DataFrame:
    """상호작용(평점) 개수가 min_interactions 이상(max_interactions 지정 시 그 이하)인 유저의 평점만 남김."""
    counts = ratings.groupby("userId").size()
    mask = counts >= min_interactions
    if max_interactions is not None:
        mask &= counts <= max_interactions
    qualified_users = counts[mask].index
    return ratings[ratings["userId"].isin(qualified_users)].sort_values(["userId", "timestamp"])


def sample_users(ratings: pd.DataFrame, n_users: int, seed: int) -> pd.DataFrame:
    """ratings에 포함된 유저 중 n_users명을 무작위 샘플링해 그 유저들의 평점만 남김."""
    users = pd.Series(ratings["userId"].unique())
    sampled = set(users.sample(n=n_users, random_state=seed).tolist())
    return ratings[ratings["userId"].isin(sampled)]


def main(min_interactions: int, max_interactions: int | None, n_sample_users: int, seed: int, raw_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    ratings = load_ratings(raw_dir)
    filtered = filter_by_min_interactions(ratings, min_interactions, max_interactions)

    suffix = f"min{min_interactions}" + (f"_max{max_interactions}" if max_interactions is not None else "")
    csv_path = os.path.join(out_dir, f"ratings_{suffix}.csv")
    filtered.to_csv(csv_path, index=False)
    print(f"[1] {csv_path}: users={filtered['userId'].nunique()}, rows={len(filtered)}")

    sampled = sample_users(filtered, n_sample_users, seed)
    xlsx_path = os.path.join(out_dir, f"ratings_{suffix}_{n_sample_users}users.xlsx")
    sampled.to_excel(xlsx_path, index=False, engine="openpyxl")
    print(f"[2] {xlsx_path}: users={sampled['userId'].nunique()}, rows={len(sampled)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MovieLens 10M 상호작용 필터링 전처리")
    parser.add_argument("--min-interactions", type=int, default=300)
    parser.add_argument("--max-interactions", type=int, default=None)
    parser.add_argument("--n-sample-users", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-dir", type=str, default="raw_data/ml-10M100K")
    parser.add_argument("--out-dir", type=str, default="preprocessed")
    args = parser.parse_args()
    main(args.min_interactions, args.max_interactions, args.n_sample_users, args.seed, args.raw_dir, args.out_dir)