import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import N_EVAL_USERS, RANDOM_SEED, TOP_K
from data import load_labels, load_movies, load_ratings, leave_one_out_split
from evaluate import hit_at_k, mrr, ndcg_at_k
from rerank import rank_candidates


def main(n_eval_users: int = N_EVAL_USERS, out_path: str | None = None):
    rng = np.random.default_rng(RANDOM_SEED)

    ratings = load_ratings()
    movies = load_movies()
    title_map = dict(zip(movies.movieId, movies.title))
    genre_map = dict(zip(movies.movieId, movies.genres))
    labels = load_labels()

    train, _ = leave_one_out_split(ratings, RANDOM_SEED)

    eval_users = rng.choice(
        labels.userId.values, size=min(n_eval_users, len(labels)), replace=False
    )

    results = []
    for uid in tqdm(eval_users):
        u_train = train[train.userId == uid].sort_values("timestamp")
        history = [
            {
                "title": title_map.get(row.movieId, str(row.movieId)),
                "genres": genre_map.get(row.movieId, "").replace("|", ", "),
                "rating": row.rating,
            }
            for row in u_train.itertuples()
        ]
        if not history:
            continue

        label_row = labels[labels.userId == uid].iloc[0]
        target_id = int(label_row.answer_movieId)
        candidates = list(zip(label_row.candidate_movieIds, label_row.candidate_titles))

        ranked = rank_candidates(history, candidates)

        results.append({
            "userId": uid,
            "history_len": len(history),
            "hit": hit_at_k(ranked, target_id, TOP_K),
            "mrr": mrr(ranked, target_id),
            "ndcg": ndcg_at_k(ranked, target_id, TOP_K),
        })

    df = pd.DataFrame(results)
    df.to_csv(out_path or "results.csv", index=False)
    print(f"[n_eval_users={n_eval_users}]")
    print(df[["hit", "mrr", "ndcg"]].mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-eval-users", type=int, default=N_EVAL_USERS)
    parser.add_argument("--out", type=str, default=None, help="output CSV path (default: results.csv)")
    args = parser.parse_args()
    main(args.n_eval_users, args.out)