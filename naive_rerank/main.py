import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import N_EVAL_USERS, N_NEGATIVES, RANDOM_SEED, TOP_K
from data import load_movies, load_ratings, leave_one_out_split
from evaluate import hit_at_k, mrr, ndcg_at_k
from rerank import rank_candidates


def main(n_eval_users: int = N_EVAL_USERS, out_path: str | None = None):
    rng = np.random.default_rng(RANDOM_SEED)

    ratings = load_ratings()
    movies = load_movies()
    title_map = dict(zip(movies.movieId, movies.title))

    train, test = leave_one_out_split(ratings, RANDOM_SEED)
    item_ids = sorted(ratings.movieId.unique())

    eval_users = rng.choice(
        test.userId.values, size=min(n_eval_users, len(test)), replace=False
    )

    results = []
    for uid in tqdm(eval_users):
        u_train = train[train.userId == uid].sort_values("timestamp")
        history_titles = [title_map.get(mid, str(mid)) for mid in u_train.movieId]
        if not history_titles:
            continue

        target_row = test[test.userId == uid].iloc[0]
        target_id = int(target_row.movieId)

        rated_ids = set(u_train.movieId) | {target_id}
        neg_pool = [m for m in item_ids if m not in rated_ids]
        negatives = rng.choice(neg_pool, size=min(N_NEGATIVES, len(neg_pool)), replace=False)

        candidates = [(target_id, title_map.get(target_id, str(target_id)))]
        candidates += [(m, title_map.get(m, str(m))) for m in negatives]
        rng.shuffle(candidates)

        ranked = rank_candidates(history_titles, candidates)

        results.append({
            "userId": uid,
            "history_len": len(history_titles),
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