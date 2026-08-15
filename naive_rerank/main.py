import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import N_EVAL_USERS, RANDOM_SEED, TOP_K
from data import load_labels, load_movies, load_ratings, split_by_labels
from evaluate import hit_at_k, mrr, ndcg_at_k
from rerank import rank_candidates


def main(n_eval_users: int = N_EVAL_USERS, out_path: str | None = None):
    rng = np.random.default_rng(RANDOM_SEED)

    ratings = load_ratings()
    movies = load_movies()
    title_map = dict(zip(movies.movieId, movies.title))

    # 후보 20개와 정답은 공용 라벨 파일에서 그대로 가져온다 -- 실행할 때마다 negative를 뽑으면
    # 같은 유저라도 후보가 달라져 다른 실험과 숫자를 나란히 놓을 수 없다. 정답도 이 파일 기준으로
    # 잡는다: timestamp가 동점인 유저가 500명 중 32명 있어서 "가장 최근 1개"가 정렬 방식에 따라
    # 달라지기 때문이다.
    answer, candidate_ids = load_labels()
    train = split_by_labels(ratings, answer)
    print(f"labels: users={len(answer)}, candidates per user={len(next(iter(candidate_ids.values())))}")

    label_users = np.array(sorted(answer))
    eval_users = rng.choice(
        label_users, size=min(n_eval_users, len(label_users)), replace=False
    )

    results = []
    for uid in tqdm(eval_users):
        u_train = train[train.userId == uid].sort_values("timestamp")
        history_titles = [title_map.get(mid, str(mid)) for mid in u_train.movieId]
        if not history_titles:
            continue

        target_id = answer[int(uid)]
        candidates = [(m, title_map.get(m, str(m))) for m in candidate_ids[int(uid)]]

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