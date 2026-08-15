import argparse
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from axis_engineering import engineer_axes_llm
from axis_persona import build_axis_personas
from config import (
    LABELS_PATH,
    PERSONAS_PATH,
    AXIS_PERSONA_N,
    FACTOR_METHODS,
    MAX_ENGINEERED_AXES,
    MAX_POOL_USERS,
    MODEL,
    MIN_HISTORY_LEN,
    N_EVAL_USERS,
    N_FACTORS,
    RANDOM_SEED,
    TOP_K,
)
from data import (
    # build_item_genre_matrix,  # genre 방법 전용 -- FACTOR_METHODS에서 제외되어 미사용
    build_rating_matrix,
    filter_users_by_history,
    split_by_labels,
    load_movielens,
)
from evaluate import hit_at_k, mrr, ndcg_at_k
from mf import top_items_per_factor, user_residual_summary
from profiling import profile_axis, profile_raw
from recommend import rank_candidates


def main(
    n_factors: int = N_FACTORS,
    n_eval_users: int = N_EVAL_USERS,
    out_path: str | None = None,
    min_history_len: int = MIN_HISTORY_LEN,
    max_history_len: int | None = None,
):
    rng = np.random.default_rng(RANDOM_SEED)
    ratings, movies = load_movielens()
    ratings = filter_users_by_history(
        ratings, min_history_len, max_history_len, max_users=MAX_POOL_USERS, seed=RANDOM_SEED
    )
    print(
        f"user pool (history in [{min_history_len}, {max_history_len or 'inf'}], "
        f"capped at {MAX_POOL_USERS}): {ratings.userId.nunique()}"
    )
    # 후보 20개와 정답은 공용 라벨 파일에서 그대로 가져온다 -- 실행할 때마다 negative를 뽑으면
    # 같은 유저라도 후보가 달라져 다른 실험과 숫자를 나란히 놓을 수 없다.
    labels = pd.read_excel(LABELS_PATH)
    answer = {int(r.userId): int(r.answer_movieId) for r in labels.itertuples()}
    candidate_ids = {int(r.userId): json.loads(r.candidate_movieIds) for r in labels.itertuples()}
    print(f"라벨 파일: {LABELS_PATH.name} -- 유저 {len(answer)}명, 유저당 후보 "
          f"{len(next(iter(candidate_ids.values())))}개")
    train = split_by_labels(ratings, answer)

    user_ids = sorted(ratings.userId.unique())
    item_ids = sorted(ratings.movieId.unique())
    user_idx = {u: i for i, u in enumerate(user_ids)}
    item_idx = {m: i for i, m in enumerate(item_ids)}
    item_idx_inv = {i: m for m, i in item_idx.items()}

    R = build_rating_matrix(train, len(user_ids), len(item_ids), user_idx, item_idx)
    mask = R != 0
    n_ratings = mask.sum(0)  # 영화별 평가 인원 -- 축 요약 프롬프트에 함께 보여줘 인기도 혼동을 막는다

    title_map = dict(zip(movies.movieId, movies.title))
    genre_map = dict(zip(movies.movieId, movies.genres))  # 이력 표시용(장르명) -- genre 방법과 무관, 계속 사용
    # item_genre_matrix, genre_names = build_item_genre_matrix(movies, item_ids)  # genre 방법 전용

    # 축 엔지니어링 결과와 축 페르소나는 결과 CSV 옆에 캐시한다 -- 둘 다 LLM이 정하는 것이라
    # 재실행 때 다시 물으면 축이 달라질 수 있고, 그러면 이미 저장된 유저들과 다른 축으로
    # 프로필을 만들게 된다. 같은 --out으로 이어서 돌리면 같은 축/같은 페르소나를 쓴다.
    out_file = out_path or "results.csv"
    cache_file = os.path.splitext(out_file)[0] + ".axes.json"
    cache_key = {
        "n_factors": n_factors, "methods": list(FACTOR_METHODS), "model": MODEL,
        "persona_n": AXIS_PERSONA_N, "engineered_axes": MAX_ENGINEERED_AXES,
        "min_history_len": min_history_len,
        "max_history_len": max_history_len, "max_pool_users": MAX_POOL_USERS,
    }
    # build_personas.py 가 만들어 커밋해둔 페르소나가 있으면 그걸 쓴다 -- 라벨 파일과 같은 취지로,
    # 매 실행 새로 생성하면 프로필 조건이 흔들려 결과를 비교할 수 없다.
    saved_personas = {}
    if os.path.exists(PERSONAS_PATH):
        with open(PERSONAS_PATH, encoding="utf-8") as f:
            saved = json.load(f)
        saved_personas = {m: saved[m] for m in FACTOR_METHODS if m in saved}
        print(f"고정 페르소나 사용: {PERSONAS_PATH} ({', '.join(saved_personas)})")

    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("key") == cache_key:
            cache = saved.get("methods", {})
            print(f"축 캐시 재사용: {cache_file} ({', '.join(cache)})")
        else:
            print(f"축 캐시 무시: 설정이 바뀌었습니다 ({cache_file})")

    # 방법별로 축/잔차를 미리 계산 (LLM 기반 feature 엔지니어링 포함)
    factorizations = {}
    for method in FACTOR_METHODS:
        # genre_kwargs = (
        #     {"item_genre_matrix": item_genre_matrix, "genre_names": genre_names}
        #     if method == "genre"
        #     else {}
        # )
        cached = cache.get(method)
        U, H, residual, engineered_desc, axis_labels, accepted = engineer_axes_llm(
            R,
            n_factors,
            mask,
            method,
            item_idx_inv,
            title_map,
            max_rounds=MAX_ENGINEERED_AXES,
            replay=cached["engineered"] if cached else None,
            # **genre_kwargs,
        )
        print(f"[{method}] engineered features: {engineered_desc}")
        summaries = top_items_per_factor(H, item_idx_inv, n=8)
        summaries = [[(title_map.get(mid, str(mid)), w) for mid, w in fs] for fs in summaries]
        # 축마다 양 끝 AXIS_PERSONA_N편씩을 보여주고 축 에이전트 페르소나를 받아둔다 (유저 수와 무관, 축당 1회)
        if cached:
            axis_personas = cached["personas"]
        elif method in saved_personas:  # build_personas.py가 만들어 커밋해둔 고정 산출물
            axis_personas = [e["raw"] for e in saved_personas[method]]
        else:
            axis_personas = build_axis_personas(
                H, item_idx_inv, title_map, n_ratings, method, n_side=AXIS_PERSONA_N
            )
        cache[method] = {"engineered": accepted, "personas": axis_personas}
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"key": cache_key, "methods": cache}, f, ensure_ascii=False, indent=1)
        for i, p in enumerate(axis_personas):
            first = (p or "(페르소나 생성 실패)").splitlines()[0]
            print(f"  [{method}] Axis{i + 1} 페르소나: {first}")
        factorizations[method] = {
            "residual": residual,
            "factor_summaries": summaries,
            "axis_labels": axis_labels,
            "axis_personas": axis_personas,
        }

    label_users = np.array([u for u in sorted(answer) if u in user_idx])
    eval_users = rng.choice(
        label_users, size=min(n_eval_users, len(label_users)), replace=False
    )

    # 유저마다 결과를 바로 append 한다 -- 500명이면 3시간 남짓 걸리는데 마지막에 한 번만
    # 저장하면 중간에 죽었을 때 전부 날아간다. 같은 --out으로 다시 실행하면 이미 끝난
    # 유저는 건너뛰고 이어서 진행한다.
    done = set()
    if os.path.exists(out_file):
        try:  # 쓰다가 죽어 마지막 줄이 잘린 CSV여도 이어서 실행이 막히지 않도록
            done = set(pd.read_csv(out_file, usecols=["userId"], on_bad_lines="skip").userId.astype(int))
            print(f"이어서 실행: {out_file}에 이미 {len(done)}명 완료 -- 건너뜁니다")
        except (pd.errors.EmptyDataError, ValueError) as e:
            print(f"기존 결과 파일을 읽지 못해 처음부터 실행합니다 ({e})")
    write_header = not os.path.exists(out_file)

    for uid in tqdm(eval_users):
        if int(uid) in done:
            continue
        u_row = user_idx[uid]
        u_train = train[train.userId == uid]
        history = [
            (title_map.get(mid, str(mid)), genre_map.get(mid, ""), r)
            for mid, r in zip(u_train.movieId, u_train.rating)
        ]
        if not history:
            continue

        # 라벨 파일이 이미 셔플해 둔 순서를 그대로 쓴다(다시 섞으면 다른 실험과 후보 순서가
        # 어긋나고, position bias는 파일 생성 시점에 이미 제거되어 있다).
        target_id = answer[int(uid)]
        candidates = [(m, title_map.get(m, str(m))) for m in candidate_ids[int(uid)]]

        row = {"userId": uid, "history_len": len(history) + 1}  # +1: 라벨로 빼둔 정답 1개

        # 인기도 베이스라인(MostPop): 유저를 전혀 보지 않고 평가 인원순으로만 정렬.
        # LLM 호출이 없어 비용 0이고, "넘어야 할 기준선" 역할을 한다 -- negative가 카탈로그에서
        # 무작위로 뽑히면 정답(중앙값 60명)이 negative(중앙값 8명)보다 훨씬 유명해서, 취향을
        # 전혀 안 봐도 Hit@5가 0.688까지 나온다. 이 값을 못 넘는 방법은 실질적 기여가 없는 셈.
        ranked_pop = sorted(candidates, key=lambda c: -n_ratings[item_idx[c[0]]])
        row["hit_pop"] = hit_at_k(ranked_pop, target_id, TOP_K)
        row["mrr_pop"] = mrr(ranked_pop, target_id)
        row["ndcg_pop"] = ndcg_at_k(ranked_pop, target_id, TOP_K)

        p_raw = profile_raw(history)
        ranked_raw = rank_candidates(p_raw, candidates)
        row["profile_raw"] = p_raw
        row["hit_raw"] = hit_at_k(ranked_raw, target_id, TOP_K)
        row["mrr_raw"] = mrr(ranked_raw, target_id)
        row["ndcg_raw"] = ndcg_at_k(ranked_raw, target_id, TOP_K)

        for method in FACTOR_METHODS:
            residual = factorizations[method]["residual"]
            summaries = factorizations[method]["factor_summaries"]
            axis_labels = factorizations[method]["axis_labels"]
            far_items = user_residual_summary(residual, u_row, item_ids, movies, mask, n=5, mode="far")
            close_items = user_residual_summary(residual, u_row, item_ids, movies, mask, n=5, mode="close")

            p_axis = profile_axis(
                history, summaries, far_items, close_items, method=method, axis_labels=axis_labels,
                axis_personas=factorizations[method]["axis_personas"],
            )
            ranked_axis = rank_candidates(p_axis, candidates)

            row[f"profile_{method}"] = p_axis
            row[f"hit_{method}"] = hit_at_k(ranked_axis, target_id, TOP_K)
            row[f"mrr_{method}"] = mrr(ranked_axis, target_id)
            row[f"ndcg_{method}"] = ndcg_at_k(ranked_axis, target_id, TOP_K)

        pd.DataFrame([row]).to_csv(out_file, mode="a", header=write_header, index=False)
        write_header = False

    df = pd.read_csv(out_file)  # 이어서 실행한 경우까지 포함해 전체를 집계
    methods = ["pop", "raw"] + FACTOR_METHODS
    metric_cols = [f"{metric}_{m}" for m in methods for metric in ("hit", "mrr", "ndcg")]
    print(f"[n_factors={n_factors} n_eval_users={len(df)}] -> {out_file}")
    print(df[metric_cols].mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-factors", type=int, default=N_FACTORS)
    parser.add_argument("--n-eval-users", type=int, default=N_EVAL_USERS)
    parser.add_argument("--out", type=str, default=None, help="output CSV path (default: results.csv)")
    parser.add_argument("--min-history-len", type=int, default=MIN_HISTORY_LEN)
    parser.add_argument("--max-history-len", type=int, default=None, help="상한 없으면 미지정")
    args = parser.parse_args()
    main(args.n_factors, args.n_eval_users, args.out, args.min_history_len, args.max_history_len)
