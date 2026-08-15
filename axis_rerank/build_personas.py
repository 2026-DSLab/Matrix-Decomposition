"""축 에이전트 페르소나를 만들어 axis_personas.json 으로 저장한다.

labels_500users.xlsx 와 같은 취지의 고정 산출물이다 -- 페르소나는 LLM이 쓰는 것이라 매번
생성하면 실행마다 조금씩 달라지고, 그러면 프로필 조건이 흔들려 결과를 비교할 수 없다.
한 번 만들어 커밋해두면 main.py 가 그대로 읽어 쓴다.

사용법 (axis_rerank/ 에서):
    python build_personas.py                 # axis_personas.json 생성
    python build_personas.py --out other.json
"""

import argparse
import json

import numpy as np

from axis_persona import build_axis_personas
from config import (AXIS_PERSONA_N, FACTOR_METHODS, LABELS_PATH, MAX_POOL_USERS,
                    MIN_HISTORY_LEN, MODEL, N_FACTORS, RANDOM_SEED)
from data import (build_rating_matrix, filter_users_by_history, load_labels,
                  load_movielens, split_by_labels)
from mf import fit_factorization

_FIELDS = [
    ("name", "Name:"),
    ("identity", "Identity:"),
    ("high_end", "High end:"),
    ("low_end", "Low end:"),
    ("judgment_rule", "Judgment rule:"),
    ("not_my_call", "Not my call:"),
]


def parse_persona(text: str) -> dict:
    """6줄 페르소나를 필드별로 쪼갠다. 원문도 raw로 함께 남긴다."""
    out = {"raw": text}
    for line in (text or "").splitlines():
        for key, prefix in _FIELDS:
            if line.startswith(prefix):
                out[key] = line[len(prefix):].strip()
    return out


def main(out_path: str):
    ratings, movies = load_movielens()
    ratings = filter_users_by_history(
        ratings, MIN_HISTORY_LEN, None, max_users=MAX_POOL_USERS, seed=RANDOM_SEED
    )
    answer, _ = load_labels()
    train = split_by_labels(ratings, answer)

    user_ids = sorted(ratings.userId.unique())
    item_ids = sorted(ratings.movieId.unique())
    user_idx = {u: i for i, u in enumerate(user_ids)}
    item_idx = {m: i for i, m in enumerate(item_ids)}
    item_idx_inv = {i: m for m, i in item_idx.items()}
    R = build_rating_matrix(train, len(user_ids), len(item_ids), user_idx, item_idx)
    n_ratings = (R != 0).sum(0)
    title_map = dict(zip(movies.movieId, movies.title))

    result = {
        "config": {
            "n_factors": N_FACTORS,
            "methods": list(FACTOR_METHODS),
            "model": MODEL,
            "persona_n": AXIS_PERSONA_N,
            "labels": LABELS_PATH.name,
            "min_history_len": MIN_HISTORY_LEN,
            "max_pool_users": MAX_POOL_USERS,
            "random_seed": RANDOM_SEED,
        }
    }
    for method in FACTOR_METHODS:
        H = fit_factorization(R, N_FACTORS, method=method)[1]
        personas = build_axis_personas(
            H, item_idx_inv, title_map, n_ratings, method, n_side=AXIS_PERSONA_N
        )
        result[method] = []
        for i, p in enumerate(personas):
            entry = {"axis": i + 1, "two_sided": bool(H[i].min() < 0)}
            entry.update(parse_persona(p) if p else {"raw": None})
            result[method].append(entry)
            print(f"[{method}] Axis{i + 1}: {entry.get('name', '(생성 실패)')}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"\n{out_path}: {len(FACTOR_METHODS)}개 기법 x {N_FACTORS}개 축 저장")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="축 에이전트 페르소나 생성 및 저장")
    parser.add_argument("--out", type=str, default="axis_personas.json")
    args = parser.parse_args()
    main(args.out)
