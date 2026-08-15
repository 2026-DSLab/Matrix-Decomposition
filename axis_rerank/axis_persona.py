"""각 축을 담당할 "축 에이전트"의 페르소나를 LLM에게 생성시킨다.

설계 원칙: **세 기법(svd/nmf/fa)이 완전히 동일한 프롬프트를 받는다.** 기법마다 문구를 손보면
성능 차이가 행렬분해 차이 때문인지 프롬프트 차이 때문인지 구분할 수 없게 된다. 그래서
기법 이름도, 기법별 규칙도, 기법별 항목명도 넣지 않는다. 프롬프트에서 기법에 따라 달라지는
것은 오직 "영화 50편과 그 점수"뿐이다.

이를 가능하게 하는 두 가지 장치:
1. 점수를 축 내부 z-score로 표준화한다. 원시 loading은 svd ±0.12 / nmf 0~6.8 / fa ±0.15로
   스케일이 제각각이라 같은 문장으로 설명할 수 없었다. z-score는 단조변환이라 어떤 영화가
   뽑히는지는 전혀 바뀌지 않고, 표시되는 숫자만 통일된다.
2. 낮은 쪽의 의미(반대 취향이냐 무관이냐)를 기법별로 알려주는 대신, 두 경우를 모두 적은
   동일한 문장을 주고 영화들을 보고 판단하게 맡긴다. nmf만 "반대가 아니다"라고 알려주는 것은
   nmf에게만 힌트를 주는 셈이라 통제를 깬다.

축 하나에는 영화 9,178편 전부가 점수를 갖지만 전부를 넣을 수는 없으므로 양 끝에서 N편씩만
뽑아 대비시킨다. |점수| 상위만 뽑던 기존 방식은 축의 한쪽 끝만 보여주는 문제가 있었다
(15개 축 중 13개가 한쪽 부호로만 채워졌음).
"""

import numpy as np

from llm_client import chat

# 기법 설명은 세 기법 모두 같은 위치, 같은 형식의 한 줄로만 들어간다.
_METHOD_DESC = {
    "svd": "SVD (the rating matrix is centered by each user's mean rating, then compressed by truncated singular value decomposition)",
    "nmf": "NMF (non-negative matrix factorization -- every score is constrained to be zero or positive)",
    "fa": "FA (factor analysis -- estimates a separate noise variance for every movie and keeps only the shared latent factors)",
}

_NEUTRAL = (
    "The axis gives a score to every one of the {n} films; the scores below are scaled so that "
    "the strongest film on this axis is 1.00.\n"
)

# 낮은 쪽이 "반대편 끝"인지 "이 축이 다루지 않는 영화"인지는 기법 이름이 아니라 **그 축의
# 점수 분포**로 판정한다 -- 음수가 존재하면 반대 방향으로 끌어당기는 성분이 실제로 있는
# 것이고, 전부 0 이상이면 낮은 쪽은 "약하게만 관련됨"이다. 규칙은 15개 축 모두에 똑같이
# 적용되므로 nmf에만 힌트를 주는 것이 아니다. (실측: svd 5/5, fa 5/5는 음수 있음, nmf 0/5)
_BOTTOM_TWO_SIDED = (
    "Scores run both positive and negative here: the bottom films carry the opposite sign, so "
    "they represent the opposing end of this same axis.\n\n"
)
_BOTTOM_ONE_SIDED = (
    "Every score on this axis is zero or positive, so the bottom films are NOT an opposing "
    "taste -- they are simply films this axis barely describes. Do not phrase them as a "
    "dislike.\n\n"
)

_FORMAT = (
    "Write the agent's persona as EXACTLY 6 lines in this format, nothing else:\n"
    "Name: (at most 6 words)\n"
    "Identity: (one sentence, first person: 'I am a viewer who judges films by ...')\n"
    "High end: (what the top-scoring films share, with 3 example titles in parentheses)\n"
    "Low end: (what the bottom-scoring films share, with 3 example titles in parentheses)\n"
    "Judgment rule: (one sentence: seeing an unfamiliar film, what do I look at to place it "
    "on this axis?)\n"
    "Not my call: (what this axis cannot judge and should be left to the other agents)\n\n"
)

# 전부 "무엇을 하라"는 지시문이다. 금지 목록("'serious vs light-hearted' 같은 표현을 쓰지
# 마라")을 넣었던 버전과 비교 실험한 결과, 금지문을 빼자 오히려 그 어휘 사용이 줄었고
# (svd 13->11, nmf 10->8, fa 16->12회) 축 간 중복도도 세 기법 모두 낮아졌다
# (0.154->0.125, 0.191->0.169, 0.163->0.148). 금지 표현을 명시하는 것 자체가 그 표현을
# 유도한 것으로 보인다. 제목 환각은 두 버전 모두 0건.
_RULES = (
    "Rules:\n"
    "- Base your answer on all 50 films.\n"
    "- Name at least TWO concrete attributes that pin down this axis in particular: era "
    "(e.g. late 1990s), sub-genre (e.g. slasher, space opera, screwball comedy), production "
    "scale (e.g. studio tentpole, low-budget cult), or reception (e.g. Razzie-level flops).\n"
    "- Copy every example title verbatim from the lists above.\n"
    "- Read the rater counts alongside the scores, and name the difference that remains once "
    "you account for how widely each film was seen.\n"
    "- If the two lists show no consistent tendency, write exactly 'cannot judge' on all six "
    "lines."
)


def _pick_ends(h, item_idx_inv, title_map, n_ratings, n_side):
    """한 축에서 점수 상위 n_side편과 하위 n_side편을 (제목, 점수, 평가인원)으로 반환.

    동점일 때는 평가 인원이 많은 쪽을 먼저 뽑는다 -- nmf는 점수가 정확히 같은 영화가 축에 따라
    수십~수백 편이라(Axis1 76편, Axis2 175편) 그냥 정렬하면 movieId 순서대로 임의의 25편이
    뽑힌다. 인원순으로 끊으면 "많이 봤는데도 이 축과 무관한 영화"가 되어 해석이 가능해진다."""
    hi = np.lexsort((-n_ratings, -h))[:n_side]
    lo = np.lexsort((-n_ratings, h))[:n_side]

    def rows(idx):
        return [
            (title_map.get(item_idx_inv[i], str(item_idx_inv[i])), float(h[i]), int(n_ratings[i]))
            for i in idx
        ]

    return rows(hi), rows(lo)


def _build_prompt(method, hi, lo, n_items, two_sided):
    """세 기법이 같은 템플릿을 쓴다. 달라지는 것은 영화 목록과 두 줄뿐이다 -- 기법 설명 1줄,
    그리고 그 축의 점수 분포에서 자동으로 결정되는 낮은 쪽 설명 1줄(svd/fa는 같은 문장이
    걸리므로 svd vs fa는 1줄, svd vs nmf는 2줄 차이)."""
    listing = lambda rows: "\n".join("- %s  %+.2f  (%d raters)" % r for r in rows)
    return (
        "You design the persona of an 'axis agent': an agent that will represent ONE latent "
        "taste dimension of a movie recommender, alongside other agents representing the other "
        "dimensions.\n\n"
        + f"This axis was extracted with {_METHOD_DESC[method]}.\n"
        + _NEUTRAL.format(n=n_items)
        + (_BOTTOM_TWO_SIDED if two_sided else _BOTTOM_ONE_SIDED)
        + f"[Top {len(hi)} films on this axis]\n{listing(hi)}\n\n"
        + f"[Bottom {len(lo)} films on this axis]\n{listing(lo)}\n\n"
        + _FORMAT
        + _RULES
    )


def build_axis_personas(H, item_idx_inv, title_map, n_ratings, method, n_side=25):
    """축마다 페르소나 6줄을 받아 리스트로 반환. 실패한 축은 None으로 남긴다(fail-safe).

    method는 로그용으로만 받고 프롬프트에는 넣지 않는다 -- 기법 이름을 넣으면 LLM이 사전학습
    지식으로 기법마다 다르게 해석할 여지가 생겨 통제가 깨진다. 실제로 이름을 넣은 변형과 뺀
    변형을 각각 15축씩 돌려본 결과, 축 간 중복도 차이는 재실행 간 변동 폭 안에 들어 이름을
    넣어서 얻는 이득은 확인되지 않았다.

    유저 수와 무관하게 기법당 축 개수만큼만 호출된다(기본 3기법 x 5축 = 15회)."""
    personas = []
    for axis in range(H.shape[0]):
        h = H[axis]
        # 최대 절댓값으로 나눠 스케일만 통일한다(단조변환이라 뽑히는 영화는 그대로).
        # z-score와 달리 부호 구조가 보존되어, 비음수 축은 표시값도 0 이상으로 남는다.
        scaled = h / (np.abs(h).max() or 1.0)
        hi, lo = _pick_ends(scaled, item_idx_inv, title_map, n_ratings, n_side)
        try:
            resp = chat(
                [{"role": "user", "content": _build_prompt(method, hi, lo, H.shape[1], bool(h.min() < 0))}],
                temperature=0.0,
                max_tokens=400,
            )
        except Exception:  # 페르소나는 부가 정보이므로 실패해도 실험 전체를 멈추지 않는다
            personas.append(None)
            continue
        lines = [ln.strip() for ln in resp.strip().splitlines() if ln.strip()][:6]
        personas.append("\n".join(lines) if lines else None)
    return personas
