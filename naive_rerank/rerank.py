import re

from llm_client import chat


def rank_candidates(history_titles, candidates):
    """history_titles: 유저가 본 영화 제목 전체 리스트.
    candidates: (movieId, title) 튜플 리스트 (1개 test label + N개 negative, 셔플됨).
    LLM이 이력 전체를 보고 후보를 선호(=다음에 봤을 법한) 순서로 재정렬."""
    n = len(candidates)
    history_listing = "\n".join(f"- {t}" for t in history_titles)
    candidate_listing = "\n".join(f"{i + 1}. {title}" for i, (_, title) in enumerate(candidates))

    prompt = (
        "A user has watched and rated the following movies:\n"
        f"{history_listing}\n\n"
        f"Here is a list of {n} candidate movies. Exactly one of them is the movie "
        "this user watched next.\n"
        f"{candidate_listing}\n\n"
        f"Rank all {n} candidates from most to least likely to be the one the user "
        "watched next. Output the numbers only, separated by commas, no explanation. "
        "Example: 3,1,7,2,5,..."
    )

    resp = chat([{"role": "user", "content": prompt}])
    nums = [int(x) for x in re.findall(r"\d+", resp)]

    seen = set()
    ranked = []
    for num in nums:
        if 1 <= num <= n and num not in seen:
            seen.add(num)
            ranked.append(candidates[num - 1])
    return ranked