import re

from llm_client import chat


def rank_candidates(history, candidates):
    """history: 유저의 전체 상호작용 이력, {"title", "genres", "rating"} dict 리스트,
    시간순(오래된 것부터) 정렬됨.
    candidates: (movieId, title_with_genre) 튜플 리스트 (1개 test label + N개 negative,
    labels_500users.xlsx에 저장된 순서 그대로, 이미 셔플됨).
    LLM이 이력 전체(장르/평점 포함)를 보고 후보를 선호(=다음에 봤을 법한) 순서로 재정렬."""
    n = len(candidates)
    history_listing = "\n".join(
        f"{i + 1}. {h['title']} ({h['genres']}) - rated {h['rating']}/5"
        for i, h in enumerate(history)
    )
    candidate_listing = "\n".join(f"{i + 1}. {title}" for i, (_, title) in enumerate(candidates))

    prompt = (
        "You are a movie recommendation system. Below is one user's complete watch "
        "history, in chronological order (oldest first), with the rating they gave "
        "each film.\n\n"
        f"{history_listing}\n\n"
        f"Here are {n} candidate movies. Exactly one of them is the movie this user "
        "watched next, right after the history above.\n"
        f"{candidate_listing}\n\n"
        "Based on patterns in the history above -- genres and combinations the user "
        "rates highly vs. rarely watches, eras/decades they favor, and any drift in "
        "taste over time (recent titles vs. earlier ones) -- rank all "
        f"{n} candidates from most to least likely to be the one the user watched "
        "next. Output the numbers only, separated by commas, no explanation. "
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