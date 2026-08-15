from llm_client import chat


def profile_raw(user_history):
    """Baseline: LLM writes a profile directly from watch history alone (KAR/ONCE style)."""
    lines = "\n".join(f"- {t} ({g}) rated {r}/5" for t, g, r in user_history)
    prompt = (
        "Below is a list of movies a user has watched and rated.\n"
        f"{lines}\n\n"
        "Based on this, describe the user's movie taste in about 200 characters. "
        "Be specific about genres, tone, and preference patterns."
    )
    return chat([{"role": "user", "content": prompt}])


def profile_axis(
    user_history, factor_summaries, far_items, close_items, method: str = "svd", axis_labels=None,
    axis_personas=None,
):
    """Axis-based: profile generated conditioned on matrix-factorization (method) axes, plus
    movies the axes already explain well (close) contrasted with ones they don't (far).
    axis_labels lets interpretable axes (e.g. genre names) be shown by name instead of "Axis N".
    axis_personas(axis_persona.build_axis_personas의 결과)가 주어지면 축마다 미리 만들어둔
    에이전트 페르소나 6줄을 대표 영화 목록과 함께 보여준다 -- 목록만으로는 |loading| 상위만
    뽑히는 탓에 축의 한쪽 끝만 전달되는 문제가 있어서, 양 끝을 다 본 페르소나를 함께 준다."""
    blocks = []
    for i, fs in enumerate(factor_summaries):
        head = axis_labels[i] if axis_labels else f"Axis {i + 1}"
        movies = ", ".join(f"{t} ({'+' if w >= 0 else '-'})" for t, w in fs[:5])
        persona = axis_personas[i] if axis_personas and i < len(axis_personas) else None
        if persona:
            blocks.append(f"{head} agent:\n{persona}\nRepresentative movies: {movies}")
        else:
            blocks.append(f"{head}: {movies}")
    factors_text = "\n\n".join(blocks)
    close_text = "\n".join(
        f"- {t}: matches the axes' prediction closely ({e:+.2f})" for t, e in close_items
    ) or "(none)"
    far_text = "\n".join(
        f"- {t}: rated {'higher' if e > 0 else 'lower'} than the axes predict ({e:+.2f})"
        for t, e in far_items
    ) or "(none)"
    lines = "\n".join(f"- {t} ({g}) rated {r}/5" for t, g, r in user_history)
    prompt = (
        "Below are the user's watch history, latent taste axes extracted by matrix "
        "factorization, and how well those axes explain specific movies the user rated.\n\n"
        f"[Watch history]\n{lines}\n\n"
        "[Latent axes] Each axis is a latent taste dimension, described below and followed by a "
        "few representative movies ('+' and '-' mark which side of the axis each movie sits on; "
        "on some axes every representative movie is on the same side).\n"
        f"{factors_text}\n\n"
        f"[Well explained by existing axes]\n{close_text}\n\n"
        "[Not well explained by existing axes] A positive gap means the user secretly likes "
        f"it more than these axes suggest; a negative gap means they like it less.\n{far_text}\n\n"
        "Based primarily on the watch history, describe the user's movie taste in about 200 "
        "characters. Be specific about genres, tone, and preference patterns, and cover the "
        "full range of genres they've actually watched -- do not narrow down to only one or "
        "two genres. Use the latent axes and the well/not-well-explained contrast only as extra "
        "nuance (e.g. a subtle preference the axes reveal), not to claim the user dislikes a "
        "genre unless the watch history itself clearly shows low ratings for it."
    )
    return chat([{"role": "user", "content": prompt}])
