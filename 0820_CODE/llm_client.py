import time

import requests
import urllib3

from config import (MAX_RETRIES, MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
                    RETRY_BACKOFF_SEC, VERIFY_SSL)

if not VERIFY_SSL:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 재시도할 상태코드: 요청 타임아웃, rate limit, 서버 측 오류. 401/400 같은 건 재시도해도
# 똑같이 실패하므로 즉시 중단한다.
_RETRYABLE = {408, 429}


def chat(messages, temperature: float = 0.3, max_tokens: int = 400) -> str:
    """OpenRouter 호출. 일시적 실패는 지수 백오프로 재시도한다.

    500명 x 8회 + 축 페르소나 = 약 4,000회를 순차 호출하는데, 그중 한 번의 rate limit이나
    네트워크 끊김으로 3시간짜리 실행이 통째로 죽는 것을 막기 위함."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("환경변수 OPENROUTER_API_KEY가 설정되지 않았습니다.")

    last_err = None
    for attempt in range(MAX_RETRIES):
        wait = RETRY_BACKOFF_SEC * (2 ** attempt)
        try:
            resp = requests.post(
                OPENROUTER_BASE_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=60,
                verify=VERIFY_SSL,
            )
        except requests.RequestException as e:  # 연결 실패/타임아웃 -- 재시도 대상
            last_err = e
        else:
            if resp.status_code < 400:
                try:
                    return resp.json()["choices"][0]["message"]["content"]
                except (ValueError, KeyError, IndexError) as e:  # 응답 형식 이상 -- 재시도
                    last_err = e
            elif resp.status_code in _RETRYABLE or resp.status_code >= 500:
                last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = max(wait, int(retry_after))
            else:  # 인증 오류 등 -- 재시도해도 소용없으므로 즉시 중단
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

        if attempt < MAX_RETRIES - 1:
            print(f"  [llm_client] 호출 실패({last_err}) -- {wait:.0f}초 후 재시도 "
                  f"{attempt + 2}/{MAX_RETRIES}", flush=True)
            time.sleep(wait)

    raise RuntimeError(f"LLM 호출이 {MAX_RETRIES}회 모두 실패했습니다: {last_err}")
