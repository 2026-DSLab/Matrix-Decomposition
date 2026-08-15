# axis_rerank

행렬분해로 뽑은 **잠재 축**을 LLM 프로필에 넣으면 추천이 더 정확해지는지를 `naive_rerank`와
같은 조건에서 비교하는 실험.

## 흐름

```
평점 행렬 (500 x 9,178)
  └ svd / nmf / fa 로 각각 5개 축 추출
      └ 축마다 양 끝 25편씩(50편)을 보여주고 LLM이 "축 에이전트 페르소나" 6줄 작성
          └ 유저별 프로필 생성 (이력 + 축 페르소나 + 축이 잘/못 설명하는 영화)
              └ 후보 20편을 프로필 조건으로 LLM이 재정렬
                  └ Hit@5 / MRR / NDCG@5
```

비교 대상은 4가지입니다.

| | 내용 |
|---|---|
| `pop` | 유저를 안 보고 평가 인원순으로만 정렬 (MostPop, LLM 호출 없음) |
| `raw` | 시청 이력만으로 프로필 생성 (baseline) |
| `svd` / `nmf` / `fa` | 이력 + 해당 기법의 축 페르소나로 프로필 생성 |

`pop`이 기준선입니다. 이 점수를 못 넘는 방법은 취향을 분석한 의미가 없습니다.

## 공유 산출물

두 파일은 **한 번 만들어 커밋해두고 그대로 읽습니다.** 실행할 때마다 새로 만들면 조건이
흔들려 다른 실험과 숫자를 나란히 놓을 수 없습니다.

- `../preprocessed/labels_500users.xlsx` — 유저별 후보 20개(정답 1 + negative 19)와 정답.
  `preprocess_code/build_label_file.py`로 생성. `naive_rerank`도 같은 파일을 씁니다.
- `axis_personas.json` — 기법 3개 x 축 5개 = 15개 페르소나. `build_personas.py`로 생성.

## 준비

```bash
pip install -r requirements.txt
# 저장소 루트 .env 에 OPENROUTER_API_KEY 입력 (.env.example 참고)
python prepare_axis_data.py     # 루트에서: 공유 xlsx -> data/ml10m-500users/*.csv 변환
```

## 실행

```bash
cd axis_rerank
python build_personas.py                      # 페르소나 생성 (LLM 15회, 최초 1회만)
python main.py --out results_500.csv          # 500명 평가
python main.py --n-eval-users 20 --out t.csv  # 소규모 확인
python main.py --n-factors 3 --out k3.csv     # 축 개수 스윕
```

중간에 끊겨도 **같은 `--out`으로 다시 실행하면 이어서** 진행합니다(유저별로 CSV에 append,
LLM 호출은 최대 5회까지 지수 백오프 재시도).

## 설정 (`config.py`)

| | 기본값 | 설명 |
|---|---|---|
| `FACTOR_METHODS` | `["svd","nmf","fa"]` | 비교할 축 추출 기법 (`genre`는 비활성) |
| `N_FACTORS` | 5 | 축 개수 |
| `AXIS_PERSONA_N` | 25 | 페르소나 생성 시 양 끝에서 뽑을 영화 수 |
| `MAX_ENGINEERED_AXES` | 0 | LLM 축 엔지니어링 (기본 끔, 아래 참고) |
| `N_EVAL_USERS` | 500 | 평가 유저 수 |
| `TOP_K` | 5 | Hit@K / NDCG@K의 K |

## 설계 메모

측정해서 정한 것들이라 바꾸기 전에 참고하세요.

- **축 페르소나 프롬프트는 세 기법이 같은 템플릿**을 씁니다. 다른 줄은 2줄뿐이고(기법 설명 1줄,
  축에 음수가 있는지로 자동 결정되는 낮은 쪽 설명 1줄) 나머지는 동일합니다. 기법마다 문구를
  손보면 성능 차이가 분해 방식 때문인지 프롬프트 때문인지 구분할 수 없습니다.
- **`fit_nmf`는 미관측을 0이 아니라 유저 평균으로 채웁니다.** 0(=최악의 평점)으로 두면 전체 칸의
  93%가 0점이 되어 예측이 통째로 낮게 깔립니다(rmse 2.36 → 0.90).
- **`svds`에 `random_state`를 지정합니다.** 없으면 같은 입력에도 실행마다 축 부호가 뒤집힙니다.
- **축 엔지니어링은 기본 비활성입니다.** 켜면 기법마다 채택 개수가 달라져(svd 4개 / nmf 1개 /
  fa 1개) 축이 비대칭으로 바뀌고, svd에서는 축 2~5가 사실상 다른 축이 되면서 잔차도 늘었습니다.
- **정답은 라벨 파일 기준입니다.** `leave_one_out_split`은 timestamp 동점 유저(500명 중 32명)에서
  "가장 최근 1개"가 정렬 방식에 따라 달라집니다.
- **`fa`는 이 데이터에서 축이 붕괴합니다.** 유저 500명 < 영화 9,178편이라 유저별 축 점수가
  500명 중 6종밖에 안 나옵니다. 대조군으로 보는 게 맞습니다.

## 파일

| | |
|---|---|
| `config.py` | 경로, 모델, 실험 하이퍼파라미터 |
| `data.py` | 데이터 로딩, 라벨 기반 split, 평점 행렬 생성 |
| `mf.py` | svd / nmf / fa 축 추출, 축별 대표 영화, 유저별 잔차 요약 |
| `axis_persona.py` | 축 페르소나 프롬프트 |
| `build_personas.py` | 페르소나 생성 → `axis_personas.json` |
| `profiling.py` | 프로필 생성 (raw / axis) |
| `recommend.py` | 프로필 조건으로 후보 랭킹 |
| `evaluate.py` | Hit@K, MRR, NDCG@K |
| `axis_engineering.py` | LLM 축 엔지니어링 (기본 비활성) |
| `main.py` | 전체 파이프라인 |
