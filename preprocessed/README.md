# preprocessed

MovieLens 10M(`raw_data/`)에서 interaction 300 이상 user만 필터링하여 전처리

- `ratings_min300.csv`: 300개 이상 user 전체 (8,269명, 4,858,294행)
- `ratings_min300_500users.xlsx`: 위 유저 중 무작위 500명 user (seed=42, 300,053행)

컬럼: `userId, movieId, rating, timestamp` (원본 그대로 유지)

재현: `python preprocess.py` (프로젝트 루트에서 실행)

- `labels_500users.xlsx`: 500명 유저별 후보 20개(정답 1 + negative 19) 라벨링 파일.

## interaction 300~500 구간 (naive_rerank용)

LLM 프롬프트에 유저 이력 전체를 넣는 `naive_rerank`에서, interaction이 너무 많은 유저(최대 7,359개)는
프롬프트가 모델 컨텍스트 한도(128k 토큰)를 초과해 실패하는 문제가 있어, 상호작용 300~500개로 상한을 둔
버전을 별도로 생성함.

- `ratings_min300_max500.csv`: interaction 300~500개 user 전체 (4,610명, 1,755,584행) — 행 수가 엑셀 한도(1,048,576)를 넘어 csv로만 존재
- `ratings_min300_max500_500users.xlsx`: 위 유저 중 무작위 500명 (seed=42, 189,527행, 유저당 최대 500개)
- `labels_min300_max500_500users.xlsx`: 위 500명 유저별 후보 20개(정답 1 + negative 19) 라벨링 파일
- `labels_min300_max500_4610users.xlsx`: interaction 300~500개 user 전체(4,610명)에 대한 후보 20개 라벨링 파일 (샘플링 없이 전체)

재현:
```bash
python preprocess_code/preprocess.py --min-interactions 300 --max-interactions 500                        # [1] ratings_min300_max500.csv (전체 4,610명) + [2] 500명 xlsx 샘플
python preprocess_code/build_label_file.py --ratings-path preprocessed/ratings_min300_max500_500users.xlsx --out preprocessed/labels_min300_max500_500users.xlsx
python preprocess_code/build_label_file.py --ratings-path preprocessed/ratings_min300_max500.csv --out preprocessed/labels_min300_max500_4610users.xlsx  # 전체 4,610명용
```