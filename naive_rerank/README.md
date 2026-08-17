# naive_rerank

LLM에게 유저의 상호작용 이력 전체를 주고, 후보 20개(정답 test label 1개 + negative 19개) 중 다음에 볼 영화를 리랭킹하게 시키는 naive 베이스라인.

## 동작 방식
1. `preprocessed/ratings_min300_max500_500users.xlsx`(상호작용 300~500개 유저)에서 유저별 최근 상호작용 1개를 test label로 분리(leave-one-out), 나머지를 이력(제목+장르+평점)으로 사용
   - interaction 상한을 500으로 둔 이유: 이력 전체를 프롬프트에 넣다 보니 상호작용이 아주 많은 유저(min300 원본은 최대 7,359개)는 모델 컨텍스트 한도(128k 토큰)를 초과해서 실패함. 자세한 내용은 `preprocessed/README.md` 참고
2. 후보 20개(정답 1 + negative 19, 셔플 완료)는 `preprocessed/labels_min300_max500_500users.xlsx`(`preprocess_code/build_label_file.py`로 생성된 고정 라벨 파일)에서 그대로 읽음
3. 이력 전체 + 20개 후보를 프롬프트에 넣어 OpenRouter LLM이 선호 순서로 랭킹
4. Hit@K / MRR / NDCG@K로 test label의 순위를 평가

## 준비
```bash
pip install -r requirements.txt
# .env에 OPENROUTER_API_KEY 입력 (.env.example 참고)
```

## 실행
```bash
python main.py                     # 기본 20명 평가
python main.py --n-eval-users 50 --out results_50.csv
```

결과는 `results.csv`에 유저별 hit/mrr/ndcg로 저장, 콘솔에 평균 출력.