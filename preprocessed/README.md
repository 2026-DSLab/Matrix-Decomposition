# preprocessed

MovieLens 10M(`raw_data/`)에서 interaction 300 이상 user만 필터링하여 전처리

- `ratings_min300.csv`: 300개 이상 user 전체 (8,269명, 4,858,294행)
- `ratings_min300_500users.xlsx`: 위 유저 중 무작위 500명 user (seed=42, 300,053행)

컬럼: `userId, movieId, rating, timestamp` (원본 그대로 유지)
