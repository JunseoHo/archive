from tqdm import tqdm
import pandas as pd

profiles = [{
    'a': 5,
    'b': 10
}, {
    'b': 1,
    'c': 3
}, {
    'a': 9,
    'b': 17
}]

top_words = ['a', 'b', 'c']
word_matrix = pd.DataFrame(index=top_words, columns=top_words, dtype=int)

# 테이블 초기화
word_matrix.fillna(0, inplace=True)

# 각 딕셔너리에서 상위 100개의 단어의 공출현 빈도 계산
for prof in tqdm(profiles, desc="공출현 빈도 계산"):
    for word1 in top_words:
        if word1 in prof:
            for word2 in prof:
                if word2 != word1 and word2 in top_words:
                    word_matrix.loc[word1, word2] += 1

print(word_matrix)