import nltk
from collections import Counter
import pandas as pd
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords

df = pd.read_csv('steam_reviews.csv')
nltk.download('wordnet')
nltk.download('stopwords')


# 텍스트 전처리 함수 정의
def preprocess_text(text):
    if type(text) != str:
        text = str(text)
    # 소문자 변환
    # 토큰화
    tokens = word_tokenize(text)
    # 불용어 제거
    stop_words = set(stopwords.words('english'))
    stop_words.add(".")
    stop_words.add("n't")
    stop_words.add("'s")
    tokens = [token for token in tokens if token not in stop_words]
    # 표제어 추출
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # 공백으로 구분된 문자열로 재결합
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


# 텍스트 전처리 적용
df['preprocessed_text'] = df['review'].apply(preprocess_text)

# 전처리된 텍스트 확인

# NLTK의 데이터 다운로드 (처음 한 번만 실행)

# 여러 개의 텍스트가 담긴 배열
texts = df['preprocessed_text']

# 전체 텍스트를 담을 리스트 초기화
all_tokens = []

# 각 텍스트에 대해 토큰화 및 단어 빈도 계산
for text in texts:
    # 텍스트 토큰화
    print(text)
    tokens = nltk.word_tokenize(text)
    all_tokens.extend(tokens)

# 전체 텍스트에 대한 단어 빈도 계산
word_freq = Counter(all_tokens)

# 가장 많이 나타나는 단어 상위 n개 추출
top_keywords = word_freq.most_common(50)

# 결과 출력
for keyword, freq in top_keywords:
    print(f"{keyword}: {freq} times")
