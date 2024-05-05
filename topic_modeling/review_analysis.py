import re

import pandas as pd
from nltk import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from tqdm import tqdm
import math
from konlpy.tag import Komoran

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   사용자 지정 변수
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# 분석할 리뷰 데이터 이름
IN_FILE_PATH = "google_play_store_reviews.csv"

# 외부에 출력할 파일 이름
TF_FILE_PATH = "data/tf.csv"
TF_TOP100_FILE_PATH = "data/tf_top100.csv"
TF_IDF_FILE_PATH = "data/tf_idf.csv"
CO_OCURR_TOP100_FILE_PATH = "data/co_occur_top100.csv"

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   불용어
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

SW_ABUSE = ['시발', '씨발', '병신', '존나', '좆', '새끼']

SW_UNIQUE = ['니케', '명일', '방주', '명일방주', '붕괴', '스타']

SW_CUSTOM = ['앞으로', '이번', '제가', '이건', '이랑', '건지']

STOPWORDS = SW_ABUSE + SW_UNIQUE + SW_CUSTOM

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   치환 사전
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# 형태소 분석 이전에 치환할 단어 사전
PRE_REPLACE = [('겜', '게임'), ('타임', '시간'), ('이야기', '스토리'), ('지역', '스테이지'), ('야가다', '노가다'), ('패치', '업데이트'), ('꿀잼', '재미'),
               ('노잼', '재미'), ('만렙', '레벨')
               ('컨트롤', '조작'), ('호요버스', ''), ('미호요', ''), ('넷마블', ''), ('코스튬', '스킨'), ('돌파', '강화')]

# 형태소 분석 이후에 치환할 단어 사전
POST_REPLACE = [('캐릭', '캐릭터'), ('능지', '지능'), ('다운', '다운로드'), ('뽑기', '가챠'), ('픽업', '가챠'), ('이성', '자원'), ('재료', '자원'),
                ('스포', '스포일러'), ('개발자', '개발사')
                ('이야기', '스토리'), ('타임', '시간'), ('컨트롤', '조작'), ('구입', '구매'), ('짱깨', '중국인'), ('맵', '스테이지'),
                ('사용자', '플레이어'), ('별점', '평점')]

# 삭제하지 않을 한 단어 사전
NO_DELETE_ONE_TERM = ['돈', '맵', '끝', '방', '길', '화']
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   메인 스크립트
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

print(f"Start with {IN_FILE_PATH}")

# 리뷰 데이터의 'review' 칼럼을 리스트로 저장
csv = pd.read_csv(IN_FILE_PATH)
review_list = csv['review'].tolist()

# 선치환
pre_replaced = []

for review in review_list:
    if type(review) is float:  # 비어있는 리뷰는 제외
        continue
    for prerep in PRE_REPLACE:
        review = re.sub(prerep[0], prerep[1], review)
    pre_replaced.append(review)

review_list = pre_replaced

# 토큰화
tokenizer = TreebankWordTokenizer()
tokens_list = []

for review in tqdm(review_list, desc="토큰화"):
    if type(review) is float:  # 비어있는 리뷰는 제외
        continue
    tokens_list.append(tokenizer.tokenize(review))

# 불용어 삭제
for index, tokens in tqdm(enumerate(tokens_list), desc="불용어 제거"):
    tokens_list[index] = [token for token in tokens if token not in STOPWORDS]

# 토큰의 개수가 4개 미만인 리뷰 제거
# tokens_list = [tokens for tokens in tqdm(tokens_list, desc="토큰 개수가 4개 미만인 리뷰 제거") if len(tokens) > 3]

# 어간 추출
# stemmer = PorterStemmer()
stemmer = Komoran()
# stemmer = konlpy.tag.
stems_list = []

for tokens in tqdm(tokens_list, desc="어간추출"):
    stems = []
    for token in tokens:
        stems += stemmer.pos(token)
    # https://docs.komoran.kr/firststep/postypes.html
    # stems = [stem[0] for stem in stems if stem[1] not in ['JKO', 'JKS', 'JKC', 'JKG', 'JKB', 'JKV', 'JKQ', 'JX', 'JC',
    #                                                       'EP', 'EF', ' EC', 'ETN', 'ETM', 'XPN', 'XSN', 'XSV', 'XSA',
    #                                                       'MAJ',
    #                                                       'VCP', 'VCN', 'IC']]
    # , 'SL', 'VV', 'VA'
    stems = [stem[0] for stem in stems if stem[1] in ['NNG', 'NNP']]
    stems = [stem for stem in stems if len(stem) != 1 or stem in NO_DELETE_ONE_TERM]
    stems = [stem for stem in stems if stem not in STOPWORDS]
    # stems = [stemmer.stem(token) for token in tokens]
    replace_stems = []
    for stem in stems:
        replaced = False
        for replace in POST_REPLACE:
            if stem == replace[0]:
                replaced = True
                replace_stems.append(replace[1])
                break
        if replaced is False:
            replace_stems.append(stem)
    stems_list.append(replace_stems)

# 프로파일 생성
term_set = set()

for stems in tqdm(tokens_list, desc="단어 가방 생성"):
    term_set.update(stems)

term_list = list(term_set)

profiles = []

for stems in tqdm(stems_list, desc="프로파일 생성"):
    counter = Counter(stems)
    profile = {term: counter[term] for term in term_list}
    profiles.append(profile)

# print(pd.DataFrame(profiles)) # 성능 상 문제로 사용하지 않음

# TF 계산 및 출력
tf_table = sum((Counter(profile) for profile in tqdm(profiles, desc="TF 계산")), Counter())

with open(TF_FILE_PATH, mode='w', encoding='utf-8') as outfile:
    # csv 파일 헤더 작성
    outfile.write("term,tf\n")
    # tf가 높은 순으로 작성
    for term, tf in tqdm(tf_table.most_common(), desc="TF 파일 작성"):
        outfile.write(f"{term},{tf}\n")

with open(TF_TOP100_FILE_PATH, mode='w', encoding='utf-8') as outfile:
    # csv 파일 헤더 작성
    outfile.write("term,tf\n")
    # tf가 높은 순으로 작성
    for term, tf in tqdm(tf_table.most_common()[:100], desc="TF-Top100 파일 작성"):
        outfile.write(f"{term},{tf}\n")

# TF-IDF 계산 및 출력
num_documents = len(profiles)
idf = {}
df_table = {term: 0 for term in tf_table}

for term in tqdm(tf_table, desc="DF 계산"):
    df = sum(1 for profile in profiles if profile[term] > 0)
    df_table[term] = df

for term, df in tqdm(df_table.items(), desc="IDF 계산"):
    idf[term] = math.log(num_documents / (1 + df))

tf_idf_table = {term: frequency * idf[term] for term, frequency in tf_table.items()}

tf_idf_table = sorted(tf_idf_table.items(), key=lambda x: x[1], reverse=True)  # 내림차순 정렬

with open(TF_IDF_FILE_PATH, mode='w', encoding='utf-8') as outfile:
    # csv 파일 헤더 작성
    outfile.write("term,tf-idf\n")
    # tf-idf가 높은 순으로 작성
    for term, tf_idf in tqdm(tf_idf_table, desc="TF-IDF 파일 작성"):
        outfile.write(f"{term},{tf_idf}\n")

# TF-IDF가 높은 상위 100개의 단어를 추출

tf_idf_table_preprocess = []
sum = 0
for term, tf in tf_table.items():
    sum += tf

sum = sum * 0.001

for term, tf_idf in tqdm(tf_idf_table, desc="TF가 0.1% 미만인 단어 제거"):
    if tf_table[term] > sum:
        tf_idf_table_preprocess.append((term, tf_idf))

tf_idf_table = tf_idf_table_preprocess

top_words = [word for word, tf_idf in sorted(tf_idf_table, key=lambda x: x[1], reverse=True)[:100]]

# 2차원 테이블 생성
word_matrix = pd.DataFrame(index=top_words, columns=top_words, dtype=int)

# 테이블 초기화
word_matrix.fillna(0, inplace=True)

filtered_profiles = []
for prof in tqdm(profiles, desc="프로파일에서 TF-IDF Top 100을 제외하고 제거"):
    filtered_profile = {top_words.index(key) for key, value in prof.items() if (key in top_words and value > 0)}
    filtered_profiles.append(filtered_profile)

# 각 딕셔너리에서 상위 100개의 단어의 공출현 빈도 계산
for prof in tqdm(filtered_profiles, desc="공출현 빈도 계산"):
    for word1 in prof:
        for word2 in prof:
            if word1 != word2:
                word_matrix.loc[top_words[word1], top_words[word2]] += 1

# CSV 파일로 저장
word_matrix.to_csv(CO_OCURR_TOP100_FILE_PATH)
