import math
import re
from collections import Counter

import pandas as pd
from konlpy.tag import Komoran
from nltk import TreebankWordTokenizer
from tqdm import tqdm


def get_profiles(infile, stopwords, prerepls, postrepls, monodict, poslist):
    csv = pd.read_csv(infile)
    reviews = csv['review'].tolist()

    prereplaced = []
    for review in tqdm(reviews, '전치환'):
        if type(review) is float:  # 비어있는 리뷰는 제외
            continue
        for prerepl in prerepls:
            review = re.sub(prerepl[0], prerepl[1], review)
        prereplaced.append(review)
    reviews = prereplaced

    tokenizer = TreebankWordTokenizer()
    tokens_list = []
    for review in tqdm(reviews, desc="토큰화"):
        tokens_list.append(tokenizer.tokenize(review))

    for index, tokens in tqdm(enumerate(tokens_list), desc="불용어 제거"):
        tokens_list[index] = [token for token in tokens if token not in stopwords]

    stemmer = Komoran()
    stems_list = []
    for tokens in tqdm(tokens_list, desc="형태소 분석"):
        stems = []
        for token in tokens:
            stems += stemmer.pos(token)
        stems = [stem[0] for stem in stems if stem[1] in poslist]
        stems = [stem for stem in stems if len(stem) != 1 or stem in monodict]
        stems = [stem for stem in stems if stem not in stopwords]  # 형태소 분석 이후에도 불용어 제거 수행

        postreplaced = []  # 후치환
        for stem in stems:
            replaced = False
            for postrepl in postrepls:
                if stem == postrepl[0]:
                    replaced = True
                    postreplaced.append(postrepl[1])
                    break
            if replaced is False:
                postreplaced.append(stem)
        stems_list.append(postreplaced)

    term_set = set()
    for stems in tqdm(stems_list, desc="단어 가방 생성"):
        term_set.update(stems)
    term_set = list(term_set)

    profiles = []
    for stems in tqdm(stems_list, desc="프로파일 생성"):
        counter = Counter(stems)
        profile = {term: counter[term] for term in term_set}
        profiles.append(profile)

    return profiles


ARKNIGHTS_INFILE = "review/arknights.csv"
ARKNIGHTS_STOPWORDS = ['명일방주', '명일', '방주', '제가', '구글', '스토어', '이번', '이랑', '앞으로', '지금', '스카디', '수르트']
ARKNIGHTS_PREREPL = [('클리어', 'Clear'), ('콜라보레이션', 'Collaboration'), ('콜라보', 'Collaboration'), ('합성옥', '재화'),
                     ('오리지늄', '재화'), ('픽업', 'Gacha'), ('뽑기', 'Gacha'), ('가챠', 'Gacha'), ('가차', 'Gacha'),
                     ('헤드헌팅', 'Gacha'), ('꿀잼', '재미'),
                     ('노잼', '재미'), ('월정액', 'MonthlyPlan'), ('애니', "Animation"), ('애니메이션', 'Animation'),
                     ('리세', 'Risemara'), ('리세마라', 'Risemara'), ('퀘스트', 'Quest'), ('일퀘', 'Quest')]
ARKNIGHTS_POSTREPL = [('이성', '피로도'), ('행동력', '피로도'), ('Clear', '클리어'), ('능지', '지능'), ('Collaboration', '콜라보레이션'),
                      ('별점', '평점'),
                      ('다운', '다운로드'), ('용문폐', '자원'), ('재료', '자원'), ('수준', '정도'), ('Gacha', '가챠'), ('갤럭시', '기종'),
                      ('아이폰', '기종'), ('스킨', '코스튬'), ('오픈', '출시'), ('패키지', '상품'), ('MonthlyPlan', '월정액'),
                      ('Animation', '애니메이션'), ('Risemara', '리세마라'), ('컨트롤', '조작'), ('개발자', '게임사'), ('스코어', '점수'),
                      ('훈장', '업적'), ('Quest', '퀘스트')]
ARKNIGHTS_MONODICT = []
ARKNIGHTS_POSLIST = ['NNG', 'NNP', 'SL']

arknights_profiles = get_profiles(ARKNIGHTS_INFILE, ARKNIGHTS_STOPWORDS, ARKNIGHTS_PREREPL, ARKNIGHTS_POSTREPL,
                                  ARKNIGHTS_MONODICT, ARKNIGHTS_POSLIST)

profiles = arknights_profiles

# TF 계산 및 출력
tf_table = sum((Counter(profile) for profile in tqdm(profiles, desc="TF 계산")), Counter())

with open("data/tf.csv", mode='w', encoding='utf-8') as outfile:
    # csv 파일 헤더 작성
    outfile.write("term,tf\n")
    # tf가 높은 순으로 작성
    for term, tf in tqdm(tf_table.most_common(), desc="TF 파일 작성"):
        outfile.write(f"{term},{tf}\n")

with open("data/tf_top100.csv", mode='w', encoding='utf-8') as outfile:
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

with open("data/tf_idf.csv", mode='w', encoding='utf-8') as outfile:
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
word_matrix.to_csv("data/co_occur.csv")
