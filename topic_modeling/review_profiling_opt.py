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

# 리뷰 데이터가 저장되어 있는 csv 파일의 위치
PREFIX = "negative_"

IN_FILE_PATH = "data/" + PREFIX + "reviews.csv"

# 외부에 출력할 파일 이름
TF_FILE_PATH = "data/" + PREFIX + "tf.csv"
TF_TOP100_FILE_PATH = "data/" + PREFIX + "tf_top100.csv"
TF_IDF_FILE_PATH = "data/" + PREFIX + "tf_idf.csv"
CO_OCURR_TOP100_FILE_PATH = "data/" + PREFIX + "co_ocurr.csv"

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   불용어
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# NLTK 라이브러리의 불용어 사전
SW_NLTK = ['ours', 'be', "you'll", 've', 'is', "isn't", "wouldn't", 'about', 'had', "that'll", 'my', 'so', "aren't",
           'weren', 'out', 'because', 'below', 'where', 'why', 'shouldn', 'her', "won't", 'over', 'no', 'couldn',
           'did', "doesn't", 'down', 'shan', 'until', "should've", 'some', 'ma', 'both', 'theirs', 'yourselves',
           'itself', 'ain', 's', 'will', 'nor', 'o', 'once', 'she', "it's", 'them', 'him', 'doing', 'into', 'who',
           'by', "couldn't", 'of', 'for', 'on', 'under', 'then', 'just', 'wasn', 'at', 'those', 'yourself', 'myself',
           'their', 'mustn', 'too', "you're", 'does', 'whom', "needn't", 'between', 'an', 'have', 'only', 'm',
           'other', 'after', 'again', 'which', 'from', 'should', 'it', "hasn't", 'own', 'hers', 'his', 'himself',
           "she's", 'each', 'its', 'themselves', 'haven', 'or', 'didn', 'they', 'as', 'd', 'hadn', 'been', 'mightn',
           'very', 'up', 'further', 'doesn', 'but', 'any', 'i', 'with', 'when', 'he', 'more', 'how', 'll', 'hasn',
           'that', 'having', 'the', 'do', 'same', 'this', 'in', 'herself', 'to', 'against', 'not', 'are', 'such',
           'all', 'off', "wasn't", 'was', "shan't", 'we', "shouldn't", 'wouldn', 'ourselves', 'here', 'won', 'can',
           "didn't", 'yours', 'aren', "you'd", "hadn't", 'before', 'y', 'few', 'now', 'most', 'am', 'than', 'during',
           'these', 're', 'don', "you've", 'through', 'has', "weren't", 'there', 'and', "don't", 'were', 'your',
           'what', 'needn', 'our', 'me', 'above', "mustn't", 'if', 't', 'you', "mightn't", 'a', 'while', "haven't",
           'being', 'isn']

# 비속어 불용어 사전
SW_ABUSE = ['nigga', 'fuck']

# 사용자 불용어 사전
SW_CUSTOM = ['the', 'and', 'i', 'to', 'a', 'you', 'of', 'it', 'is', 'in', 'that', 'for', 's', 'with', 'but', 'be',
             'are', 'can', 'on', 'have', 't', 'as', 'your', 'if', 'my', 'get', 'there', 'more', 'so', 'or', 'just',
             'at', 'some', 'one', 'they', 'me', 'will', 'an', 'from', 'up', 'do', 'b', 'what', 'by', 'them' 'also',
             '10', 'when', 'which', 'even', 'would', 'about', 'other', 'much', 'thing', 'too', 'into', 'way', 'want',
             'lot', 'love', 're', 'm', 'than', 'how', 'while', 'most', 'their', 'after', 'could', 'still', 'where',
             'then', '2', 'h1', 'who', 'through', 'each', '1', 'over', '3', 'we', 'am'
             ]

SW_CUSTOM = ['하', '아서', '지만', '어서', '네요', '라고', '거나', '이런', '그런', '저런', '이것', '저것', '그것' '이렇', '어디',
             '특히', '어느', '때문', '스팀', '건지', '누드', '새끼', '여기', '저기', '거기', '이건', '저건', '그건', '대부분', '나름', '무엇', '이다',
             '시발', '제가', '조금', '누구']

# STOPWORDS = SW_NLTK + SW_ABUSE + SW_CUSTOM
STOPWORDS = SW_CUSTOM

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   치환 사전
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

REPLACE = [('재밌', '재미'), ('재미없', '재미')]

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   메인 스크립트
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# 리뷰 데이터의 'review' 칼럼을 리스트로 저장
csv = pd.read_csv(IN_FILE_PATH)
review_list = csv['review'].tolist()

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
stems_list = []

for tokens in tqdm(tokens_list, desc="어간추출"):
    stems = []
    for token in tokens:
        stems += stemmer.pos(token)
    # stems = [stem[0] for stem in stems if stem[1] not in ['JKO', 'JKS', 'JKC', 'JKG', 'JKB', 'JKV', 'JKQ', 'JX', 'JC',
    #                                                       'EP', 'EF', ' EC', 'ETN', 'ETM', 'XPN', 'XSN', 'XSV', 'XSA',
    #                                                       'MAJ',
    #                                                       'VCP', 'VCN', 'IC']]
    stems = [stem[0] for stem in stems if stem[1] in ['NNG', 'NNP', 'NNB', 'NP', 'NR', 'VV', 'VA']]
    stems = [stem for stem in stems if len(stem) != 1]
    stems = [stem for stem in stems if stem not in STOPWORDS]
    # stems = [stemmer.stem(token) for token in tokens]
    replace_stems = []
    for stem in stems:
        replaced = False
        for replace in REPLACE:
            if stem == replace[0]:
                replaced = True
                replace_stems.append(replace[1])
                break
        if replaced is False:
            replace_stems.append(stem)
    stems_list.append(replace_stems)
    stems_list.append(stems)

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
