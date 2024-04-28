import pandas as pd
from nltk import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from tqdm import tqdm
from collections import OrderedDict
import math

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   사용자 지정 변수
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# 리뷰 데이터가 저장되어 있는 csv 파일의 위치
IN_FILE_PATH = "data/reviews.csv"

# 외부에 출력할 파일 이름
TF_FILE_PATH = "data/tf.csv"
TF_IDF_FILE_PATH = "data/tf_idf.csv"

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   불용어
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

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

SW_ABUSE = ['nigga', 'fuck']

SW_CUSTOM = ['the', 'and', 'i', 'to', 'a', 'you', 'of', 'it', 'is', 'in', 'that', 'for', 's', 'with', 'but', 'be',
             'are', 'can', 'on', 'have', 't', 'as', 'your', 'if', 'my', 'get', 'there', 'more', 'so', 'or', 'just',
             'at', 'some', 'one', 'they', 'me', 'will', 'an', 'from', 'up', 'do', 'b', 'what', 'by', 'them' 'also',
             '10', 'when', 'which', 'even', 'would', 'about', 'other', 'much', 'thing', 'too', 'into', 'way', 'want',
             'lot', 'love', 're', 'm', 'than', 'how', 'while', 'most', 'their', 'after', 'could', 'still', 'where',
             'then', '2', 'h1', 'who', 'through', 'each', '1', 'over', '3', 'we', 'am'
             ]

STOPWORDS = SW_NLTK + SW_ABUSE + SW_CUSTOM

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   메인 스크립트
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# 리뷰 데이터의 'review' 칼럼을 리스트로 저장
csv = pd.read_csv(IN_FILE_PATH)
review_list = csv['review'].tolist()

# 토큰화
tokenizer = TreebankWordTokenizer()
tokens_list = []

for review in tqdm(review_list, desc=f"{"토큰화":15}"):
    if (type(review) == float):
        continue
    tokens_list.append(tokenizer.tokenize(review))

# 불용어 삭제
for i, tokens in tqdm(enumerate(tokens_list), desc="불용어 제거"):
    tokens_list[i] = [token for token in tokens if token not in STOPWORDS]
# for tokens in tqdm(tokens_list, desc=f"{"불용어 제거":15}"):
#     tokens = [token for token in tokens if token not in STOPWORDS]

# 전처리 (토큰의 개수가 4개 미만인 리뷰는 제외)
tokens_list = [tokens for tokens in tqdm(tokens_list, desc=f"{"토큰 개수가 4개 미만인 리뷰 제거":15}") if len(tokens) > 3]

# 어간 추출
stemmer = PorterStemmer()
stems_list = []

for tokens in tqdm(tokens_list, desc=f"{"어간추출":15}"):
    stems = [stemmer.stem(token) for token in tokens]
    stems_list.append(stems)

# 프로파일 생성
term_set = set()

for stems in tqdm(tokens_list, desc=f"{"BoW 생성":15}"):
    term_set.update(stems)

term_list = list(term_set)

profiles = []

for stems in tqdm(stems_list, desc=f"{"DTM 생성":15}"):
    counter = Counter(stems)
    profile = {term: counter[term] for term in term_list}
    profiles.append(profile)

# print(pd.DataFrame(profiles))

# TF 계산 및 출력
tf = sum((Counter(profile) for profile in profiles), Counter())

with open(TF_FILE_PATH, mode='w', encoding='utf-8') as outfile:
    # csv 파일 헤더 작성
    outfile.write("term,frequency\n")
    # tf가 높은 순으로 작성
    for term, frequency in tqdm(tf.most_common(), desc=f"{"출력 생성":15}"):
        outfile.write(f"{term},{frequency}\n")

# TF-IDF 계산 및 출력
# 전체 문서 수
num_documents = len(profiles)

# 각 단어의 IDF를 계산합니다.
# 전체 문서 수
num_documents = len(profiles)

# 각 단어의 IDF를 계산합니다.
idf = {}
term_document_count = {term: 0 for term in tf}

# 각 단어가 등장한 문서의 개수를 세어줍니다.
for term in tf:
    document_frequency = sum(1 for profile in profiles if profile[term] > 0)
    term_document_count[term] = document_frequency

# 각 단어의 IDF를 계산합니다.
for term, document_frequency in term_document_count.items():
    idf[term] = math.log(num_documents / (1 + document_frequency))

# TF-IDF를 계산합니다.
tf_idf = {term: frequency * idf[term] for term, frequency in tf.items()}

# TF-IDF를 내림차순으로 정렬합니다.
sorted_tf_idf = sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)

# TF-IDF를 파일로 출력합니다.
with open(TF_IDF_FILE_PATH, mode='w', encoding='utf-8') as outfile:
    # csv 파일 헤더 작성
    outfile.write("term,tf-idf\n")
    # tf-idf가 높은 순으로 작성
    for term, tf_idf_value in tqdm(sorted_tf_idf, desc="출력 생성"):
        outfile.write(f"{term},{tf_idf_value}\n")
