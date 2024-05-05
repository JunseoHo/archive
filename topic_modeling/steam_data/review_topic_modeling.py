import re

import pandas as pd
from nltk import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from tqdm import tqdm
import math
from konlpy.tag import Komoran
from gensim.models.ldamodel import LdaModel
from gensim.models.callbacks import CoherenceMetric
from gensim import corpora
from gensim.models.callbacks import PerplexityMetric

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   사용자 지정 변수
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# 리뷰 데이터가 저장되어 있는 csv 파일의 위치
PREFIX = "negative_"

IN_FILE_PATH = "steam_data/" + PREFIX + "reviews.csv"
OUT_FILE_PATH = "steam_data/" + PREFIX + "topics.html"

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

SW_CUSTOM = ['아서', '지만', '어서', '네요', '라고', '거나', '이런', '그런', '저런', '이것', '저것', '그것', '이렇', '어디',
             '특히', '어느', '때문', '스팀', '건지', '누드', '새끼', '여기', '저기', '거기', '이건', '저건', '그건', '대부분', '나름', '무엇', '이다',
             '시발', '제가', '조금', '누구', '자체', '병신', "씨발", '동안', '이랑', '정도', '하나', '이후', '스티커', '이거', '저거', '그거', '경우', '부분']

# STOPWORDS = SW_NLTK + SW_ABUSE + SW_CUSTOM

SW_CUSTOM = ['아서', '지만', '어서', '네요', '라고', '거나', '이런', '그런', '저런', '이것', '저것', '그것', '이렇', '어디',
             '특히', '어느', '때문', '스팀', '건지', '누드', '새끼', '여기', '저기', '거기', '이건', '저건', '그건', '대부분', '나름', '무엇', '이다',
             '시발', '제가', '조금', '누구', '자체', '병신', "씨발", '동안', '이랑', '정도', '하나', '이후', '스티커', '이거', '저거', '그거', '경우',
             '부분']

# STOPWORDS = SW_NLTK + SW_ABUSE + SW_CUSTOM
STOPWORDS = SW_CUSTOM

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   치환 사전
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

PRE_REPLACE = [('로그라이트', '로그라이크'), ('로그라이크', 'Roguelike')]

POST_REPLACE = [('재밌', '재미'), ('재미없', '재미'), ('느끼', '느낌'), ('타임', '시간'), ('이야기', '스토리'), ('지역', '스테이지')]

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
    stems = [stem[0] for stem in stems if stem[1] in ['NNG', 'NNP', 'SL']]
    stems = [stem for stem in stems if len(stem) != 1]
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

dictionary = corpora.Dictionary(stems_list)

dictionary.filter_extremes(no_below=2, no_above=0.5)

corpus = [dictionary.doc2bow(text) for text in stems_list]
temp = dictionary[0]
id2word = dictionary.id2token
ldaModel = LdaModel(corpus=corpus,num_topics=4, id2word=id2word,chunksize=2000, passes=10, iterations=400)

top_topics = ldaModel.top_topics(corpus) #, num_words=20)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / 5
print('Average topic coherence: %.4f.' % avg_topic_coherence)

from pprint import pprint
pprint(top_topics)

import pickle
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt

lda_visualization = gensimvis.prepare(ldaModel, corpus, dictionary, sort_topics=False)
pyLDAvis.save_html(lda_visualization, OUT_FILE_PATH)