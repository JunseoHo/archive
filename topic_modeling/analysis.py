import math
import re
from collections import Counter

import pandas as pd
from konlpy.tag import Komoran
from nltk import TreebankWordTokenizer
from tqdm import tqdm

"""
재화 : 게임 외부적으로 과금을 통해서만 획득할 수 있는 재화
자원 : 게임 내에서 별도의 과금 없이 획득 가능한 재화 
가챠 : 확률형 아이템
리세마라 : 계정 생성 시 원하는 캐릭터 또는 아이템을 가진 상태로 게임을 플레이 하기 위해 원하는 캐릭터, 아이템을 얻을 때까지 계정생성-가챠-계정삭제를 반복하는 행위
콜라보레이션 : 타 IP를 빌려와 게임 내 컨텐츠로 추가하는 행위
업적 :게임의 변수 바깥에서 정의된 메타(meta)성 목표이며, 비디오 게임 내에서 플레이어가 특정 작업이나 과제를 숙달했음을 나타내는 디지털 보상
피로도 : 게임에서 게임에 지나치게 과몰입하는 현상을 막기 위한 목적으로, 게임을 일정 시간 이상 지속적으로 플레이하면 경험치를 깎아버리거나 아예 사냥을 못 뛰게 만드는 등 플레이어에게 불이익을 주는 모든 시스템
"""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   스크립트 상수
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# 공통 불용어
COMMON_STOPWORDS = ['제가', '구글', '스토어', '이번', '이랑', '앞으로', '지금', '인가', '해주시', '화이팅']
# 공통 전치환
COMMON_PREREPLS = [('클리어', 'Clear'), ('콜라보레이션', 'Collaboration'), ('콜라보', 'Collaboration'), ('픽업', 'Gacha'),
                   ('뽑기', 'Gacha'), ('가챠', 'Gacha'), ('가차', 'Gacha'), ('꿀잼', '재미'), ('노잼', '재미'),
                   ('월정액', 'MonthlyPlan'), ('애니', "Animation"), ('애니메이션', 'Animation'), ('리세', 'Risemara'),
                   ('리세마라', 'Risemara'), ('퀘스트', 'Quest'), ('일퀘', 'Quest'), ('스킬', '기술'), ('오토', '자동'), ('핸드폰', '모바일'),
                   ('컨텐츠', '콘텐츠'), ('그림', 'Illust'), ('게관위', 'GRAC'), ('겜관위', 'GRAC'), ('게임물관리위원회', 'GRAC'),
                   ('연차', '가차'), ('패치', '업데이트')]
# 공통 한단어 사전
COMMON_MONODICT = ['운', '돈', '말', '섭', '폰', '맘', '답', '팀', '욕']
# 공통 후치환
COMMON_POSTREPLS = [('Clear', '클리어'), ('행동력', '피로도'), ('능지', '지능'), ('Collaboration', '콜라보레이션'), ('별점', '평점'),
                    ('다운', '다운로드'), ('Gacha', '가챠'), ('스킨', '코스튬'), ('오픈', '출시'), ('컨트롤', '조작'),
                    ('스코어', '점수'), ('MonthlyPlan', '월정액'), ('Animation', '애니메이션'), ('Risemara', '리세마라'), ('훈장', '업적'),
                    ('Quest', '퀘스트'), ('재료', '자원'), ('섭', '서버'), ('폰', '모바일'), ('맘', '마음'), ('Illust', '일러스트'),
                    ('GRAC', '게임물관리위원회'), ('사료', '재화')]

# 명일방주 데이터 파일 위치
ARKNIGHTS_INFILE = "review/arknights.csv"
# 명일방주 불용어
ARKNIGHTS_STOPWORDS = ['명일', '방주', '명일방주']
# 명일방주 전치환
ARKNIGHTS_PREREPL = [('합성옥', '재화'), ('오리지늄', '재화'), ('헤드헌팅', 'Gacha'), ('패키지', '상품')]
# 명일방주 한단어 사전
ARKNIGHTS_MONODICT = ['돌']
# 명일방주 후치환
ARKNIGHTS_POSTREPL = [('이성', '피로도'), ('용문폐', '자원'), ('돌', '재화')]

# 블루아카이브 데이터 파일 위치
BLUEARCHIVE_INFILE = "review/bluearchive.csv"
# 블루아카이브 불용어
BLUEARCHIVE_STOPWORDS = ['블루', '아카이브', '블루아카이브', '신이', '미카', '시노', '블루아']
# 블루아카이브 전치환
BLUEARCHIVE_PREREPL = [('김용하', '개발자'), ('용하', '개발자'), ('넥슨', '개발자'), ('청휘석', '재화'), ('학생', '캐릭터'), ('총력전', '콘텐츠'),
                       ('메모리얼', 'Illust'), ('전술대회', '콘텐츠'), ('전술대회', '콘텐츠'), ('대항전', '콘텐츠'), ('전술대항전', '콘텐츠')]
# 블루아카이브 한단어 사전
BLUEARCHIVE_MONODICT = []
# 블루아카이브 후치환
BLUEARCHIVE_POSTREPL = [('사랑해', '사랑'), ('만원', '돈')]

# 형태소 분석에서 추출할 품사 코드 (Komoran)
POSLIST = ['NNG', 'NNP', 'SL']

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   스크립트 함수
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def get_profiles(infile, stopwords, prerepls, postrepls, monodict):
    stopwords += COMMON_STOPWORDS
    prerepls += COMMON_PREREPLS
    postrepls += COMMON_POSTREPLS
    monodict += COMMON_MONODICT

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
        stems = [stem[0] for stem in stems if stem[1] in POSLIST]
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


def analyze(profiles):
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
        df = sum(1 for profile in profiles if profile.get(term) is not None and profile[term] > 0)
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
    sum_of_tf = 0
    for term, tf in tf_table.items():
        sum_of_tf += tf

    sum_of_tf = sum_of_tf * 0.001

    for term, tf_idf in tqdm(tf_idf_table, desc="TF가 0.1% 미만인 단어 제거"):
        if tf_table[term] > sum_of_tf:
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


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   메인 스크립트
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

arknights_profiles = get_profiles(ARKNIGHTS_INFILE, ARKNIGHTS_STOPWORDS, ARKNIGHTS_PREREPL, ARKNIGHTS_POSTREPL,
                                  ARKNIGHTS_MONODICT)

bluearchive_profiles = get_profiles(BLUEARCHIVE_INFILE, BLUEARCHIVE_STOPWORDS, BLUEARCHIVE_PREREPL,
                                    BLUEARCHIVE_POSTREPL,
                                    BLUEARCHIVE_MONODICT)

profiles = arknights_profiles + bluearchive_profiles

analyze(profiles)
