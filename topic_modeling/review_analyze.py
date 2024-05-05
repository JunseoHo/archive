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
# Komoran 사용자 사전 경로
KOMORAN_USER_DICT = "./komoran_user.dict"

# 형태소 분석에서 추출할 품사 코드 (Komoran)
POSLIST = ['NNG', 'NNP']

# 도메인 용어
CASH = "캐시"  # 게임 내에서 획득이 불가능하며 과금으로만 획득 가능한 재화
GACHA = "가챠"  # 확률적으로 캐릭터나 아이템을 획득할 수 있는 상품
FATIGUE = "피로도"  # 게임 과몰입을 막기 위해 게임을 일정 시간 이상 플레이하면 플레이어에게 불이익을 주는 시스템
COLLABORATION = "콜라보레이션"  # 타 IP를 일시적으로 빌려 게임 내 컨텐츠로 추가하는 행위
PACKAGE = "패키지"  # 게임 내에서 캐시로 구매할 수 있는 상품
RESOURCE = "자원"  # 게임 내에서 획득할 수 있는 재화
CONTENTS = "콘텐츠"  # 게임 내에서 플레이 가능한 요소
CHARACTER = "캐릭터"
ITEM = "아이템"

# 공통 불용어
COMMON_STOPWORDS = ['시발', '씨발', '병신', '구글스토어', '구글', '스토어', '해주시', '이제']
# 공통 한단어 사전
COMMON_MONODICT = ['운', '돈', '말', '섭', '폰', '맘', '답', '팀', '욕', '맵']
# 공통 후치환
COMMON_REPLDICT = [('행동력', FATIGUE), ('능지', '지능'), ('콜라보', COLLABORATION), ('별점', '평점'),
                   ('다운', '다운로드'), ('가차', GACHA), ('픽업', GACHA), ('뽑기', GACHA), ('스킨', '코스튬'), ('오픈', '출시'),
                   ('컨트롤', '조작'),
                   ('스코어', '점수'), ('애니', '애니메이션'), ('리세', '리세마라'),
                   ('재료', '자원'), ('섭', '서버'), ('폰', '모바일'), ('맘', '마음'),
                   ('게관위', '게임물관리위원회'), ('겜관위', '게임물관리위원회'), ('사료', CASH), ('재화', '자원'), ('현질', '과금'), ('노래', '음악'),
                   ('서브컬쳐', '서브컬처'), ('에러', '오류'), ('초보자', '뉴비'), ('초보', '뉴비'), ('레어템', ITEM), ('레어도', '등급'),
                   ('티어', '등급'), ('숙제', '일일퀘스트'), ('일퀘', '일일퀘스트'), ('제화', '재화'), ('오토', '자동'), ('신규유저', '뉴비'),
                   ('이야기', '스토리'), ('헬적화', '최적화'), ('맵', '스테이지'), ('픽뚫', GACHA), ('수정', '업데이트'), ('업뎃', '업데이트')]

# 명일방주 데이터 파일 위치
ARKNIGHTS_INFILE = "review/arknights.csv"
# 명일방주 불용어
ARKNIGHTS_STOPWORDS = ['명일', '방주', '명일방주']
# 명일방주 한단어 사전
ARKNIGHTS_MONODICT = ['돌']
# 명일방주 후치환
ARKNIGHTS_REPLDICT = [('합성옥', CASH), ('오리지늄', CASH), ('헤드헌팅', GACHA), ('이성', FATIGUE), ('용문폐', RESOURCE), ('돌', CASH),
                      ('타워디펜스', '디펜스'), ('협약', CONTENTS), ('작전', CONTENTS), ('섬멸전', CONTENTS), ('섬멸작전', CONTENTS),
                      ('대리지휘', '자동'), ('오퍼레이터', CHARACTER), ('오퍼', CHARACTER), ('훈장', '업적'), ('위기협약', CONTENTS),
                      ('로그인', '접속'), ('유저', '플레이어'), ('무자본', '무과금')]

# 블루아카이브 데이터 파일 위치
BLUEARCHIVE_INFILE = "review/bluearchive.csv"
# 블루아카이브 불용어
BLUEARCHIVE_STOPWORDS = ['블루', '아카이브', '블루아카이브', '신이', '미카', '시노', '블루아']
# 명일방주 한단어 사전
BLUEARCHIVE_MONODICT = []
# 블루아카이브 전치환
BLUEARCHIVE_REPLDICT = [('김용하', '개발자'), ('용하', '개발자'), ('넥슨', '개발자'), ('청휘석', CASH), ('휘석', CASH), ('학생', CHARACTER),
                        ('총력전', CONTENTS),
                        ('전술대회', CONTENTS), ('대항전', CONTENTS), ('전술대항전', CONTENTS), ('만원', '돈'), ('연차', GACHA),
                        ('메모리얼', '일러스트')]

# 원신 데이터 파일 위치
GENSHIN_INFILE = "review/genshin.csv"
# 원신 불용어
GENSHIN_STOPWORDS = ['원신', '호요버스', '스타레일']
# 원신 한단어 사전
GENSHIN_MONODICT = []
# 원신 전치환
GENSHIN_REPLDICT = [('원석', CASH), ('성유물', ITEM), ('비경', CONTENTS), ('나선비경', CONTENTS), ('회사', '게임사')]

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   스크립트 함수
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def get_profiles(infile, stopwords, monodict, repldict):
    stopwords += COMMON_STOPWORDS
    monodict += COMMON_MONODICT
    repldict += COMMON_REPLDICT

    csv = pd.read_csv(infile)
    reviews = csv['review'].tolist()

    tokenizer = TreebankWordTokenizer()  # 토크나이저를 수정하려면 이곳을 수정
    tokens_list = []
    for review in tqdm(reviews, desc=f"{infile}, 토큰화"):
        tokens_list.append(tokenizer.tokenize(review))

    for index, tokens in tqdm(enumerate(tokens_list), desc=f"{infile}, 불용어 제거"):
        tokens_list[index] = [token for token in tokens if token not in stopwords]

    stemmer = Komoran(userdic=KOMORAN_USER_DICT)
    stems_list = []
    for tokens in tqdm(tokens_list, desc=f"{infile}, 형태소 분석"):
        stems = []
        for token in tokens:
            stems += stemmer.pos(token)
        stems = [stem[0] for stem in stems if stem[1] in POSLIST]
        stems = [stem for stem in stems if len(stem) != 1 or stem in monodict]
        stems = [stem for stem in stems if stem not in stopwords]  # 형태소 분석 이후에도 불용어 제거 수행

        replaced_stems = []  # 후치환
        for stem in stems:
            replaced = False
            for repl in repldict:
                if stem == repl[0]:
                    replaced = True
                    replaced_stems.append(repl[1])
                    break
            if replaced is False:
                replaced_stems.append(stem)
        stems_list.append(replaced_stems)

    term_set = set()
    for stems in tqdm(stems_list, desc=f"{infile}, 단어 가방 생성"):
        term_set.update(stems)
    term_set = list(term_set)

    profiles = []
    for stems in tqdm(stems_list, desc=f"{infile}, 프로파일 생성"):
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

arknights_profiles = get_profiles(ARKNIGHTS_INFILE, ARKNIGHTS_STOPWORDS, ARKNIGHTS_MONODICT, ARKNIGHTS_REPLDICT)

# bluearchive_profiles = get_profiles(BLUEARCHIVE_INFILE, BLUEARCHIVE_STOPWORDS, BLUEARCHIVE_MONODICT,
#                                     BLUEARCHIVE_REPLDICT)
#
# genshin_profiles = get_profiles(GENSHIN_INFILE, GENSHIN_STOPWORDS, GENSHIN_MONODICT, GENSHIN_REPLDICT)

profiles = arknights_profiles

analyze(profiles)
