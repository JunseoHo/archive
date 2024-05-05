"""
    탭문자가 제대로 작동하지 않는 시스템에서 Komoran 사용자 사전을 생성하기 위한 스크립트
    Komoran 품사표 : https://docs.komoran.kr/firststep/postypes.html
"""

# userdict에 사용자 사전에 추가하고자 하는 단어를 추가
userdict = [
    ('콜라보레이션', 'NNG'),
    ('클리어', 'NNG'),
    ('콜라보', 'NNG'),
    ('픽업', 'NNG'),
    ('뽑기', 'NNG'),
    ('가챠', 'NNG'),
    ('가차', 'NNG'),
    ('월정액', 'NNG'),
    ('애니', "NNG"),
    ('애니메이션', "NNG"),
    ('리세', 'NNG'),
    ('리세마라', 'NNG'),
    ('퀘스트', 'NNG'),
    ('일퀘', 'NNG'),
    ('주간퀘', 'NNG'),
    ('핸드폰', 'NNG'),
    ('일러스트', 'NNG'),
    ('게관위', 'NNP'),
    ('겜관위', 'NNP'),
    ('게임물관리위원회', 'NNP'),
    ('운', 'NNG'),
    ('돈', 'NNG'),
    ('말', 'NNG'),
    ('섭', 'NNG'),
    ('폰', 'NNG'),
    ('맘', 'NNG'),
    ('팀', 'NNG'),
    ('욕', 'NNG'),
    ('클리어', 'NNG'),
    ('행동력', 'NNG'),
    ('별점', 'NNG'),
    ('운', 'NNG'),
    ('제가', 'NP'),
    ('지금', 'MAG'),
    ('타워디펜스', 'NNG'),
    ('구글스토어', 'NNP'),
    ('스킬', 'NNG'),
    ('요즘', 'MAG'),
    ('헤드헌팅', 'NNG'),
    ('앞으로', 'MAG'),
    ('섬멸전', 'NNG'),
    ('섬멸작전', 'NNG'),
    ('대리지휘', 'NNP'),
    ('라이트', 'NNG'),
    ('이후', 'MAG'),
    ('이번', 'NP'),
    ('청휘석', 'NNP'),
    ('휘석', 'NNP'),
    ('블루아카이브', 'NNP'),
    ('김용하', 'NNP'),
    ('용하', 'NNP'),
    ('사랑', 'NNG'),
    ('메모리얼', 'NNP'),
    ('이랑', 'JKG'),
    ('호요버스', 'NNP'),
    ('성유물', 'NNP'),
    ('스타레일', 'NNP'),
    ('원신', 'NNP'),
    ('나선비경', 'NNP')
]

with open('komoran_user.dict', 'w', encoding="utf-8") as dictfile:
    for userword in userdict:
        dictfile.write(f"{userword[0]}\t{userword[1]}\n")

print("komoran_user.dict에 사용자 사전이 작성되었습니다!")
