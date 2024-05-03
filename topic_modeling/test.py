from konlpy.tag import Komoran


def extract_stem(text):
    mecab = Komoran()
    morphemes = mecab.pos(text)

    return morphemes


# 텍스트 입력
text = "로구라이크 액션이니까  이 장르에 대한 생각 먼저 적고"
# 어간 추출 실행
result = extract_stem(text)

# 결과 출력
print(result)
