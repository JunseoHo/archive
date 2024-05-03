from konlpy.tag import Komoran


def extract_stem(text):
    mecab = Komoran()
    morphemes = mecab.pos(text)

    return morphemes


# 텍스트 입력
text = "마음에 들지는 않아서"

# 어간 추출 실행
result = extract_stem(text)

# 결과 출력
print(result)
