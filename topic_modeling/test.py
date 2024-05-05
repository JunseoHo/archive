import re

from konlpy.tag import Komoran


# def extract_stem(text):
#     mecab = Komoran()
#     morphemes = mecab.pos(text)
#
#     return morphemes
#
#
# # 텍스트 입력
# text = "내가 조종하는 캐릭이 약함"
# # 어간 추출 실행
# result = extract_stem(text)
#
# # 결과 출력
# print(result)

import re

sentence = "별표 5개 만점에 4개를 받았습니다."

# 정규 표현식을 사용하여 숫자 추출
match = re.search(r'만점에 (\d+)개를 받았습니다', sentence)

if match:
    number = match.group(1)  # 첫 번째 괄호 안에 있는 것이 숫자에 해당
    print("숫자:", number)
else:
    print("숫자를 찾을 수 없습니다.")




