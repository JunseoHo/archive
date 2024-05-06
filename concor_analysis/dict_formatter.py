"""
    dict.txt 파일의 공백을 탭 문자로 치환하는 스크립트
"""

from konlpy.tag import Komoran

file_name = 'dict.txt'

with open(file_name, 'r', encoding='utf-8') as file:
    lines = file.readlines()
with open(file_name, 'w', encoding='utf-8') as file:
    for line in lines:
        tokens = line.split()
        if len(tokens) < 2:
            file.write(line)
        else:
            file.write(f'{tokens[0]}\t{tokens[1]}\n')

# dict.txt 파일이 정상적으로 저장되었는지 확인하세요!
kom = Komoran(userdic='dict.txt')
print(kom.pos('NPC가 예쁘다'))
