"""
    https://store.steampowered.com/search/?sort_by=Reviews_DESC&tags=492%2C122&category1=998&supportedlang=koreana&ndl=1
"""

import requests
from bs4 import BeautifulSoup
import re


def get_href_from_search_results(url):
    try:
        # URL에서 HTML 가져오기
        response = requests.get(url)
        response.raise_for_status()  # 요청이 성공적으로 이루어졌는지 확인
        html = response.text

        # BeautifulSoup를 사용하여 HTML 파싱
        soup = BeautifulSoup(html, 'html.parser')

        # id가 'search_resultsRows'인 태그 찾기
        search_results = soup.find(id='search_resultsRows')

        # 찾은 태그 내의 모든 'a' 태그에서 href 속성 값 추출하여 리스트에 저장
        href_list = [a['href'] for a in search_results.find_all('a') if 'href' in a.attrs]

        return href_list

    except Exception as e:
        print("Error:", e)
        return None


# URL 입력 받기
url = "https://store.steampowered.com/search/?sort_by=Reviews_DESC&tags=492%2C122&category1=998&supportedlang=koreana&ndl=1"

# get_href_from_search_results 함수 호출하여 href 속성 값 리스트 가져오기
href_values = get_href_from_search_results(url)

# href 속성 값 리스트 출력
print(len(href_values))
if href_values:
    print("href 속성 값 리스트:")
    for href in href_values:
        print(href)

# 숫자를 저장할 배열
numbers = []

# 각 href 속성 값에서 숫자 추출하여 배열에 추가
for href in href_values:
    match = re.search(r'/(\d+)/', href)
    if match:
        numbers.append(match.group(1))

# 숫자 배열 출력
print(numbers)