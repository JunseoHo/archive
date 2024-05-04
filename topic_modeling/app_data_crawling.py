from selenium import webdriver
from selenium.webdriver.common.by import By
from tqdm import tqdm
import time
import re

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    사용자 지정 변수
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# 크롤링을 수행할 스팀 앱 페이지 주소
# 2024년 5월 기준 Top Sellers 로 분류된 인디 태그를 포함하는 게임들
URL = "https://store.steampowered.com/search/?supportedlang=koreana&tags=492&category1=998&filter=topsellers&ndl=1"

# 스크롤 후 페이지 로딩을 기다리는 시간 (단위 : 초)
LOAD_TIME = 5

# 스크롤 횟수 : 스크롤 한번 당 50개의 게임을 추가로 로딩
SCROLL_COUNT = 20

# 외부에 출력할 파일 이름
OUT_FILE_PATH = "data/app_data.csv"

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    스크립트 상수
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

ANSI_RED = '\x1b[38;2;247;84;100m'
ANSI_NC = '\x1b[0m'

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    메인 스크립트
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

print(f"{ANSI_RED}스팀 어플리케이션 데이터 크롤링을 시작합니다.{ANSI_NC}")

# 웹드라이버 초기화
driver = webdriver.Chrome()

# HTTP 요청
driver.get(URL)

# 웹페이지 최하단 좌표 반환
scroll_height = driver.execute_script("return document.body.scrollHeight")

for i in tqdm(range(SCROLL_COUNT), desc=f"{"웹페이지 로딩":15s}"):
    # 웹페이지 최하단으로 스크롤
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    # 웹페이지가 완전히 로딩될 때까지 대기
    time.sleep(LOAD_TIME)
    # 새롭게 로딩이 완료된 웹페이지의 최하단 좌표 반환
    next_scroll_height = driver.execute_script("return document.body.scrollHeight")
    # 더 이상 스크롤할 수 없으면 반복문 종료
    if scroll_height == next_scroll_height:
        break
    # 웹페이지의 최하단 좌표 갱신
    scroll_height = next_scroll_height

# 어플리케이션 리스트에서 각 어플리케이션 페이지의 링크를 href 속성으로 가지고 있는 a 태그 리스트 추출
a_tags = (driver.find_element(by=By.ID, value='search_resultsRows')
          .find_elements(by=By.TAG_NAME, value='a'))

# 각 a 태그의 href 속성으로부터 어플리케이션 데이터 추출
app_data = []

for a_tag in a_tags:
    match = re.match(r"https://store\.steampowered\.com/app/(\d+)/([^/]+)/", a_tag.get_attribute('href'))
    if match is None:
        print(a_tag.get_attribute('href'))
        continue
    app_data.append({"app_id": match.group(1), "app_name": match.group(2)})

# 웹 드라이버 종료
driver.quit()

print(f"{ANSI_RED}{len(app_data)}개의 어플리케이션 데이터가 크롤링되었습니다.{ANSI_NC}")

# csv 파일로 결과 출력
with open(OUT_FILE_PATH, mode='w', encoding='utf-8') as outfile:
    # csv 파일 헤더 작성
    outfile.write("app_id,app_name\n")
    # 어플리케이션 데이터 작성
    for a_tag in app_data:
        outfile.write(f"{a_tag['app_id']},{a_tag['app_name']}\n")

print(f"{ANSI_RED}크롤링된 어플리케이션 데이터가 {OUT_FILE_PATH}에 저장되었습니다.{ANSI_NC}")
