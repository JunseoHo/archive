from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from tqdm import tqdm
import time
import re

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   사용자 지정 변수
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
INPUT_DIR = "html/"
INPUT_NAMES = ['genshin']

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   스크립트 상수
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
OUTPUT_DIR = "review/"

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   메인 스크립트
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

for input_name in tqdm(INPUT_NAMES, 'HTML로부터 리뷰 데이터 크롤링'):
    with open(f"{INPUT_DIR}{input_name}.html", "r", encoding="utf-8") as infile:
        html = infile.read()
    soup = BeautifulSoup(html, 'html.parser')
    author_tags = soup.find_all(class_="X5PpBb")
    date_tags = soup.find_all(class_="bp9Aid")
    review_tags = soup.find_all(class_="h3YV2d")
    rating_tags = soup.find_all(class_="iXRFPc")
    with open(f"{OUTPUT_DIR}{input_name}.csv", "w", encoding="utf-8") as outfile:
        outfile.write("author,date,review,score\n")
        for author_tag, date_tag, review_tag, rating_tag in zip(author_tags, date_tags, review_tags, rating_tags):
            author = author_tag.text.replace(",", ' ')
            date = date_tag.text
            review = review_tag.text.replace("\r", '').replace("\n", '').replace(",", ' ')
            review = re.sub(r'[^가-힣a-zA-Z0-9]', ' ', review)
            rating = re.search(r'만점에 (\d+)개를 받았습니다', rating_tag['aria-label']).group(1)
            outfile.write(f"{author},{date},{review},{rating}\n")
