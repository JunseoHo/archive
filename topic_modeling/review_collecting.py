from urllib.parse import quote
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import re
import requests
import time

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    사용자 지정 변수
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# 리뷰를 가져올 앱의 아이디 리스트의 파일 경로
IN_FILE_PATH = "data/app_data.csv"

# recent - 생성 시간으로 정렬
# updated - 마지막 업데이트 시간으로 정렬
# all - (기본값) 유용함에 따라 정렬, DAY_RANGE 매개변수에 따라 슬라이딩 윈도우 방식으로 탐색하며 항상 결과값을 반환합니다
QRY_FILTER = "all"

# 리뷰를 작성한 사용자가 선택한 언어권
QRY_LANGUAGE = "english"

# 오늘부터 n일 전까지의 유용한 평가를 찾습니다
# FILTER의 값이 'all' 일 때만 적용할 수 있습니다
# 최대 값은 365입니다
QRY_DAY_RANGE = 365

# all - (기본값) 모든 평가
# positive - 긍정적인 평가만
# negative - 부정적인 평가만
QRY_REVIEW_TYPE = "negative"

# all – 모든 평가
# non_steam_purchase – 앱을 스팀에서 구매하지 않은 사람의 평가만 (베타 테스터, 선물 받은 사람 등)
# steam – (기본값) 앱을 스팀에서 구매한 사람의 평가만
QRY_PURCHASE_TYPE = "all"

# API 호출 시 반환되는 리뷰의 개수
# 기본 값은 20이며 최대 값은 100입니다
QRY_NUM_PER_PAGE = 100

# 0 = 게임과 관련 없는 리뷰를 포함
# 1 = 게임과 관련 없는 리뷰를 제외
# 리뷰 폭탄과 같이 게임과 관련 없는 내용의 리뷰 제외 여부이며 기본 값은 1입니다
QRY_FILTER_OFFTOPIC_ACTIVITY = 1

# 앱 하나당 수집할 리뷰의 최대 개수
# None 일 경우 수집할 수 있는 모든 리뷰를 수집합니다
MAX_COUNT = 50

# 외부에 출력할 파일 이름
OUT_FILE_PATH = "data/reviews.csv"

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    스크립트 상수
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

BASE_URL = "https://store.steampowered.com/appreviews/"

RECOMMENDATION_ID = "recommendationid"
AUTHOR = "author"
STEAM_ID = "steamid"
NUM_GAME_OWNED = "num_games_owned"
NUM_REVIEWS = "num_reviews"
PLAYTIME_FOREVER = "playtime_forever"
PLAYTIME_LAST_TWO_WEEKS = "playtime_last_two_weeks"
PLAYTIME_AT_REVIEW = "playtime_at_review"
LAST_PLAYED = "last_played"
LANGUAGE = "language"
REVIEW = "review"
TIMESTAMP_CREATED = "timestamp_created"
TIMESTAMP_UPDATED = "timestamp_updated"
VOTED_UP = "voted_up"
VOTES_UP = "votes_up"
VOTES_FUNNY = "votes_funny"
WEIGHTED_VOTE_SCORE = "weighted_vote_score"
COMMENT_COUNT = "comment_count"
STEAM_PURCHASE = "steam_purchase"
RECEIVED_FOR_FREE = "received_for_free"
WRITTEN_DURING_EARLY_ACCESS = "written_during_early_access"

ANSI_RED = '\x1b[38;2;247;84;100m'
ANSI_NC = '\x1b[0m'

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    사용자 정의 함수
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


# 리뷰 데이터를 csv 포맷으로 저장할 수 있도록 전처리하는 함수
def preprocess_review(data):
    # 줄바꿈 문자 삭제
    data[REVIEW] = data[REVIEW].replace("\r", '').replace("\n", '')

    # 마크다운 태그 삭제
    data[REVIEW] = re.sub(r'\[(h1|h2|h3|b|u|i|strike|spoiler|hr|noparse|tr|th|td)]', '', data[REVIEW])
    data[REVIEW] = re.sub(r'\[/(h1|h2|h3|b|u|i|strike|spoiler|hr|noparse|tr|th|td)]', '', data[REVIEW])
    data[REVIEW] = re.sub(r'\[url=[^\]]*\]|\[/url\]', '', data[REVIEW])

    # 알파벳, 숫자 외의 문자를 공백으로 치환
    data[REVIEW] = re.sub(r'[^a-zA-Z0-9]', ' ', data[REVIEW])

    # 모든 알파벳을 소문자로 치환하고 양 쪽 끝의 공백을 제거
    data[REVIEW] = data[REVIEW].lower().strip()

    # 유닉스 시간 포맷을 년-월-일 시간 포맷으로 변경
    data[LAST_PLAYED] = datetime.fromtimestamp(data[LAST_PLAYED]).strftime("%Y-%m-%d")
    data[TIMESTAMP_CREATED] = datetime.fromtimestamp(data[TIMESTAMP_CREATED]).strftime("%Y-%m-%d")
    data[TIMESTAMP_UPDATED] = datetime.fromtimestamp(data[TIMESTAMP_UPDATED]).strftime("%Y-%m-%d")

    return data


def make_request(url, cursor):
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'
    }
    while True:
        try:
            response = requests.get(url=f"{url}&cursor={cursor}", headers=headers)
            # 요청이 성공했을 때 응답 반환
            if response.status_code == 200:
                return response
            # 요청이 실패했을 때는 잠시 대기 후 다시 시도
            else:
                time.sleep(5)
        except requests.exceptions.RequestException as e:
            # 네트워크 오류 등으로 인한 요청 실패 시 잠시 대기 후 다시 시도
            time.sleep(5)


# API를 호출하여 앱의 리뷰 정보를 담은 딕셔너리의 리스트로 반환하는 함수
def get_reviews(app_id, filter, language, day_range, review_type,
                purchase_type, num_per_page, filter_offtopic_activity, max_count):
    # API 를 호출할 URL 생성
    url = (BASE_URL +
           f"{app_id}?"
           f"json=1"
           f"&filter={filter}"
           f"&language={language}"
           f"&day_range={day_range}"
           f"&review_type={review_type}"
           f"&purchase_type={purchase_type}"
           f"&num_per_page={num_per_page}"
           f"&filter_offtopic_activity={filter_offtopic_activity}")

    # 여러 개의 리뷰를 가져오기 위한 커서 (첫번째 커서는 '*' 로 약속됨)
    cursor = "*"
    cursor_histories = []
    # 앱의 리뷰 정보를 담을 딕셔너리의 리스트
    reviews = []

    while (True):
        # API 요청
        response = make_request(url, cursor)

        # HTTP 응답의 상태 코드가 200이 아니면 함수 종료
        if response.status_code != 200:
            return None

        # HTTP 응답의 바디를 json 으로 변환
        json = response.json()

        # API의 'success' 값이 1이 아니면 정상적인 응답 생성에 실패한 것이므로 함수 종료
        if json['success'] != 1:
            return None

        # 반한된 리뷰의 개수가 0 이면 앱의 모든 리뷰를 반환한 것이므로 함수 종료
        if (json['query_summary']['num_reviews'] == 0):
            return reviews

        # 리뷰 데이터들을 딕셔너리로 변환하여 리스트에 저장
        for review in json['reviews']:
            # 리뷰 데이터를 csv 포맷으로 저장할 수 있도록 전처리
            preprocessed_review = preprocess_review({
                RECOMMENDATION_ID: review[RECOMMENDATION_ID],
                STEAM_ID: review[AUTHOR][STEAM_ID],
                NUM_GAME_OWNED: review[AUTHOR][NUM_GAME_OWNED],
                NUM_REVIEWS: review[AUTHOR][NUM_REVIEWS],
                PLAYTIME_FOREVER: review[AUTHOR][PLAYTIME_FOREVER],
                PLAYTIME_LAST_TWO_WEEKS: review[AUTHOR][PLAYTIME_LAST_TWO_WEEKS],
                PLAYTIME_AT_REVIEW: review[AUTHOR].get(PLAYTIME_AT_REVIEW, 0),  # 생략되어 있는 경우에는 0으로 지정
                LAST_PLAYED: review[AUTHOR][LAST_PLAYED],
                LANGUAGE: review[LANGUAGE],
                REVIEW: review[REVIEW],
                TIMESTAMP_CREATED: review[TIMESTAMP_CREATED],
                TIMESTAMP_UPDATED: review[TIMESTAMP_UPDATED],
                VOTED_UP: review[VOTED_UP],
                VOTES_UP: review[VOTES_UP],
                VOTES_FUNNY: review[VOTES_FUNNY],
                WEIGHTED_VOTE_SCORE: review[WEIGHTED_VOTE_SCORE],
                COMMENT_COUNT: review[COMMENT_COUNT],
                STEAM_PURCHASE: review[STEAM_PURCHASE],
                RECEIVED_FOR_FREE: review[RECEIVED_FOR_FREE],
                WRITTEN_DURING_EARLY_ACCESS: review[WRITTEN_DURING_EARLY_ACCESS]
            })
            # 전처리된 리뷰 데이터 딕셔너리를 리스트에 저장 (전처리 결과가 공백인 경우는 저장하지 않음)
            if not preprocessed_review[REVIEW].isspace():
                reviews.append(preprocessed_review)

        # 커서 갱신 : 반드시 URLEncoded 처리가 진행되어야 한다, 여기서는 quote 함수 사용
        cursor = quote(json['cursor'])

        # 지금까지 수집한 리뷰의 개수가 MAX_COUNT 이상이면 함수 종료
        if max_count != None and len(reviews) >= max_count:
            return reviews[:max_count]

        # 현재 커서를 통해 API 를 호출한 적이 있다면 앱의 모든 리뷰를 반환한 것이므로 함수 종료
        if cursor in cursor_histories:
            return reviews

        # 커서 히스토리 갱신
        cursor_histories.append(cursor)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   메인 스크립트
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

print(f"{ANSI_RED}스팀 리뷰 수집을 시작합니다.{ANSI_NC}")

# 리뷰 데이터를 수집할 앱 아이디 리스트를 파일로부터 가져오기
csv = pd.read_csv(IN_FILE_PATH)
app_ids = csv['app_id'].tolist()

# 각 앱마다의 리뷰 데이터를 수집
outfile = open(OUT_FILE_PATH, mode='w', newline='\n', encoding='utf-8')
outfile.write(f"{RECOMMENDATION_ID},"  # csv 파일에 헤더 작성
              f"{STEAM_ID},"
              f"{NUM_GAME_OWNED},"
              f"{NUM_REVIEWS},"
              f"{PLAYTIME_FOREVER},"
              f"{PLAYTIME_LAST_TWO_WEEKS},"
              f"{PLAYTIME_AT_REVIEW},"
              f"{LAST_PLAYED},"
              f"{LANGUAGE},"
              f"{REVIEW},"
              f"{TIMESTAMP_CREATED},"
              f"{TIMESTAMP_UPDATED},"
              f"{VOTED_UP},"
              f"{VOTES_UP},"
              f"{VOTES_FUNNY},"
              f"{WEIGHTED_VOTE_SCORE},"
              f"{COMMENT_COUNT},"
              f"{STEAM_PURCHASE},"
              f"{RECEIVED_FOR_FREE},"
              f"{WRITTEN_DURING_EARLY_ACCESS}\n"
              )

for app_id in tqdm(app_ids, desc=f"{"리뷰 수집":15s}"):
    reviews = get_reviews(app_id,
                          filter=QRY_FILTER,
                          language=QRY_LANGUAGE,
                          day_range=QRY_DAY_RANGE,
                          review_type=QRY_REVIEW_TYPE,
                          purchase_type=QRY_PURCHASE_TYPE,
                          num_per_page=QRY_NUM_PER_PAGE,
                          filter_offtopic_activity=QRY_FILTER_OFFTOPIC_ACTIVITY,
                          max_count=MAX_COUNT)
    for review in reviews:
        outfile.write(f"{review[RECOMMENDATION_ID]},"
                      f"{review[STEAM_ID]},"
                      f"{review[NUM_GAME_OWNED]},"
                      f"{review[NUM_REVIEWS]},"
                      f"{review[PLAYTIME_FOREVER]},"
                      f"{review[PLAYTIME_LAST_TWO_WEEKS]},"
                      f"{review[PLAYTIME_AT_REVIEW]},"
                      f"{review[LAST_PLAYED]},"
                      f"{review[LANGUAGE]},"
                      f"\"{review[REVIEW]}\","
                      f"{review[TIMESTAMP_CREATED]},"
                      f"{review[TIMESTAMP_UPDATED]},"
                      f"{review[VOTED_UP]},"
                      f"{review[VOTES_UP]},"
                      f"{review[VOTES_FUNNY]},"
                      f"{review[WEIGHTED_VOTE_SCORE]},"
                      f"{review[COMMENT_COUNT]},"
                      f"{review[STEAM_PURCHASE]},"
                      f"{review[RECEIVED_FOR_FREE]},"
                      f"{review[WRITTEN_DURING_EARLY_ACCESS]}\n"
                      )

outfile.close()

print(f"{ANSI_RED}수집된 리뷰 데이터가 {OUT_FILE_PATH} 파일에 저장되었습니다.{ANSI_NC}")


