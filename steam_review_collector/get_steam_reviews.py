import sys
from urllib.parse import quote
import requests
from datetime import datetime
from console_utils import *

"""
    API Keys
    Find detailed API specifications here : https://partner.steamgames.com/doc/store/getreviews
"""
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


def preprocess_steam_review(review):
    review[REVIEW] = (review[REVIEW]
                      .lower()
                      .replace("\n", "\\n")
                      .replace("\r", "\\r")
                      .replace("\"", "")
                      .replace(",", ""))
    review[LAST_PLAYED] = datetime.fromtimestamp(review[LAST_PLAYED]).strftime("%Y-%m-%d")
    review[TIMESTAMP_CREATED] = datetime.fromtimestamp(review[TIMESTAMP_CREATED]).strftime("%Y-%m-%d")
    review[TIMESTAMP_UPDATED] = datetime.fromtimestamp(review[TIMESTAMP_UPDATED]).strftime("%Y-%m-%d")

    return review


def get_steam_reviews(appid, filter="all", language="all", day_range=30, review_type="all",
                      purchase_type="all", num_per_page=20, filter_offtopic_activity=1):
    url = (f"https://store.steampowered.com/appreviews/"
           f"{appid}?"
           f"json=1"
           f"&filter={filter}"
           f"&language={language}"
           f"&day_range={day_range}"
           f"&review_type={review_type}"
           f"&purchase_type={purchase_type}"
           f"&num_per_page={num_per_page}"
           f"&filter_offtopic_activity={filter_offtopic_activity}")
    cursor = "*"
    steam_reviews = []

    print_console(f"Start fetching steam reviews... (App ID : {appid})", color="blue")

    while (True):
        response = requests.get(f"{url}&cursor={cursor}")

        if response.status_code != 200:
            sys.stderr.write(f"API error: HTTP status is {response.status_code}\n")
            return None

        json = response.json()

        if json['success'] != 1:
            sys.stderr.write(f"API error: Success value is {json['success']}, {json['error']}\n")
            return None

        if (json['query_summary']['num_reviews'] == 0):
            print_console(f"Retrieved all {len(steam_reviews)} steam reviews! (App ID : {appid})", color="green")
            return steam_reviews

        for review in json['reviews']:
            steam_reviews.append(preprocess_steam_review({
                RECOMMENDATION_ID: review[RECOMMENDATION_ID],
                STEAM_ID: review[AUTHOR][STEAM_ID],
                NUM_GAME_OWNED: review[AUTHOR][NUM_GAME_OWNED],
                NUM_REVIEWS: review[AUTHOR][NUM_REVIEWS],
                PLAYTIME_FOREVER: review[AUTHOR][PLAYTIME_FOREVER],
                PLAYTIME_LAST_TWO_WEEKS: review[AUTHOR][PLAYTIME_LAST_TWO_WEEKS],
                PLAYTIME_AT_REVIEW: review[AUTHOR].get(PLAYTIME_AT_REVIEW, 0),
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
            }))
        cursor = quote(json['cursor'])


def get_steam_reviews_all(appids, filter="all", language="all", day_range=30, review_type="all",
                          purchase_type="all", num_per_page=20, filter_offtopic_activity=1):
    steam_reviews = []
    for appid in appids:
        steam_reviews += get_steam_reviews(appid, filter=filter, language=language, day_range=day_range,
                                           review_type=review_type, num_per_page=num_per_page,
                                           filter_offtopic_activity=filter_offtopic_activity)

    print_console("\n*** Completed ***", color="green")
    return steam_reviews


def out_steam_reviews(file_name, steam_reviews):
    with open("steam_reviews.csv", mode='w', newline='\n', encoding='utf-8') as file:
        file.write(f"{RECOMMENDATION_ID},"
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
        for review in steam_reviews:
            file.write(f"{review[RECOMMENDATION_ID]},"
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
