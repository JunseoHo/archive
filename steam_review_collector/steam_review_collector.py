import sys
from urllib.parse import quote
import requests
from datetime import datetime
from enum import Enum

# class SteamReview:
#     def __init__(self,
#                  recommendation_id,
#                  steam_id,
#                  num_games_owned,
#                  num_reviews,
#                  playtime_forever,
#                  playtime_last_two_weeks,
#                  playtime_at_review,
#                  last_played,
#                  language,
#                  review,
#                  timestamp_created,
#                  timestamp_updated,
#                  voted_up,
#                  votes_up,
#                  votes_funny,
#                  weighted_vote_score,
#                  comment_count,
#                  steam_purchase,
#                  received_for_free,
#                  written_during_early_access,
#                  ):
#         self.recommendation_id = recommendation_id
#         self.steam_id = steam_id
#         self.num_games_owned = num_games_owned
#         self.num_reviews = num_reviews
#         self.playtime_forever = playtime_forever
#         self.playtime_last_two_weeks = playtime_last_two_weeks
#         self.playtime_at_review = playtime_at_review
#         self.last_played = last_played
#         self.language = language
#         self.review = review
#         self.timestamp_created = timestamp_created
#         self.timestamp_updated = timestamp_updated
#         self.voted_up = voted_up
#         self.votes_up = votes_up
#         self.votes_funny = votes_funny
#         self.weighted_vote_score = weighted_vote_score
#         self.comment_count = comment_count
#         self.steam_purchase = steam_purchase
#         self.received_for_free = received_for_free
#         self.written_during_early_access = written_during_early_access


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


def __str__(self):
    return self.value


# review["recommendationid"],
# review["author"]["steamid"],
# review["author"]["num_games_owned"],
# review["author"]["num_reviews"],
# review["author"]["playtime_forever"],
# review["author"]["playtime_last_two_weeks"],
# review["author"]["playtime_at_review"],
# review["author"]["last_played"],
# review["language"],
# review["review"],
# review["timestamp_created"],
# review["timestamp_updated"],
# review["voted_up"],
# review["votes_up"],
# review["votes_funny"],
# review["weighted_vote_score"],
# review["comment_count"],
# review["steam_purchase"],
# review["received_for_free"],
# review["written_during_early_access"],

def get_steam_reviews_by_api(appid=None, filter="all", language="english", day_range=365, review_type="all",
                             purchase_type="all", num_per_page=20, filter_offtopic_activity=1):
    if (appid == None):
        raise ValueError("appid is required")
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
    cursor_histories = []
    reviews = []
    while (True):
        response = requests.get(f"{url}&cursor={cursor}")
        if response.status_code != 200:
            raise ConnectionError(f"API error: HTTP status is {response.status_code}")
        json = response.json()
        print(json)
        if json['success'] != 1:
            sys.stderr.write(f"API error: Success value is {json['success']}, {json['error']}\n")
            return
        for review in json['reviews']:
            reviews.append({
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
            })
        if json['cursor'] in cursor_histories:
            break
        cursor_histories.append(json['cursor'])
        cursor = quote(json['cursor'])
    return reviews


def preprocess(reviews):
    for review in reviews:
        review[REVIEW] = (review[REVIEW]
                          .lower()
                          .replace("\n", "\\n")
                          .replace("\r", "\\r")
                          .replace("\"", "")
                          .replace(",", ""))
        review[LAST_PLAYED] = datetime.fromtimestamp(review[LAST_PLAYED])
        review[TIMESTAMP_CREATED] = datetime.fromtimestamp(review[TIMESTAMP_CREATED])
        review[TIMESTAMP_UPDATED] = datetime.fromtimestamp(review[TIMESTAMP_UPDATED])

reviews = get_steam_reviews_by_api(appid=568220,
                                   filter="recent",
                                   language="english",
                                   review_type="negative",
                                   num_per_page=50)

print(len(reviews))

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
    preprocess(reviews)
    for review in reviews:
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
