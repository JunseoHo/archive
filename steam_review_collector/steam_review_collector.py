import requests
import datetime


class SteamReview:
    def __init__(self,
                 recommendation_id,
                 steam_id,
                 num_games_owned,
                 num_reviews,
                 playtime_forever,
                 playtime_last_two_weeks,
                 playtime_at_review,
                 last_played,
                 language,
                 review,
                 timestamp_created,
                 timestamp_updated,
                 voted_up,
                 votes_up,
                 votes_funny,
                 weighted_vote_score,
                 comment_count,
                 steam_purchase,
                 received_for_free,
                 written_during_early_access,
                 ):
        self.recommendation_id = recommendation_id
        self.steam_id = steam_id
        self.num_games_owned = num_games_owned
        self.num_reviews = num_reviews
        self.playtime_forever = playtime_forever
        self.playtime_last_two_weeks = playtime_last_two_weeks
        self.playtime_at_review = playtime_at_review
        self.last_played = last_played
        self.language = language
        self.review = review
        self.timestamp_created = timestamp_created
        self.timestamp_updated = timestamp_updated
        self.voted_up = voted_up
        self.votes_up = votes_up
        self.votes_funny = votes_funny
        self.weighted_vote_score = weighted_vote_score
        self.comment_count = comment_count
        self.steam_purchase = steam_purchase
        self.received_for_free = received_for_free
        self.written_during_early_access = written_during_early_access


def get_steam_reviews_by_api(
        app_id=None,
        filter="all",
        language="english",
        day_range=365,
        review_type="all",
        purchase_type="all",
        num_per_page=20,
        filter_offtopic_activity=1
):
    url = (f"https://store.steampowered.com/appreviews/"
           f"{app_id}?"
           f"json=1"
           f"&filter={filter}"
           f"&language={language}"
           f"&day_range={day_range}"
           f"&review_type={review_type}"
           f"&purchase_type={purchase_type}"
           f"&num_per_page={num_per_page}"
           f"&filter_offtopic_activity={filter_offtopic_activity}")
    cursor = "*"
    cursor_history = []
    reviews = []
    while (True):
        response = requests.get(url + f"&cursor={cursor}")
        print(url)
        if response.status_code != 200:
            print(f"get steam reviews by api failed: http status is {response.status_code}")
            return None
        json = response.json()
        if json['success'] != 1:
            print(f"get steam reviews by api failed: success value is {json["success"]}")
            return None
        for review in json["reviews"]:
            reviews.append(SteamReview(
                review["recommendationid"],
                review["author"]["steamid"],
                review["author"]["num_games_owned"],
                review["author"]["num_reviews"],
                review["author"]["playtime_forever"],
                review["author"]["playtime_last_two_weeks"],
                review["author"]["playtime_at_review"],
                review["author"]["last_played"],
                review["language"],
                review["review"],
                review["timestamp_created"],
                review["timestamp_updated"],
                review["voted_up"],
                review["votes_up"],
                review["votes_funny"],
                review["weighted_vote_score"],
                review["comment_count"],
                review["steam_purchase"],
                review["received_for_free"],
                review["written_during_early_access"],
            ))
        if json["cursor"] in cursor_history:
            break
        cursor_history.append(json["cursor"])
    return reviews

reviews = get_steam_reviews_by_api(app_id=262060,
                                   language="english",
                                   num_per_page=100)

with open("steam_reviews.csv", mode='w', newline='\n', encoding='utf-8') as file:
    file.write("recommendation_id,"
               "steam_id,"
               "num_games_owned,"
               "num_reviews,"
               "playtime_forever,"
               "playtime_last_two_weeks,"
               "playtime_at_review,"
               "last_played,"
               "language,"
               "review,"
               "timestamp_created,"
               "timestamp_updated,"
               "voted_up,"
               "votes_up,"
               "votes_funny,"
               "weighted_vote_score,"
               "comment_count,"
               "steam_purchase,"
               "received_for_free,"
               "written_during_early_access\n"
               )
    for review in reviews:
        file.write(f"{review.recommendation_id},"
                   f"{review.steam_id},"
                   f"{review.num_games_owned},"
                   f"{review.num_reviews},"
                   f"{review.playtime_forever},"
                   f"{review.playtime_last_two_weeks},"
                   f"{review.playtime_at_review},"
                   f"{review.last_played},"
                   f"{review.language},"
                   f"\"{review.review.replace("\n", "\\n").replace("\r", "\\r").replace("\"", "")}\","
                   f"{review.timestamp_created},"
                   f"{review.timestamp_updated},"
                   f"{review.voted_up},"
                   f"{review.votes_up},"
                   f"{review.votes_funny},"
                   f"{review.weighted_vote_score},"
                   f"{review.comment_count},"
                   f"{review.steam_purchase},"
                   f"{review.received_for_free},"
                   f"{review.written_during_early_access}\n"
                   )
