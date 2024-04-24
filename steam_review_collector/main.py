import requests
import datetime


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

    print(url)
    cursor = "*"
    reviews = []
    cursor_history = []
    while (True):
        response = requests.get(url + f"&cursor={cursor}")
        if response.status_code != 200:
            print(f"get steam reviews by api failed: HTTP status is {response.status_code}")
            break
        json = response.json()
        for review in json["reviews"]:
            reviews.append(review["review"])
        if json["cursor"] in cursor_history:
            break
        cursor_history.append(json["cursor"])
    return reviews


reviews = get_steam_reviews_by_api(app_id=262060,
                                   language="korean",
                                   num_per_page=100)

print(reviews)
