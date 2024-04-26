from lib.get_steam_reviews import get_steam_reviews_all, out_steam_reviews

APP_ID_LIST = [1623730, 413150, 1203620, 899770, 379430]
OUT_FILE_NAME = "steam_reviews.csv"

steam_reviews = get_steam_reviews_all(appids=APP_ID_LIST,
                                      filter="all",
                                      day_range=30,
                                      language="korean",
                                      review_type="positive",
                                      num_per_page=100,
                                      max_count=500)

out_steam_reviews(OUT_FILE_NAME, steam_reviews)
