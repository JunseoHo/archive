from get_steam_reviews import *

APP_ID_LIST = [1623730, 413150, 1203620, 899770, 379430]
OUT_FILE_NAME = "steam_reviews.csv"

steam_reviews = get_steam_reviews_all(appids=APP_ID_LIST,
                                      filter="all",
                                      day_range=30,
                                      language="english",
                                      review_type="negative",
                                      num_per_page=100)

out_steam_reviews(OUT_FILE_NAME, steam_reviews)
