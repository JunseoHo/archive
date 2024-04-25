from get_steam_reviews import *


APP_ID_LIST = [568220]
OUT_FILE_NAME = "steam_reviews.csv"

steam_reviews = get_steam_reviews_all(appids=APP_ID_LIST,
                                      filter="recent",
                                      language="english",
                                      review_type="positive",
                                      num_per_page=100)

out_steam_reviews(OUT_FILE_NAME, steam_reviews)






