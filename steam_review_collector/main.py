from get_steam_reviews import *
import pandas as pd

APP_ID_LIST = [985810, 568220]
OUT_FILE_NAME = "steam_reviews.csv"

steam_reviews = get_steam_reviews_all(appids=APP_ID_LIST,
                                      filter="recent",
                                      language="english",
                                      review_type="negative",
                                      num_per_page=100)

out_steam_reviews(OUT_FILE_NAME, steam_reviews)

df = pd.read_csv(OUT_FILE_NAME)

print(df)
