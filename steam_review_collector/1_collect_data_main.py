from lib.get_steam_reviews import get_steam_reviews_all, out_steam_reviews

"""
    Steam indie game ranking : https://steamdb.info/charts/?tagid=492
"""

APP_ID_LIST = ['413150', '1145360', '1794680', '420530', '432350', '760890', '646570', '1150690', '1240210', '1275670', '246420', '799640', '1082710', '1123450', '1783360', '15373', '206440', '512900', '874260', '816340', '1535560', '559210', '1958220', '885810', '2084000', '1487270', '926340', '1534980', '1056490', '233860', '881100', '1562430', '35720', '77828', '250760', '335670', '1229380', '1488200', '259680', '2113430', '1293860', '1333200', '1769170', '1562920', '349760', '1617220', '2299150', '2106810', '1321120', '1687550']

OUT_FILE_NAME = "steam_reviews.csv"

steam_reviews = get_steam_reviews_all(appids=APP_ID_LIST,
                                      filter="all",
                                      day_range=365,
                                      language="english",
                                      review_type="negative",
                                      num_per_page=100,
                                      max_count=100)

out_steam_reviews(OUT_FILE_NAME, steam_reviews)

"""
    positive : charact, stori, fun, great, experi, best, recommend
    negative : charact, enemi, stori, boss, run, fight, level, item attack, puzzl, die, difficulti
"""