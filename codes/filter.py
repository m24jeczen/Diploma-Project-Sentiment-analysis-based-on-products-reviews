import pandas as pd
from datetime import datetime
import json, os
from codes.parameters import target_directory



def filter(category, min_length, min_reviews_per_product, min_average_rating):
    data = pd.read_csv(os.path.join(target_directory, f"{category}.csv"))
    #data['timestamp_date'] = pd.to_datetime(data['timestamp'], unit='ms')     

    # time filtring
    #data = data[(data['timestamp_date'] >= start_date) & (data['timestamp_date'] <= end_date)]
    # min text length filtring
    data = data[data["text"].str.split().str.len() >= min_length]
    # min reviews per product filter
    data = data[data["rating_number"] >= min_reviews_per_product]
    # min reviews per product filter
    data = data[data["average_rating"] >= min_average_rating]

    return data