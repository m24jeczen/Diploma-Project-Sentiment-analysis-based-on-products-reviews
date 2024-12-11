import pandas as pd
from datetime import datetime
import json, os
from codes.parameters import target_directory
from codes.downloader import load_store_data



def filter(category, min_text_length = None, start_date = None, end_date = None, min_reviews_per_product = None, min_average_rating = None, store = None, search_value = None):
    product_data = pd.read_csv(os.path.join(target_directory, f"product_{category}.csv")) 
    review_data = pd.read_csv(os.path.join(target_directory, f"{category}.csv"))  
    
    # Filtering with used parameteres. In later version will be implemented more elegant propably with object filter
    if store is not None:
        # store_data is a dict with key as a store name and value is set of products of the store
        store_data = load_store_data(category)
        if store not in store_data.keys():
            return f"Store {store} doesn't exist."
        
        store_products=store_data[store]
        product_data = product_data[product_data["parent_asin"].isin(store_products)]

    if search_value is not None:
        if not isinstance(search_value, str):
            return f"{search_value} should be a string."
        product_data=product_data[product_data["title"].str.contains(search_value, case=False, na=False)]


    if min_reviews_per_product is not None:
        if not isinstance(min_reviews_per_product, int) and min_reviews_per_product <= 0:
            return f"{min_reviews_per_product} should be positivie integer."
        product_data = product_data[product_data["rating_number"] >= min_reviews_per_product]

    if min_average_rating is not None:
        if not isinstance(min_average_rating, int) and not (1 <= min_average_rating <= 5):
            return f"{min_average_rating} should be anumber beetwen 1 and 5"
        product_data = product_data[product_data["average_rating"] >= min_average_rating] 

    if any(param is not None for param in [min_reviews_per_product, min_average_rating, store, search_value]):
        review_data = review_data[review_data["parent_asin"].isin(product_data['parent_asin'])]

    # time filtring
    if start_date is not None and end_date is not None:
        try:
            start_date= pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            review_data['timestamp'] = pd.to_datetime(review_data['timestamp'])
            review_data = review_data[(review_data['timestamp'] >= start_date) & (review_data['timestamp'] <= end_date)]
        except ValueError:
            return "Wrong date format."
    # min text length filtring
    if min_text_length is not None:
        if not isinstance(min_text_length, int) and min_text_length <= 0:
            return f"{min_text_length} should be positivie integer."
        review_data = review_data[review_data["text"].str.split().str.len() >= min_text_length]

    return review_data.reset_index(drop=True)