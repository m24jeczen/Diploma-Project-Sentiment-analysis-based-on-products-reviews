import requests
import gzip
import os
import pandas as pd
from io import BytesIO
import json
from codes.parameters import target_directory, categories
from collections import defaultdict

def download_and_save_csv(category):
    if category not in categories:
        print(f"Category {category} does not exist in available categories.")
        return 
    # Checking if folder for data esists
    os.makedirs(target_directory, exist_ok=True)
    # Preaparing url and downloading results
    reviews_url = f"https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/{category}.jsonl.gz"
    print(f"Downloading {category} and preparing local files")
    reviews_response = requests.get(reviews_url)
    reviews_response.raise_for_status()  
    review_path = os.path.join(target_directory,category + ".csv")
    selected_columns = ["rating","text", "parent_asin","timestamp"]
    review_data = []
    # Opening results and saving in csv file
    with gzip.open(BytesIO(reviews_response.content), "rt", encoding="utf-8") as gz_file:
        for line in gz_file:
            review = json.loads(line)
            values = [review.get(column) for column in selected_columns]
            #Fix rating into int
            values[0] = int(values[0])
            review_data.append(values)

    # Creating data frame in order to aggregate data and save them in csv file
    review_data = pd.DataFrame(review_data, columns=selected_columns)
    review_data['timestamp'] = pd.to_datetime(review_data['timestamp'], unit='ms')
    review_data.drop_duplicates(inplace=True)
    review_data.to_csv(review_path, index=False)
    aggregated = review_data.groupby('parent_asin').agg(
        rating_number=('rating', 'size'),  
        average_rating=('rating', 'mean')  
    ).reset_index()

    # Preaparing url and downloading results for meta data
    meta_url = f"https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_{category}.jsonl.gz"
    meta_response = requests.get(meta_url)
    meta_response.raise_for_status() 
    product_data = []
    selected_columns = ["store","title", "parent_asin"]
    store_data = defaultdict(set)
    with gzip.open(BytesIO(meta_response.content), "rt", encoding="utf-8") as gz_file:
        for line in gz_file:
            if line.strip():  
                try:
                    record = json.loads(line)
                    selected_params = {field: record.get(field, "") for field in selected_columns}
                    product_data.append(selected_params)
                    store_data[selected_params["store"]].add(selected_params["parent_asin"])
                except json.JSONDecodeError as e:
                    print(f"Problem with decoding meta line: {line}")
    product_data = pd.DataFrame(product_data, columns=selected_columns)

    # Saving store data
    store_file = os.path.join(target_directory,f"store_{category}.txt")
    with open(store_file, mode='w', encoding='utf-8') as outfile:
        for key, values in store_data.items():
            outfile.write(f"{key}: {', '.join(values)}\n")

    # Saving product data with aggregated reviews data
    product_data = product_data.merge(aggregated, left_on='parent_asin', right_on='parent_asin')
    product_file = os.path.join(target_directory,f"product_{category}.csv")
    product_data.to_csv(product_file, index=False)