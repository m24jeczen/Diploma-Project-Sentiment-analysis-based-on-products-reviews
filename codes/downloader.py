import requests
import gzip
import os
import pandas as pd
from io import BytesIO
import json
from codes.parameters import target_directory, categories
from collections import defaultdict
import csv

 
def download_source_data(category):
     # Checking if input is available category
    if category not in categories:
        print(f"Category {category} does not exist in available categories.")
        return 
    # Checking if folder for data esists
    os.makedirs(target_directory, exist_ok=True)

    # Downloading and saving original review data
    reviews_url = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/"
    reviews_extracted_filename = category + ".jsonl"
    reviews_jsonl_file_path = os.path.join(target_directory, reviews_extracted_filename)
 
    reviews_url = reviews_url +reviews_extracted_filename+ ".gz"
    print(f"Downloading {category}")
    reviews_response = requests.get(reviews_url)
    reviews_response.raise_for_status()  
    
    with gzip.open(BytesIO(reviews_response.content), "rt", encoding="utf-8") as gz_file:
        with open(reviews_jsonl_file_path, "w", encoding="utf-8") as jsonl_file:
            jsonl_file.write(gz_file.read())
 
    print(f"File saved and loaded as {reviews_extracted_filename}")

    # Downloading and saving original meta data
    meta_url = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/"

    meta_extracted_filename = "meta_" + category + ".jsonl"
    meta_jsonl_file_path = os.path.join(target_directory, meta_extracted_filename)
 
    meta_url = meta_url +meta_extracted_filename+ ".gz"
    meta_response = requests.get(meta_url)
    meta_response.raise_for_status() 
   
    with gzip.open(BytesIO(meta_response.content), "rt", encoding="utf-8") as gz_file:
        with open(meta_jsonl_file_path, "w", encoding="utf-8") as jsonl_file:
            jsonl_file.write(gz_file.read())
 
    print(f"File saved and loaded as {meta_extracted_filename}")


# Function to save reviews locally in csv and return average rating and number of rating per product
def create_aggregated_data_and_save_reviews_data(category):
    output_file = os.path.join(target_directory,category + ".csv")
    review_path= os.path.join(target_directory,category+".jsonl")
    selected_columns = ["rating","text", "parent_asin","timestamp"]
    data = []
    with open(review_path, 'r', encoding='utf-8') as review_file:
        for line in review_file:
            if line.strip():  
                try:
                    record = json.loads(line)
                    selected_params = {field: record.get(field, "") for field in selected_columns}
                    data.append(selected_params)
                except json.JSONDecodeError as e:
                    print(f"Problem with decoding meta line: {line}")

    data = pd.DataFrame(data, columns=selected_columns)
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.drop_duplicates(inplace=True)
    data.to_csv(output_file, index=False)
    aggregated = data.groupby('parent_asin').agg(
        rating_number=('rating', 'size'),  
        average_rating=('rating', 'mean')  
    ).reset_index()

    return aggregated

# Function which create local data of stores and return data of products useful in bussines analysis of data
def create_store_product_data(category):
    meta_path = os.path.join(target_directory,"meta_"+category+".jsonl")
    selected_columns = ["parent_asin", "title", "store"]
    product_data = []
    store_data = defaultdict(set)
    with open(meta_path, 'r', encoding='utf-8') as meta_file:
        for line in meta_file:
            if line.strip():  
                try:
                    record = json.loads(line)
                    selected_params = {field: record.get(field, "") for field in selected_columns}
                    product_data.append(selected_params)
                    store_data[selected_params["store"]].add(selected_params["parent_asin"])
                except json.JSONDecodeError as e:
                    print(f"Problem with decoding meta line: {line}")

    product_data = pd.DataFrame(product_data, columns=selected_columns)
    store_file = os.path.join(target_directory,"store_"+category+".txt")
    with open(store_file, mode='w', encoding='utf-8') as outfile:
        for key, values in store_data.items():
            outfile.write(f"{key}: {', '.join(values)}\n")

    return product_data


# Create 3 local datasets     
def create_local_data(category):
    if category not in categories:
        print(f"Category {category} does not exist in available categories.")
        return 
    
    review_path, meta_path = os.path.join(target_directory,category+".jsonl"), os.path.join(target_directory,"meta_"+category+".jsonl")

    if not os.path.exists(review_path) or not os.path.exists(meta_path):
        print(f"Data {category}  is not available locally. Downloading neccesary files.")
        download_source_data(category)

    aggregated = create_aggregated_data_and_save_reviews_data(category)
    product_data = create_store_product_data(category)
    product_data = product_data.merge(aggregated, left_on='parent_asin', right_on='parent_asin')
    product_file = os.path.join(target_directory,f"product_{category}.csv")
    product_data.to_csv(product_file, index=False)

    
# Functions to load datasets
def load_store_data(category):
    review_path, meta_path = os.path.join(target_directory,category+".jsonl"), os.path.join(target_directory,"meta_"+category+".jsonl")
    if not os.path.exists(review_path) or not os.path.exists(meta_path):
        print(f"Data {category}  is not available locally. Downloading neccesary files.")
        download_source_data(category)

    store_file= os.path.join(target_directory,"store_"+category+".txt")
    if not os.path.exists(store_file):
        print(f"Store_data for {category}  is not available. Creating neccesary files.")
        create_local_data(category)

    store_data={}
    with open(store_file, mode='r', encoding='utf-8') as infile:
        for line in infile:
            # Deleting few stores with invalid names, very small sample lost
            if line.count(":") == 1:
                key, values = line.strip().split(":")
                store_data[key.strip()] = set(values.strip().split(", "))
               
    return store_data

def load_reviews(category):
    if category not in categories:
        print(f"Category {category} does not exist in available categories.")
        return 
    review_path = os.path.join(target_directory,f"{category}.csv")

    if  not os.path.exists(review_path):
        print(f"Store_data for {category}  is not available. Creating neccesary files.")
        create_local_data(category)
    
    return pd.read_csv(review_path)

def load_products(category):
    if category not in categories:
        print(f"Category {category} does not exist in available categories.")
        return 
    product_path = os.path.join(target_directory,f"product_{category}.csv")

    if  not os.path.exists(product_path):
        print(f"Store_data for {category}  is not available. Creating neccesary files.")
        create_local_data(category)
    
    return pd.read_csv(product_path)
 