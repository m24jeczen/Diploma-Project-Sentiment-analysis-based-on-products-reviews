import os
import pandas as pd
from codes.loading_and_filtering.parameters import target_directory, categories
from codes.loading_and_filtering.prepare_category import download_and_save_csv
    
# Functions to load datasets
def load_store_data(category):
    review_path, meta_path = os.path.join(target_directory,category+".jsonl"), os.path.join(target_directory,"meta_"+category+".jsonl")
    if not os.path.exists(review_path) or not os.path.exists(meta_path):
        print(f"Data {category} is not available locally. Downloading neccesary files.")
        download_and_save_csv(category)

    store_file= os.path.join(target_directory,"store_"+category+".txt")
    if not os.path.exists(store_file):
        print(f"Store_data for {category} is not available. Creating neccesary files.")
        download_and_save_csv(category)

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
        print(f"Store_data for {category} is not available. Creating neccesary files.")
        download_and_save_csv(category)
    
    return pd.read_csv(review_path)

def load_products(category):
    if category not in categories:
        print(f"Category {category} does not exist in available categories.")
        return 
    product_path = os.path.join(target_directory,f"product_{category}.csv")

    if  not os.path.exists(product_path):
        print(f"Store_data for {category} is not available. Creating neccesary files.")
        download_and_save_csv(category)
    
    return pd.read_csv(product_path)
 