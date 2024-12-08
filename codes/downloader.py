import requests
import gzip
import os
import pandas as pd

from io import BytesIO

 
def download_and_save_category(category):

    current_directory = os.getcwd()

    # Uzyskanie ścieżki dwa katalogi wyżej
    upper_catalog = os.path.abspath(os.path.join(current_directory, ".."))

    # Ścieżka do folderu, który chcesz utworzyć
    target_directory = os.path.join(upper_catalog, "amazon_data")

    # Tworzenie folderu, jeśli nie istnieje
    os.makedirs(target_directory, exist_ok=True)
    reviews_url = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/"
    
    # Nadanie nazw plikom
    reviews_extracted_filename = category + ".jsonl"
    reviews_jsonl_file_path = os.path.join(target_directory, reviews_extracted_filename)
 
    reviews_url = reviews_url +reviews_extracted_filename+ ".gz"
    # Pobranie pliku z URL i zapisanie go lokalnie
    reviews_response = requests.get(reviews_url)
    reviews_response.raise_for_status()  # Sprawdź, czy nie wystąpiły błędy podczas pobierania
   
    with gzip.open(BytesIO(reviews_response.content), "rt", encoding="utf-8") as gz_file:
        with open(reviews_jsonl_file_path, "w", encoding="utf-8") as jsonl_file:
            jsonl_file.write(gz_file.read())
 
    print(f"File saved and loaded as {reviews_extracted_filename}")
    
    meta_url = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/"
   
    # Nadanie nazw plikom
    meta_extracted_filename = "meta_" + category + ".jsonl"
    meta_jsonl_file_path = os.path.join(target_directory, meta_extracted_filename)
 
    meta_url = meta_url +meta_extracted_filename+ ".gz"
    # Pobranie pliku z URL i zapisanie go lokalnie
    meta_response = requests.get(meta_url)
    meta_response.raise_for_status()  # Sprawdź, czy nie wystąpiły błędy podczas pobierania
   
    with gzip.open(BytesIO(meta_response.content), "rt", encoding="utf-8") as gz_file:
        with open(meta_jsonl_file_path, "w", encoding="utf-8") as jsonl_file:
            jsonl_file.write(gz_file.read())
 
    print(f"File saved and loaded as {meta_extracted_filename}")
    
def load_products(category, products):
    products = set(products)
    upper_catalog = os.path.abspath(os.path.join(os.getcwd(), ".."))
    meta_filename= "meta_" + category+".jsonl"
    # Ścieżka do folderu, który chcesz utworzyć
    target_directory = os.path.join(upper_catalog, "amazon_data")
    meta_path = os.path.join(target_directory , meta_filename)
    meta_data = pd.read_json(meta_path, lines=True)
    #meta_data = meta_data[meta_data["title"].isin(products)]
    products_ids = set(meta_data["parent_asin"])
    print(products_ids)
    review_filename = category + ".jsonl"
    review_path = os.path.join(target_directory, review_filename)
    review_data = pd.read_json(review_path, lines=True)
    #review_data = review_data[review_data["parent_asin"].isin(products_ids)]
    
    return meta_data, review_data

 