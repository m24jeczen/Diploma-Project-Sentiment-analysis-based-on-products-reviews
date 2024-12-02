import requests
import gzip
import os
from io import BytesIO
 
def load_and_save_category(category):

    current_directory = os.getcwd()

    # Uzyskanie ścieżki dwa katalogi wyżej
    two_levels_up = os.path.abspath(os.path.join(current_directory, ".."))

    # Ścieżka do folderu, który chcesz utworzyć
    target_directory = os.path.join(two_levels_up, "amazon_data")

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
 