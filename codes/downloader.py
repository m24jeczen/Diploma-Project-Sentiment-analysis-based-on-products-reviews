import requests
import gzip
import os
import pandas as pd
from io import BytesIO
import json, csv
from codes.parameters import target_directory

 
def download_source_data(category):

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
    
def create_local_data(category):
    # Meta data not used yet maybe will be later
    meta_path, review_path = os.path.join(target_directory,"meta_" + category+".csv"), os.path.join(target_directory,category+".jsonl")
    output_file = os.path.join(target_directory,category + ".csv")
    selected_columns = ["rating","text", "parent_asin","timestamp"]
    data = []
    with open(review_path, 'r', encoding='utf-8') as review_file:
        for line in review_file:
            if line.strip():  
                try:
                    record = json.loads(line)
                    # Get selected parameters and load them to local csv
                    selected_params = {field: record.get(field, "") for field in selected_columns}
                    data.append(selected_params)
                except json.JSONDecodeError as e:
                    print(f"Problem with decoding meta line: {line}")
    data = pd.DataFrame(data, columns=selected_columns)
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.drop_duplicates(inplace=True)
    aggregated = data.groupby('parent_asin').agg(
        rating_number=('rating', 'size'),  # Liczba wierszy dla każdej grupy
        average_rating=('rating', 'mean')  # Średnia wartość rating
    ).reset_index()
    data = data.merge(aggregated, on='parent_asin', how='left')
    data.to_csv(output_file, index=False)


    


 