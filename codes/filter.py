import pandas as pd
from datetime import datetime
import json


def filter_by_date_range(df):
    if 'timestamp' not in df.columns:
        print("Error: Column 'timestamp' does not exist in the DataFrame.")
        return None

    df['timestamp_date'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    earliest_date = df['timestamp_date'].min()
    latest_date = df['timestamp_date'].max()
    
    print(f"Earliest review date: {earliest_date.strftime('%d-%m-%Y')}")
    print(f"Latest review date: {latest_date.strftime('%d-%m-%Y')}")
    
    date_range = input("Enter the date range in format 'dd-mm-yyyy, dd-mm-yyyy': ").strip()
    try:
        start_date_str, end_date_str = date_range.split(',')
        start_date = datetime.strptime(start_date_str.strip(), '%d-%m-%Y')
        end_date = datetime.strptime(end_date_str.strip(), '%d-%m-%Y')
    except ValueError:
        print("Error: Invalid date format. Please use 'dd-mm-yyyy, dd-mm-yyyy'.")
        return None
    
    filtered_df = df[(df['timestamp_date'] >= start_date) & (df['timestamp_date'] <= end_date)]
    
    print(f"Filtered {len(filtered_df)} records between {start_date.strftime('%d-%m-%Y')} and {end_date.strftime('%d-%m-%Y')}.")
    return filtered_df

def filter_by_text_length(df):
    min_length = int(input("Enter the minimum length of text for the column: ").strip())
    text = 'text'
    if text not in df.columns:
        print(f"Error: Column '{text}' does not exist in the DataFrame.")
        return None
    
    filtered_df = df[df[text].str.len() >= min_length]
    
    print(f"Filtered {len(filtered_df)} records with {text} length >= {min_length}.")
    return filtered_df

def filter_large_jsonl_from_paths(meta_path, review_path, product_names):
    product_ids = set()
    with open(meta_path, 'r', encoding='utf-8') as meta_file:
        for line in meta_file:
            if line.strip():  
                try:
                    record = json.loads(line)
                    if 'title' in record and any(name.strip().lower() in record['title'].lower() for name in product_names):
                        product_ids.add(record['parent_asin'])
                except json.JSONDecodeError as e:
                    print(f"Problem with decoding meta line: {line}")
    
    filtered_reviews = []
    with open(review_path, 'r', encoding='utf-8') as review_file:
        for line in review_file:
            if line.strip():  
                try:
                    record = json.loads(line)
                    if record['parent_asin'] in product_ids:
                        filtered_reviews.append(record)
                except json.JSONDecodeError as e:
                    print(f"Problem with decoding review line: {line}")
    
    filtered_meta = []
    with open(meta_path, 'r', encoding='utf-8') as meta_file:
        for line in meta_file:
            if line.strip():  
                try:
                    record = json.loads(line)
                    if record['parent_asin'] in product_ids:
                        filtered_meta.append(record)
                except json.JSONDecodeError as e:
                    print(f"Problem with decoding meta line: {line}")
    
    meta_df = pd.DataFrame(filtered_meta)
    review_df = pd.DataFrame(filtered_reviews)
    
    result_df = pd.merge(meta_df, review_df, left_on='parent_asin', right_on='parent_asin', how='left')
    
    return result_df
