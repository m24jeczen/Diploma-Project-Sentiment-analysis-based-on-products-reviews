import pandas as pd
from sklearn.utils import resample

# 
def balance_data(data, label):
    # Find the minimum class size
    min_count = data[label].value_counts().min()

    # Group data by ratings and perform undersampling for larger classes
    balanced_df = pd.concat([
        resample(group, replace=False, n_samples=min_count, random_state=42)
        for _, group in data.groupby(label)
    ])

    # Shuffle the resulting balanced DataFrame
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_df


# This function takes treshold from user and creates new labels 
def map_ratings_into_sentiment(data,positive_threshold = 4):

    data['star_based_sentiment'] = data['rating'].apply(lambda x: 1 if x >= positive_threshold else 0)
    return data
