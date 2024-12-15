import pyLDAvis
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def display_visuals_LDA(model, texts_bow, dictionary):
    try:
        # Check if the inputs are valid
        if model is None or not texts_bow or dictionary is None:
            raise ValueError("Invalid inputs provided to display_visuals_LDA. Check the model, texts_bow, and dictionary.")

        # Prepare the LDA visualization
        LDAvis_prepared = pyLDAvis.gensim.prepare(model, texts_bow, dictionary)
        return LDAvis_prepared
    except ValueError as ve:
        print(f"ValueError in display_visuals_LDA: {ve}")
    except ImportError:
        print("pyLDAvis library is not installed or improperly imported. Please install it using `pip install pyLDAvis`.")
    except Exception as e:
        print(f"An unexpected error occurred in display_visuals_LDA: {e}")
    return None



def create_review_topic_matrix(df, lda_model, texts_bow, n_topics):

    unique_reviews = sorted(df['rating'].unique())
    
    matrix = np.zeros((len(unique_reviews), n_topics))

    topics_per_document = [lda_model.get_document_topics(bow, minimum_probability=0.0) for bow in texts_bow]

    for i, review in enumerate(unique_reviews):
        indices = df[df['rating'] == review].index.tolist()

        for idx in indices:
            if idx < len(topics_per_document):  
                topic_probs = topics_per_document[idx]
                for topic_id, prob in topic_probs:
                    matrix[i, topic_id] += prob

        row_sum = matrix[i, :].sum()
        if row_sum > 0:
            matrix[i, :] /= row_sum

    matrix = pd.DataFrame(matrix, index=unique_reviews, columns=[f"Topic_{i}" for i in range(n_topics)])

    plt.figure(figsize=(20, 6))
    sns.heatmap(matrix, annot=True, fmt=".5f", cmap="coolwarm", cbar=True)
    plt.title("Review-Topic Matrix Heatmap")
    plt.xlabel("Topics")
    plt.ylabel("Review Scores")
    plt.show()

    return matrix




def generate_topic_rating_matrix(df, n_topics=15):
  
    matrix = np.zeros((5, n_topics))  
    
    for rating in range(1, 6): 
        subset = df[df['rating'] == rating]
        
        for topic in range(n_topics):
            topic_assignments = subset[subset['assigned_topic'] == topic]
            if len(topic_assignments) > 0:
                matrix[rating - 1, topic] = topic_assignments.shape[0] / len(subset)  
    
    matrix_df = pd.DataFrame(matrix, columns=[f"Topic_{i}" for i in range(n_topics)], index=[1, 2, 3, 4, 5])

    plt.figure(figsize=(20, 6))
    sns.heatmap(matrix_df, annot=True, fmt=".5f", cmap="coolwarm", cbar=True)
    plt.title("Review-Topic Matrix Heatmap")
    plt.xlabel("Topics")
    plt.ylabel("Review Scores")
    plt.show()

    return matrix_df