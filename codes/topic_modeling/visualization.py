import pyLDAvis
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


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



def create_review_topic_matrix_stars(df, lda_model, texts_bow, n_topics, min_prob=0.0):

    # Getting unique ratings in df
    unique_reviews = sorted(df['rating'].unique())
    
    matrix = np.zeros((len(unique_reviews), n_topics))

    # From each document, based on LDA model, we get a prop distribution of topics. 
    topics_per_document = [lda_model.get_document_topics(bow, minimum_probability=min_prob) for bow in texts_bow]

    for i, review in enumerate(unique_reviews):
        indices = df[df['rating'] == review].index.tolist()
        for idx in indices:
            if idx < len(topics_per_document):  
                topic_probs = topics_per_document[idx]
                for topic_id, prob in topic_probs:
                    # in every cell of matrix, there is a sum of prop
                    matrix[i, topic_id] += prob

        # Normalization of rows
        row_sum = matrix[i, :].sum()
        if row_sum > 0:
            matrix[i, :] /= row_sum

    # Normalization by columns
    # for i in range (n_topics):
    #     col_sum = matrix[:, i].sum()
    #     if col_sum > 0:
    #         matrix[:, i] /= col_sum

    matrix = pd.DataFrame(matrix, index=unique_reviews, columns=[f"Topic_{i}" for i in range(n_topics)])

    plt.figure(figsize=(20, 6))
    sns.heatmap(matrix, annot=True, fmt=".5f", cmap="coolwarm", cbar=True)
    plt.title("Review-Topic Matrix Heatmap")
    plt.xlabel("Topics")
    plt.ylabel("Review Scores")
    plt.show()

    return matrix


def create_review_topic_matrix_sentiment(df_with_sentiment, lda_model, texts_bow, n_topics, min_prob=0.0):
    matrix = np.zeros((2, n_topics))

    topics_per_document = [lda_model.get_document_topics(bow, minimum_probability=min_prob) for bow in texts_bow]
    sentiments = ['negative', 'positive']

    for i, sentiment in enumerate(sentiments):
        indices = df_with_sentiment[df_with_sentiment['sentiment'] == sentiment].index.tolist()

        for idx in indices:
            if idx < len(topics_per_document):  
                topic_probs = topics_per_document[idx]
                for topic_id, prob in topic_probs:
                    # in every cell of matrix, there is a sum of prop
                    matrix[i, topic_id] += prob

        # # Normalization of rows
        row_sum = matrix[i, :].sum()
        if row_sum > 0:
            matrix[i, :] /= row_sum

    # Normalization by columns
    # for i in range (n_topics):
    #     col_sum = matrix[:, i].sum()
    #     if col_sum > 0:
    #         matrix[:, i] /= col_sum

    matrix = pd.DataFrame(matrix, index=sentiments, columns=[f"Topic_{i}" for i in range(n_topics)])

    plt.figure(figsize=(20, 6))
    sns.heatmap(matrix, annot=True, fmt=".5f", cmap="coolwarm", cbar=True)
    plt.title("Review-Topic Matrix Heatmap")
    plt.xlabel("Topics")
    plt.ylabel("Review Scores")
    plt.show()

    return matrix

def create_review_topic_matrix_sentiment_for_app(df_with_sentiment, lda_model, texts_bow, n_topics, min_prob=0.0):
    matrix = np.zeros((2, n_topics))

    topics_per_document = [lda_model.get_document_topics(bow, minimum_probability=min_prob) for bow in texts_bow]
    sentiments = ['negative', 'positive']

    for i, sentiment in enumerate(sentiments):
        indices = df_with_sentiment[df_with_sentiment['sentiment'] == sentiment].index.tolist()

        for idx in indices:
            if idx < len(topics_per_document):  
                topic_probs = topics_per_document[idx]
                for topic_id, prob in topic_probs:
                    # in every cell of matrix, there is a sum of prop
                    matrix[i, topic_id] += prob

        # # Normalization of rows
        row_sum = matrix[i, :].sum()
        if row_sum > 0:
            matrix[i, :] /= row_sum

    # Normalization by columns
    # for i in range (n_topics):
    #     col_sum = matrix[:, i].sum()
    #     if col_sum > 0:
    #         matrix[:, i] /= col_sum

    matrix = pd.DataFrame(matrix, index=sentiments, columns=[f"Topic_{i}" for i in range(n_topics)])

    plt.figure(figsize=(20, 6))
    sns.heatmap(matrix, annot=True, fmt=".5f", cmap="coolwarm", cbar=True)
    plt.title("Review-Topic Matrix Heatmap")
    plt.xlabel("Topics")
    plt.ylabel("Review Scores")
    plt.show()

    st.pyplot(plt)
    plt.close()

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


def create_review_topic_matrix_stars_for_app(df, lda_model, texts_bow, n_topics, min_prob=0.0):
    """
    Creates and displays a review-topic matrix as a heatmap.

    Parameters:
        df (DataFrame): DataFrame containing review data.
        lda_model: LDA model object.
        texts_bow: Bag-of-words representation of the texts.
        n_topics (int): Number of topics.
        min_prob (float): Minimum probability threshold for topics.

    Returns:
        matrix (DataFrame): The review-topic matrix.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import streamlit as st

    # Getting unique ratings in df
    unique_reviews = sorted(df['rating'].unique())
    matrix = np.zeros((len(unique_reviews), n_topics))

    # From each document, based on LDA model, we get a probability distribution of topics.
    topics_per_document = [lda_model.get_document_topics(bow, minimum_probability=min_prob) for bow in texts_bow]

    for i, review in enumerate(unique_reviews):
        indices = df[df['rating'] == review].index.tolist()
        for idx in indices:
            if idx < len(topics_per_document):
                topic_probs = topics_per_document[idx]
                for topic_id, prob in topic_probs:
                    # Add probabilities to the matrix
                    matrix[i, topic_id] += prob

        # Normalize rows
        row_sum = matrix[i, :].sum()
        if row_sum > 0:
            matrix[i, :] /= row_sum

    # Convert matrix to DataFrame
    matrix = pd.DataFrame(matrix, index=unique_reviews, columns=[f"Topic_{i}" for i in range(n_topics)])

    # Generate heatmap
    plt.figure(figsize=(20, 6))
    sns.heatmap(matrix, annot=True, fmt=".5f", cmap="coolwarm", cbar=True)
    plt.title("Review-Topic Matrix Heatmap")
    plt.xlabel("Topics")
    plt.ylabel("Ratings")

    # Use Streamlit to display the heatmap
    st.pyplot(plt)
    plt.close()

    return matrix