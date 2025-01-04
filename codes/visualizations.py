import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st  

def create_wordcloud_from_df(df):
    text = " ".join(df['text'].dropna().astype(str))
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def create_wordcloud_from_df_for_app(df):
    text = " ".join(df['text'].dropna().astype(str))
    
    wordcloud = WordCloud(
        width=800,  # Width of the image
        height=400,  # Height of the image
        background_color="#1C2138",  # Dark background
        colormap="Oranges",  # Color palette matching the primary color
        contour_color="#F6B17A",  # Outline color
        contour_width=0,  # Outline width
        margin=0,  # Remove margins around the word cloud
        max_words=200,  # Limit the number of words (adjust as needed)
        prefer_horizontal=1.0,  # Force words to be horizontal (optional)
        mode="RGB",  # Color mode
    ).generate(text)
    
    plt.figure(figsize=(8, 4))  # Adjusted figure size
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    
    # Ensure no additional space around the image
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Streamlit-specific display
    st.pyplot(plt, bbox_inches='tight', pad_inches=0)

def display_top_words_for_topics(lda_model, n_topics, n_words=20):
    """
    Generate a string of top words for each topic from the LDA model.

    Parameters:
        lda_model: The trained LDA model.
        n_topics (int): Number of topics.
        n_words (int): Number of words to display per topic.

    Returns:
        str: Formatted string of top words for each topic.
    """
    try:
        if lda_model is None or n_topics <= 0:
            raise ValueError("Invalid LDA model or number of topics provided.")

        topics_summary = []
        for topic_id in range(n_topics):
            try:
                top_words = lda_model.show_topic(topic_id, topn=n_words)
                words = [word for word, _ in top_words]
                topics_summary.append(f"**Topic {topic_id}:** {', '.join(words)}")
            except Exception as e:
                topics_summary.append(f"**Topic {topic_id}:** Error retrieving words ({e})")

        return "\n\n".join(topics_summary)
    except Exception as e:
        return f"An error occurred: {e}"

def display_top_words_for_topics2(lda_model, n_topics, n_words=20):
    """
    Generate a DataFrame with the top words for each topic from the LDA model.

    Parameters:
        lda_model: The trained LDA model.
        n_topics (int): Number of topics.
        n_words (int): Number of words to display per topic.

    Returns:
        pd.DataFrame: DataFrame containing topics and corresponding words.
    """
    try:
        if lda_model is None or n_topics <= 0:
            raise ValueError("Invalid LDA model or number of topics provided.")

        # List to store topic ids and words
        topics_data = []
        
        for topic_id in range(n_topics):
            try:
                # Get top words for each topic
                top_words = lda_model.show_topic(topic_id, topn=n_words)
                words = [word for word, _ in top_words]
                topics_data.append([f"Topic {topic_id}", ', '.join(words)])
            except Exception as e:
                topics_data.append([f"Topic {topic_id}", f"Error retrieving words ({e})"])

        # Create a DataFrame
        df_topics = pd.DataFrame(topics_data, columns=["Topic", "Words"])

        # Return the DataFrame
        return df_topics
    except Exception as e:
        return f"An error occurred: {e}"
