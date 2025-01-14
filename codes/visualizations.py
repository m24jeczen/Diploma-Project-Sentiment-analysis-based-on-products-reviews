import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st  
from io import BytesIO

import streamlit as st
from io import BytesIO
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer


def create_topic_word_menu(lda_model, n_topics):
    topic_word_menu = {}
    for topic_id in range(n_topics):
        words_probs = lda_model.show_topic(topic_id, topn=20)
        topic_word_menu[topic_id] = [(word, prob) for word, prob in words_probs]
    return topic_word_menu

def display_topic_words(selected_topic, topic_word_menu):
    if selected_topic in topic_word_menu:
        st.write(f"### Top Words for Topic {selected_topic}")
        top_words = pd.DataFrame(topic_word_menu[selected_topic], columns=["Word", "Probability"])
        st.dataframe(top_words, use_container_width=True)
    else:
        st.warning("Selected topic not found.")


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
        width=800,  
        height=400,  
        background_color="#1C2138",  
        colormap="Oranges",  
        contour_color="#F6B17A",  
        contour_width=0,  
        margin=0,  
        max_words=200, 
        prefer_horizontal=1.0,  
        mode="RGB", 
    ).generate(text)
    
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    
    # its done this way so that the streamlit can display it
    img_stream = BytesIO()
    wordcloud.to_image().save(img_stream, format="PNG")
    img_stream.seek(0)  
    
    return img_stream

def create_tfidf_wordcloud(df, rating_column='rating', text_column='text'):
    # Ensure there is data in the dataframe
    if df.empty:
        return {}

    # Collect all text to identify common words across all ratings
    all_text_data = df[text_column].dropna().astype(str).tolist()
    global_tfidf = TfidfVectorizer(stop_words='english', max_features=300)
    global_tfidf_matrix = global_tfidf.fit_transform(all_text_data)
    feature_names = global_tfidf.get_feature_names_out()
    global_word_scores = global_tfidf_matrix.sum(axis=0).A1
    common_words = {word for word, score in zip(feature_names, global_word_scores) if score > 0}

    unique_ratings = sorted(df[rating_column].unique())
    word_clouds = {}

    for rating in unique_ratings:
        rating_df = df[df[rating_column] == rating]
        text_data = rating_df[text_column].dropna().astype(str).tolist()
        
        # Skip if no valid data
        if not text_data:
            continue

        # Create a new vectorizer for each rating to ensure consistency
        tfidf = TfidfVectorizer(stop_words='english', max_features=300)
        rating_tfidf_matrix = tfidf.fit_transform(text_data)
        feature_names = tfidf.get_feature_names_out()
        scores = rating_tfidf_matrix.sum(axis=0).A1

        # Remove common words across all ratings
        word_scores = {word: scores[idx] for idx, word in enumerate(feature_names) if word not in common_words and scores[idx] > 0}

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,  
            height=400,  
            background_color="#1C2138",  
            colormap="Oranges",  
            contour_color="#F6B17A",  
            max_words=200,  
        ).generate_from_frequencies(word_scores)

        # Save the word cloud to a bytes stream
        img_stream = BytesIO()
        wordcloud.to_image().save(img_stream, format="PNG")
        img_stream.seek(0)

        word_clouds[rating] = img_stream

    return word_clouds




def display_top_words_for_topics(lda_model, n_topics, n_words=20):
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
    try:
        if lda_model is None or n_topics <= 0:
            raise ValueError("Invalid LDA model or number of topics provided.")

        topics_data = []
        
        for topic_id in range(n_topics):
            try:
                top_words = lda_model.show_topic(topic_id, topn=n_words)
                words = [word for word, _ in top_words]
                topics_data.append([f"Topic {topic_id}", ', '.join(words)])
            except Exception as e:
                topics_data.append([f"Topic {topic_id}", f"Error retrieving words ({e})"])

        df_topics = pd.DataFrame(topics_data, columns=["Topic", "Words"])

        return df_topics
    except Exception as e:
        return f"An error occurred: {e}"
