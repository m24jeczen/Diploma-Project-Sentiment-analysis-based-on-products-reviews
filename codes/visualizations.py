import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st  
from io import BytesIO
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


################ Visualizations for LDA
import pyLDAvis


def display_visuals_LDA(model, texts_bow, dictionary):
    try:
        if model is None or not texts_bow or dictionary is None:
            raise ValueError("Invalid inputs provided to display_visuals_LDA. Check the model, texts_bow, and dictionary.")

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

    unique_reviews = sorted(df['rating'].unique())
    matrix = np.zeros((len(unique_reviews), n_topics))

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


def create_review_topic_matrix_sentiment(df_with_sentiment, lda_model, texts_bow, n_topics, min_prob=0.2):
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

def create_review_topic_matrix_sentiment_for_app(df_with_sentiment, lda_model, texts_bow, n_topics, min_prob=0.2):
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


def create_review_topic_matrix_stars_for_app(df, lda_model, texts_bow, n_topics, min_prob=0.2):
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import streamlit as st

    unique_reviews = sorted(df['rating'].unique())
    matrix = np.zeros((len(unique_reviews), n_topics))

    topics_per_document = [lda_model.get_document_topics(bow, minimum_probability=min_prob) for bow in texts_bow]

    for i, review in enumerate(unique_reviews):
        indices = df[df['rating'] == review].index.tolist()
        for idx in indices:
            if idx < len(topics_per_document):
                topic_probs = topics_per_document[idx]
                for topic_id, prob in topic_probs:
                    matrix[i, topic_id] += prob

        # Normalize rows
        row_sum = matrix[i, :].sum()
        if row_sum > 0:
            matrix[i, :] /= row_sum

    matrix = pd.DataFrame(matrix, index=unique_reviews, columns=[f"Topic_{i}" for i in range(n_topics)])

    plt.figure(figsize=(20, 6))
    sns.heatmap(matrix, annot=True, fmt=".5f", cmap="coolwarm", cbar=True)
    plt.title("Review-Topic Matrix Heatmap")
    plt.xlabel("Topics")
    plt.ylabel("Ratings")

    img_stream = BytesIO()
    plt.savefig(img_stream, format="png", bbox_inches="tight")
    img_stream.seek(0)  
    plt.close()

    return img_stream

def create_review_topic_matrix_stars_for_app_new_2(df, lda_model, texts_bow, n_topics, min_prob=0.1):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from io import BytesIO
    from matplotlib.colors import LinearSegmentedColormap

    # App theme colors
    primary_color = "#F6B17A"
    secondary_background_color = "#424769"
    text_color = "#E2E8F0"
    background_color = "#1C2138"

    # Custom colormap: Shades from secondary background to primary color
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_theme", [secondary_background_color, primary_color], N=256
    )

    # Extract unique review ratings and initialize the matrix
    unique_reviews = sorted(df['rating'].unique())
    matrix = np.zeros((len(unique_reviews), n_topics))

    # Get the topic probabilities for each document
    topics_per_document = [lda_model.get_document_topics(bow, minimum_probability=min_prob) for bow in texts_bow]

    # Populate the matrix with probabilities
    for i, review in enumerate(unique_reviews):
        indices = df[df['rating'] == review].index.tolist()
        for idx in indices:
            if idx < len(topics_per_document):
                topic_probs = topics_per_document[idx]
                for topic_id, prob in topic_probs:
                    matrix[i, topic_id] += prob

        # Normalize rows
        row_sum = matrix[i, :].sum()
        if row_sum > 0:
            matrix[i, :] /= row_sum

    # Convert to DataFrame
    matrix = pd.DataFrame(matrix, index=unique_reviews, columns=[f"Topic_{i}" for i in range(n_topics)])

    # Plot the heatmap
    plt.figure(figsize=(20, 6))
    ax = sns.heatmap(
        matrix,
        annot=True,
        fmt=".5f",
        cmap=custom_cmap,
        cbar_kws={'shrink': 0.8, 'format': '%.2f'},
        annot_kws={"color": text_color}
    )

    # Customize the color bar font color
    colorbar = ax.collections[0].colorbar
    colorbar.ax.yaxis.set_tick_params(color=text_color)
    plt.setp(colorbar.ax.yaxis.get_majorticklabels(), color=text_color)

    # Customizing the heatmap's appearance to match the app theme
    plt.title("Rating-Topic Matrix Heatmap", color=text_color)
    plt.xlabel("Topics", color=text_color)
    plt.ylabel("Ratings", color=text_color)

    # Set axis label colors
    ax.tick_params(colors=text_color)

    # Change the background color of the figure
    ax.figure.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    # Save the plot to a BytesIO stream
    img_stream = BytesIO()
    plt.savefig(img_stream, format="png", bbox_inches="tight", facecolor=background_color)
    img_stream.seek(0)
    plt.close()

    return img_stream


def create_review_topic_matrix_stars_for_app_new(df, lda_model, texts_bow, n_topics):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from io import BytesIO
    from matplotlib.colors import LinearSegmentedColormap

    # App theme colors
    primary_color = "#F6B17A"
    secondary_background_color = "#424769"
    text_color = "#E2E8F0"
    background_color = "#1C2138"

    # Custom colormap: Shades from secondary background to primary color
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_theme", [secondary_background_color, primary_color], N=256
    )

    # Extract unique review ratings and initialize the matrix
    unique_reviews = sorted(df['rating'].unique())
    matrix = np.zeros((len(unique_reviews), n_topics))

    # Get the topic with the highest probability for each document
    topics_per_document = [lda_model.get_document_topics(bow) for bow in texts_bow]
    max_topic_per_document = [max(topic_probs, key=lambda x: x[1])[0] for topic_probs in topics_per_document]

    # Populate the matrix with counts
    for i, review in enumerate(unique_reviews):
        indices = df[df['rating'] == review].index.tolist()
        for idx in indices:
            if idx < len(max_topic_per_document):
                assigned_topic = max_topic_per_document[idx]
                matrix[i, assigned_topic] += 1

    # Normalize rows
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

    # Convert to DataFrame
    matrix = pd.DataFrame(matrix, index=unique_reviews, columns=[f"Topic_{i}" for i in range(n_topics)])

    # Plot the heatmap
    plt.figure(figsize=(20, 6))
    ax = sns.heatmap(
        matrix,
        annot=True,
        fmt=".5f",
        cmap=custom_cmap,
        cbar_kws={'shrink': 0.8, 'format': '%.2f'},
        annot_kws={"color": text_color}
    )

    # Customize the color bar font color
    colorbar = ax.collections[0].colorbar
    colorbar.ax.yaxis.set_tick_params(color=text_color)
    plt.setp(colorbar.ax.yaxis.get_majorticklabels(), color=text_color)

    # Customizing the heatmap's appearance to match the app theme
    plt.title("Review-Topic Matrix Heatmap", color=text_color)
    plt.xlabel("Topics", color=text_color)
    plt.ylabel("Ratings", color=text_color)

    # Set axis label colors
    ax.tick_params(colors=text_color)

    # Change the background color of the figure
    ax.figure.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    # Save the plot to a BytesIO stream
    img_stream = BytesIO()
    plt.savefig(img_stream, format="png", bbox_inches="tight", facecolor=background_color)
    img_stream.seek(0)
    plt.close()

    return img_stream



################

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
