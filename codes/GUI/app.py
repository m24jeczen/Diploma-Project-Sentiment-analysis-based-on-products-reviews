import streamlit as st
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Adding root path to the project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
os.environ["STREAMLIT_CONFIG_FILE"] = os.path.abspath(
    os.path.join(project_root, ".streamlit/config.toml")
)

from codes.loading_and_filtering.parameters import categories
from codes.loading_and_filtering.filter import filter
from codes.loading_and_filtering.data_loader import load_products, load_reviews
from codes.topic_modeling.text_preprocessing import preprocess_text
from codes.topic_modeling.LDA import LDA_training
from codes.topic_modeling.visualization import *
from codes.visualizations import *

# Set page configuration
st.set_page_config(page_title="Amazon Products", layout="wide")

# Inside styles css is the custom colors for the app
css_file_path = os.path.join(project_root, "static", "styles.css")
with open(css_file_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <h1 style="color:#F6B17A; text-align: center; font-size: 35px;">AMAZON PRODUCTS ANALYSIS</h1>
    """,
    unsafe_allow_html=True
)

# The first page is Menu
if "page" not in st.session_state:
    st.session_state.page = "Menu"

# Navigation bar for switching between pages (after filtering)
def render_nav_bar():
    return st.radio(
        "",
        ["NLP Results", "Model xyz Results"],
        horizontal=True,
        key="navigation"
    )

# Menu Page (Main Entry Point)
if st.session_state.page == "Menu":
    st.write("### Main Menu")
    
    if st.button("Go to Filter Products"):
        st.session_state.page = "Filter Products"
        st.rerun()

    if st.button("Go to ratings and words analysis"):
        st.session_state.page = "Ratings and words analysis"
        st.rerun()

    if st.button("Go to Model xyz Results"):
        st.session_state.page = "Model xyz Results"
        st.rerun()

# Page 1: Filter Products
elif st.session_state.page == "Filter Products":
    st.markdown(
        """
        <h3 style="text-align: center;">Set Filters</h3>
        """,
        unsafe_allow_html=True
    )

    selected_category = st.selectbox("Choose category:", categories, index=categories.index("CDs_and_Vinyl"))
    min_text_length = st.number_input("Minimal number of words in review", min_value=1, value=20)
    start_date = st.date_input("Start date:", value=pd.to_datetime("2018-01-01"))
    end_date = st.date_input("End date:", value=pd.to_datetime("2018-03-01"))
    min_reviews_per_product = st.number_input("Minimal number of reviews per product", min_value=1, value=10)
    min_average_rating = st.slider("Minimal average rating of product:", 1.0, 5.0, 1.0)
    store = st.text_input("Filter store by name (optional)")
    search_value = st.text_input("Filter products by name (optional)")

    
    if st.button("Apply Filters"):
        try:
            def load_data(category):
                with st.spinner("Loading data..."):
                    product_data = load_products(category)
                    review_data = load_reviews(category)
                return product_data, review_data

            product_data, review_data = load_data(selected_category)

            if product_data is None or review_data is None:
                st.error("Failed to load data. Please check the category or try again.")
            else:
                with st.spinner("Applying filters..."):
                    filtered_reviews = filter(
                        category=selected_category,
                        min_text_length=min_text_length,
                        start_date=start_date,
                        end_date=end_date,
                        min_reviews_per_product=min_reviews_per_product,
                        min_average_rating=min_average_rating,
                        #store=store if store else None,  
                        #search_value=search_value if search_value else None 
                    )

                if filtered_reviews.empty:
                    st.write("No data matches these filters.")
                else:
                    st.success("Filtering successful! Redirecting to the menu...")

                    st.session_state.filtered_reviews = filtered_reviews
                    st.session_state.page = "Menu"  # Redirect to the menu page
                    st.rerun()  

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Page 2: Ratings and words analysis
elif st.session_state.page == "Ratings and words analysis":
    st.write("### Ratings and words analysis")
    
    if "filtered_reviews" in st.session_state:
        filtered_reviews = st.session_state.filtered_reviews
        
        st.write("#### Word Clouds by Star Rating")
        ratings = [1, 2, 3, 4, 5]
        
        if "word_clouds_by_rating" not in st.session_state:
            st.session_state.word_clouds_by_rating = {}

        try:
            # Creating columns for displaying word clouds in the same row
            cols = st.columns(5)
            
            for i, rating in enumerate(ratings):
                with cols[i]:
                    if rating in st.session_state.word_clouds_by_rating:
                        word_cloud_image = st.session_state.word_clouds_by_rating[rating]
                    else:
                        rating_df = filtered_reviews[filtered_reviews['rating'] == rating]
                        word_cloud_image = create_wordcloud_from_df_for_app(rating_df)
                        st.session_state.word_clouds_by_rating[rating] = word_cloud_image
                    
                    st.image(word_cloud_image, caption=f"{rating}-Star Reviews")
        except Exception as e:
            st.error(f"An error occurred while generating word clouds: {e}")
        
        

        # LDA Parameters and Analysis
        st.write("#### LDA Analysis Parameters")
        with st.form("lda_parameters_form"):
            n_topics = st.number_input("Number of Topics", min_value=1, max_value=20, value=9, step=1)
            chunksize = st.number_input("Chunksize", min_value=10, max_value=5000, value=1000, step=10)
            passes = st.number_input("Number of Passes", min_value=1, max_value=100, value=10, step=1)
            iterations = st.number_input("Iterations", min_value=10, max_value=500, value=200, step=10)
            update_every = st.selectbox("Update Every", options=[1, 0], index=0)
            eval_every = st.number_input("Evaluate Every", min_value=1, max_value=500, value=100, step=10)

            submit_button = st.form_submit_button("Perform LDA Analysis")

        if "lda_model" not in st.session_state:
            st.session_state.lda_model = None
            st.session_state.topic_word_menu = None
            st.session_state.review_topic_matrix = None
            st.session_state.selected_topic_id = None

        if submit_button:
            try:
                if "texts_bow" in st.session_state and "dictionary" in st.session_state and "id2word" in st.session_state:
                    texts_bow = st.session_state.texts_bow
                    dictionary = st.session_state.dictionary
                    id2word = st.session_state.id2word
                else:
                    texts_bow, dictionary, id2word = preprocess_text(filtered_reviews, 100, 0.85)
                    st.session_state.texts_bow = texts_bow
                    st.session_state.dictionary = dictionary
                    st.session_state.id2word = id2word

                lda_model = LDA_training(
                    filtered_reviews, texts_bow, dictionary, id2word,
                    n_topics=n_topics, chunksize=chunksize, passes=passes,
                    iterations=iterations, update_every=update_every, eval_every=eval_every
                )

                st.session_state.lda_model = lda_model
                st.session_state.n_topics = n_topics
                st.session_state.topic_word_menu = create_topic_word_menu(lda_model, n_topics)

                review_topic_image = create_review_topic_matrix_stars_for_app(
                    filtered_reviews, lda_model, texts_bow, n_topics, 0.3
                )
                st.session_state.review_topic_matrix = review_topic_image

                st.success("LDA analysis complete. Results saved.")
            except Exception as e:
                st.error(f"Error during LDA analysis: {e}")

        if st.session_state.topic_word_menu is not None and st.session_state.review_topic_matrix is not None:
            cols = st.columns([1, 2, 5])  

            with cols[0]:
                st.write("### Topics Menu")
                for topic_id in range(st.session_state.n_topics):
                    if st.button(f"Topic {topic_id}", key=f"topic_{topic_id}"):
                        st.session_state.selected_topic_id = topic_id 
                    # else:
                    #     st.session_state.selected_topic_id = 0

            with cols[1]:
                if 'selected_topic_id' in st.session_state:
                    display_topic_words(st.session_state.selected_topic_id, st.session_state.topic_word_menu)

            with cols[2]:
                st.image(st.session_state.review_topic_matrix, caption="Review-Topic Matrix Heatmap")

    else:
        st.warning("No filtered data available. Please filter data on the home page first.")

    if st.button("Back to Menu"):
        st.session_state.page = "Menu"
        st.rerun()


# Page 3: Model xyz Results
elif st.session_state.page == "Model xyz Results":
    st.write("### Model xyz Results")
    
    if "filtered_reviews" in st.session_state:
        try:
            # Examples:
            st.metric("Model Accuracy", "92.5%")
            st.metric("F1 Score", "89.7%")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("No filtered data or model results available. Please go to 'Filter Products' page first and apply filters.")

    if st.button("Back to Menu"):
        st.session_state.page = "Menu"
        st.rerun()
