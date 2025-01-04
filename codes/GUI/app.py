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

# Your existing app code
st.set_page_config(page_title="Amazon Products", layout="wide")

# Load external CSS
css_file_path = os.path.join(project_root, "static", "styles.css")

# Inject the CSS file into the Streamlit app
with open(css_file_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# App title
st.markdown(
    """
    <h1 style="color:#F6B17A; text-align: center; font-size: 35px;">Amazon Product Analysis</h1>
    """,
    unsafe_allow_html=True
)

# Horizontal navigation bar
st.markdown("---")
nav_option = st.radio(
    "",
    ["Filter Products", "NLP Results", "Model Results"],
    horizontal=True,
    key="navigation"
)
st.markdown("---")

# Page 1: Filter Products
if nav_option == "Filter Products":
    st.markdown(
        """
        <h3 style="text-align: center;">Set Filters</h3>
        """,
        unsafe_allow_html=True
    )
    
    # Check if we have stored filter settings in session_state
    if "selected_category" in st.session_state:
        st.write('Jest')
        selected_category = st.session_state["selected_category"]
    else:
        selected_category = categories[0]

    # Pre-fill the filter values if they are stored
    selected_category = st.selectbox("Choose category:", categories, index=categories.index(selected_category))
    min_text_length = st.number_input("Minimal number of words in review", min_value=1, value=st.session_state.get("min_text_length", 10))
    start_date = st.date_input("Start date:", value=st.session_state.get("start_date", pd.to_datetime("2023-01-01")))
    end_date = st.date_input("End date:", value=st.session_state.get("end_date", pd.to_datetime("2023-12-31")))
    min_reviews_per_product = st.number_input("Minimal number of reviews per product", min_value=1, value=st.session_state.get("min_reviews_per_product", 10))
    min_average_rating = st.slider("Minimal average rating of product:", 1.0, 5.0, st.session_state.get("min_average_rating", 4.0))
    store = st.text_input("Filter store by name (optional)", value=st.session_state.get("store", ""))
    search_value = st.text_input("Filter products by name (optional)", value=st.session_state.get("search_value", ""))


    if st.button("Apply Filters"):
        try:
            # Function to load data
            def load_data(category):
                with st.spinner("Loading data..."):
                    product_data = load_products(category)
                    review_data = load_reviews(category)
                return product_data, review_data

            # Load data
            product_data, review_data = load_data(selected_category)

            if product_data is None or review_data is None:
                st.error("Failed to load data. Please check the category or try again.")
            else:
                with st.spinner("Applying filters..."):
                    # Apply filters
                    filtered_reviews = filter(
                        category=selected_category,
                        min_text_length=min_text_length,
                        start_date=start_date,
                        end_date=end_date,
                        min_reviews_per_product=min_reviews_per_product,
                        min_average_rating=min_average_rating,
                    )

                if filtered_reviews.empty:
                    st.write("No data matches these filters.")
                else:
                    st.success("Filtering successful!")
                    st.subheader("Filtered Data:")
                    st.dataframe(filtered_reviews)
                    st.subheader("Summary of Filtered Data:")
                    st.write(f"Number of reviews: {len(filtered_reviews)}")

                    # Store filtered reviews and filter settings in session state
                    st.session_state["filtered_reviews"] = filtered_reviews
                    st.session_state["selected_category"] = selected_category
                    st.session_state["min_text_length"] = min_text_length
                    st.session_state["start_date"] = start_date
                    st.session_state["end_date"] = end_date
                    st.session_state["min_reviews_per_product"] = min_reviews_per_product
                    st.session_state["min_average_rating"] = min_average_rating
                    st.session_state["store"] = store
                    st.session_state["search_value"] = search_value
                    
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Page 2: NLP Results
elif nav_option == "NLP Results":
    st.subheader("NLP Results")
    st.write("Here you can display the results of your Natural Language Processing analysis.")

    if "filtered_reviews" in st.session_state:
        filtered_reviews = st.session_state["filtered_reviews"]
        st.write("Filtered data is available for NLP analysis.")
        
        st.subheader("Word Cloud from Reviews")
        col1, col2 = st.columns([3, 1])  # The first column takes 3 parts, and the second takes 1 part
        
        # In the first column, display the word cloud
        with col1:
            create_wordcloud_from_df_for_app(filtered_reviews)
        
        # In the second column, you can place any other content (e.g., matrix, table, etc.)
        with col2:
            st.write("Here you can place your matrix or any other content.")

        # st.subheader("Word Cloud from Reviews")
        # create_wordcloud_from_df_for_app(filtered_reviews)
        
        if st.button("Perform LDA Analysis"):
            try:
                # Use filtered_reviews for LDA
                texts_bow, dictionary, id2token = preprocess_text(filtered_reviews, 100, 0.85)
                n_topics = 9
                model = LDA_training(filtered_reviews, texts_bow, dictionary, id2token, n_topics, 1000, 1, 100)

                st.session_state["lda_model"] = model
                st.session_state["n_topics"] = n_topics

                st.subheader("Top Words for Topics")
                #top_words_summary = display_top_words_for_topics(model, n_topics)
                #st.markdown(top_words_summary)

                topics_df = display_top_words_for_topics2(model, n_topics)
                topics_df = topics_df.reset_index(drop=True)
        
                # Display the DataFrame with custom styling
                styled_df = topics_df.style.set_table_styles(
                    [{
                        'selector': 'thead th',
                        'props': [('background-color', '#F6B17A'), ('color', 'black'), ('font-weight', 'bold'), ('padding', '10px')]
                    }, {
                        'selector': 'tbody tr:nth-child(odd)',
                        'props': [('background-color', '#f2f2f2')]
                    }, {
                        'selector': 'tbody tr:nth-child(even)',
                        'props': [('background-color', '#ffffff')]
                    }, {
                        'selector': 'table',
                        'props': [('border-collapse', 'collapse'), ('width', '100%')]
                    }, {
                        'selector': 'td, th',
                        'props': [('padding', '12px'), ('text-align', 'left')]
                    }, {
                        'selector': 'tbody td:first-child',
                        'props': [('display', 'none')]  # Hide the first column (index column)
                    }]
                ).hide(axis="index")  # Hide the default pandas index
                
                # Display the styled dataframe
                st.dataframe(styled_df)

                st.subheader("Review-Topic Matrix")
                matrix = create_review_topic_matrix_stars_for_app(filtered_reviews, model, texts_bow, n_topics, 0.3)

            except Exception as e:
                st.error(f"An error occurred while performing LDA analysis: {e}")
    else:
        st.warning("No filtered data available. Please filter data on the 'Filter Products' page first.")

# Page 3: Model Results
elif nav_option == "Model Results":
    st.subheader("Model Results")
    st.write("These are example model evaluation metrics. There can be plots or whatever")
    # Example: Display dummy metrics
    try:
        st.metric("Model Accuracy", "92.5%")
        st.metric("F1 Score", "89.7%")
        st.metric("Precision", "90.1%")
        st.metric("Recall", "88.3%")
    except Exception as e:
        st.error(f"An error occurred: {e}")
