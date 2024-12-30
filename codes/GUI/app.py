import streamlit as st
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Adding root path to the project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
os.environ["STREAMLIT_CONFIG_FILE"] = os.path.abspath(
    os.path.join(project_root, ".streamlit/config.toml")
)

from codes.loading_and_filtering.parameters import categories
from codes.loading_and_filtering.filter import filter
from codes.loading_and_filtering.data_loader import load_products, load_reviews

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
    # Filter bar content
    selected_category = st.selectbox("Choose category:", categories)

    min_text_length = st.number_input("Minimal number of words in review", min_value=1, value=10)
    start_date = st.date_input("Start date:")
    end_date = st.date_input("End date:")
    min_reviews_per_product = st.number_input("Minimal number of reviews per product", min_value=1, value=10)
    min_average_rating = st.slider("Minimal average rating of product:", 1.0, 5.0, 4.0)
    store = st.text_input("Filter store by name (optional)")
    search_value = st.text_input("Filter products by name (optional)")

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
                # Apply filters
                filtered_reviews = filter(
                    category=selected_category,
                    min_text_length=min_text_length,
                    start_date=start_date,
                    end_date=end_date,
                    min_reviews_per_product=min_reviews_per_product,
                    min_average_rating=min_average_rating,
                )

                st.success("Filtering successful!")
                st.subheader("Filtered Data:")

                if filtered_reviews.empty:
                    st.write("No data matches these filters.")
                else:
                    st.dataframe(filtered_reviews)
                    st.subheader("Summary of Filtered Data:")
                    st.write(f"Number of reviews: {len(filtered_reviews)}")
                    st.write(f"Average rating: {filtered_reviews['average_rating'].mean():.2f}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Page 2: NLP Results
elif nav_option == "NLP Results":
    st.subheader("NLP Results")
    st.write("Here you can display the results of your Natural Language Processing analysis.")
    # Example: Sentiment analysis visualization
    try:
        sentiment_scores = np.random.rand(100)  # Dummy sentiment scores for demonstration
        st.line_chart(sentiment_scores)
        st.write("This is an example sentiment analysis chart.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

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
