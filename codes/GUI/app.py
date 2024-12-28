# It is a very basic version! - further changes to be done
# ! streamlit is a web app - it wont work so that it pops as another window. But i dont think its an issue
import streamlit as st

import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import os



# Adding root path to the project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(project_root)
sys.path.append(project_root)
os.environ["STREAMLIT_CONFIG_FILE"] = os.path.abspath(
    os.path.join(project_root, ".streamlit/config.toml")
)
    
from codes.loading_and_filtering.parameters import categories
from codes.loading_and_filtering.filter import filter
from codes.loading_and_filtering.data_loader import load_products, load_reviews

st.set_page_config(page_title="Amazon products", layout="wide")

st.markdown(
    """
    <h1 style="color:#F6B17A;">Filter Amazon Products</h1>
    """, 
    unsafe_allow_html=True
)

# st.title("Filter amazon products")
 

st.sidebar.header("Choose category and set filters")
selected_category = st.sidebar.selectbox("Choose category:", categories)

min_text_length = st.sidebar.number_input("Minimal number of words in review", min_value=1, value=10)
start_date = st.sidebar.date_input("Start date:")
end_date = st.sidebar.date_input("End date:")
min_reviews_per_product = st.sidebar.number_input("Minimal number of reviews per product", min_value=1, value=10)
min_average_rating = st.sidebar.slider("Minimal average rating of product:", 1.0, 5.0, 4.0)
store = st.sidebar.text_input("Filter store by name (optional)")
search_value = st.sidebar.text_input("Filter products by name (optional)")

# Load the products and reviews data first
def load_data(category):
    # Load products and reviews for the selected category
    st.spinner("Loading data...")  # Add a spinner while loading data
    product_data = load_products(category)
    review_data = load_reviews(category)
    return product_data, review_data



if st.sidebar.button("Filter"):
    try:
        with st.spinner("Loading and filtering data..."):
            # Load data
            product_data, review_data = load_data(selected_category)

            if product_data is None or review_data is None:
                st.error("Failed to load data. Please check the category or try again.")

            # Now apply the filter to the loaded data
            filtered_reviews = filter(
                category=selected_category,
                min_text_length=min_text_length,
                start_date=start_date,
                end_date=end_date,
                min_reviews_per_product=min_reviews_per_product,
                min_average_rating=min_average_rating,
            )

            st.success("Filtering successful!")
            st.subheader("Filtered data:")

            if filtered_reviews.empty:
                st.write("No data matches these filters.")
            else:
                st.dataframe(filtered_reviews)

                st.subheader("Summary of filtered data:")
                st.write(f"Number of reviews: {len(filtered_reviews)}")
                st.write(f"Average rating: {filtered_reviews['average_rating'].mean():.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
