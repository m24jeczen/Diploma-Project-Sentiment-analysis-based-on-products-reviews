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
from codes.loading_and_filtering.data_loader import *
from codes.topic_modeling.text_preprocessing import preprocess_text
from codes.topic_modeling.LDA import LDA_training
from codes.topic_modeling.visualization import *
from codes.visualizations import *

from codes.deep_learning.download_model import *
from codes.deep_learning.predict_on_model import *
from codes.deep_learning.preprocessing import *
from codes.deep_learning.rating_analysis import *


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

if "available_models" not in st.session_state:
    st.session_state.available_models = {}

if "selected_model_name" not in st.session_state:
    st.session_state.selected_model_name = None

if "selected_model_path" not in st.session_state:
    st.session_state.selected_model_path = None

# Navigation bar for switching between pages (after filtering)
def render_nav_bar():
    return st.radio(
        "",
        ["NLP Results", "Models Results"],
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

    if st.button("Go to Models Results"):
        st.session_state.page = "Models Results"
        st.rerun()

# Page 1: Filter Products
elif st.session_state.page == "Filter Products":
    st.write("### Add External Data")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("Data loaded successfully!")

    st.markdown(
        """
        <h3 style="text-align: center;">Set Filters</h3>
        """,
        unsafe_allow_html=True
    )

    selected_category = st.selectbox("Choose category:", categories, index=categories.index("CDs_and_Vinyl"))

    stores = list(load_store_data(selected_category).keys())

    selected_store = st.selectbox("Filter by store (optional):", [""] + stores)

    min_text_length = st.number_input("Minimal number of words in review", min_value=1, value=20)
    start_date = st.date_input("Start date:", value=pd.to_datetime("2018-01-01"))
    end_date = st.date_input("End date:", value=pd.to_datetime("2018-03-01"))
    min_reviews_per_product = st.number_input("Minimal number of reviews per product", min_value=1, value=10)
    min_average_rating = st.slider("Minimal average rating of product:", 1.0, 5.0, 1.0)
    search_value = st.text_input("Filter products by name (optional)")

    st.markdown("""<h3 style="text-align: center;">Model Selection</h3>""", unsafe_allow_html=True)
    st.write("### Available model Selection")
    if st.button("Show Available Models"):
        st.session_state.available_models = get_available_models()
    if st.session_state.available_models:
        st.write("### Available Models:")
        
        # Keep the selection box visible until a model is confirmed
        selected_model_name = st.selectbox(
            "Select a pre-trained model:", 
            list(st.session_state.available_models.keys()), 
            key="model_selectbox"
        )
        
        # Update the session state only when the button is clicked
        if st.button("Confirm Selected Model"):
            st.session_state.selected_model_name = selected_model_name
            st.session_state.selected_model_path = st.session_state.available_models[selected_model_name]
            st.success(f"Selected model: {st.session_state.selected_model_name}")
            st.write(f"Selected Model Path: {st.session_state.selected_model_path}")

        if st.session_state.available_models is None:
            st.warning("No models found in the models directory.")

    st.write("### New model Selection")

    # Adding multiple model selection functionality
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = []

    model_name = st.selectbox("Choose a model:", ["bert_classification", "bert_regression", "roberta_sentiment", "bert_sentiment_prediction", "vader_prediction"], key="model_name")
    prediction_task = st.selectbox("Choose prediction target:", ["rating", "sentiment"], key="prediction_task")

    # Show training parameters below model and prediction target selection
    st.write("### Training Parameters")
    max_epochs = st.number_input("Max Epochs", min_value=1, value=3, step=1, key="max_epochs")
    batch_size = st.number_input("Batch Size", min_value=1, value=16, step=1, key="batch_size")
    lr = st.number_input("Learning Rate", min_value=1e-6, value=2e-5, step=1e-6, format="%e", key="lr")
    max_len = st.number_input("Max Length", min_value=1, value=128, step=1, key="max_len")
    val_split = st.slider("Validation Split", min_value=0.0, max_value=1.0, value=0.2, step=0.01, key="val_split")
    localname = st.text_input("Model name", value='model_new',key="localname")
    early_stopping = st.checkbox("Enable Early Stopping", value=True, key="early_stopping")
    patience = st.number_input("Early Stopping Patience", min_value=1, value=3, step=1, key="patience")
    dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=1.0, value=0.0, step=0.05, key="dropout_rate")

    if st.button("Add Model"):
        st.session_state.selected_models.append({
            "model_name": model_name,
            "task": prediction_task,
            "parameters": {
                "max_epochs": max_epochs,
                "batch_size": batch_size,
                "lr": lr,
                "max_len": max_len,
                "val_split": val_split,
                "localname": localname,
                "early_stopping": early_stopping,
                "patience": patience,
                "dropout_rate": dropout_rate
            }
        })
        st.success(f"Added model: {model_name} for task: {prediction_task} with parameters.")

    # Display selected models and parameters
    if st.session_state.selected_models:
        st.write("### Selected Models")
        for idx, model_info in enumerate(st.session_state.selected_models):
            st.write(f"{idx + 1}. Model: {model_info['model_name']}, Task: {model_info['task']}")
            st.json(model_info['parameters'])

    if st.button("Apply Filters and train models"):
        try:
            with st.spinner("Loading data and applying filters..."):
                filtered_reviews = filter(
                    category=selected_category,
                    min_text_length=min_text_length,
                    start_date=start_date,
                    end_date=end_date,
                    min_reviews_per_product=min_reviews_per_product,
                    min_average_rating=min_average_rating,
                    store=selected_store if selected_store else None

                )

            if filtered_reviews.empty:
                st.write("No data matches these filters.")
            else:
                st.success("Filtering successful!")
                st.write(f"Number of records after filtering: {len(filtered_reviews)}")
                st.dataframe(filtered_reviews, height=400)

                filtered_reviews = map_ratings_into_sentiment(filtered_reviews, positive_threshold=4)
                if st.session_state.selected_models:
                    for idx, model_info in enumerate(st.session_state.selected_models):
                        name = model_info["parameters"]["localname"]
                        if model_info["task"] == "rating" and model_info["model_name"] == "bert_classification":
                            model_path = rf".\models\classification\{name}"
                            filtered_reviews[f'target_{name}'] = [int(x)-1 for x in filtered_reviews["rating"]]
                            with st.spinner(f"Training model {name}..."):    
                                train_model(filtered_reviews, 'classification',f'target_{name}',5, **model_info["parameters"])
                            with st.spinner(f"Predicting on model {name}..."):
                                filtered_reviews[f"predictions_{name}"]=predict_on_tuned_model(filtered_reviews,model_path)
                        if model_info["task"] == "rating" and model_info["model_name"] == "bert_regression":
                            model_path = rf".\models\regression\{name}"
                            filtered_reviews[f'target_{name}'] = (filtered_reviews["rating"]-1)/4
                            with st.spinner(f"Training model {name}..."):    
                                train_model(filtered_reviews, 'regression',f'target_{name}',5, **model_info["parameters"])
                            with st.spinner(f"Predicting on model {name}..."):
                                filtered_reviews[f"predictions_{name}"]=predict_on_tuned_model(filtered_reviews,model_path)
                        if model_info["model_name"]=="bert_sentiment_prediction":
                            model_path = rf".\models\sentiment_prediction\{name}"
                            with st.spinner(f"Training model {name}..."):    
                                train_model(filtered_reviews,target="star_based_sentiment",num_classes=2, **model_info["parameters"])
                            with st.spinner(f"Predicting on model {name}..."):
                                filtered_reviews[f"predictions_{name}"]=predict_on_tuned_model(filtered_reviews,model_path)

                if "selected_model_path" in st.session_state and st.session_state.selected_model_path is not None:
                    try:
                        name = st.session_state.selected_model_name
                        with st.spinner(f"Predicting on {name} using the chosen pre-trained model..."):
                            filtered_reviews[f'target_{name}'] = (filtered_reviews["rating"]-1)/4
                            filtered_reviews[f"predictions_{name}"] = predict_on_tuned_model(filtered_reviews, st.session_state.selected_model_path)
                            st.success("Prediction completed successfully!")
                    except Exception as e:
                        st.error(f"An error occurred while predicting: {e}")

                st.session_state.filtered_reviews = filtered_reviews
                st.session_state.page = "Menu"
                if st.button("Back to Menu"):
                    st.rerun()  

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Page 2: Ratings and words analysis
elif st.session_state.page == "Ratings and words analysis":
    st.write("### Ratings and words analysis")
    
    if "filtered_reviews" in st.session_state:
        filtered_reviews = st.session_state.filtered_reviews
        
        st.write("#### Word Clouds by Star Rating")
        try:
            # Only generate word clouds if not already cached
            if "word_clouds_by_rating" not in st.session_state:
                st.session_state.word_clouds_by_rating = create_tfidf_wordcloud(filtered_reviews)
            
            # Filter only ratings with data and word clouds generated
            word_clouds_available = {
                rating: st.session_state.word_clouds_by_rating[rating] 
                for rating in st.session_state.word_clouds_by_rating
                if rating in st.session_state.word_clouds_by_rating
            }
            
            # Display only columns for ratings with data
            unique_ratings = sorted(word_clouds_available.keys())
            if unique_ratings:
                cols = st.columns(len(unique_ratings))
                for i, rating in enumerate(unique_ratings):
                    with cols[i]:
                        st.image(word_clouds_available[rating], caption=f"{rating}-Star Reviews")
            else:
                st.warning("No word clouds available for the selected reviews.")
        
        except Exception as e:
            st.error(f"An error occurred while generating word clouds: {str(e)}")
        
        

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


# Page 3: Models Results
elif st.session_state.page == "Models Results":
    st.write("### Models Results")

    if "filtered_reviews" in st.session_state and "selected_models" in st.session_state:
        try:
            filtered_reviews = st.session_state.filtered_reviews
            for idx, model_info in enumerate(st.session_state.selected_models):
                name = model_info["parameters"]["localname"]
                st.write(f"### Results for Model: {name} (Task: {model_info["task"]})")

                # Placeholder for predictions and metrics
                st.write("Predictions and metrics will be displayed here.")
                if model_info["task"] == "rating" and model_info["model_name"] == "bert_classification":
                    img_stream = heatmap(filtered_reviews, f"target_{name}", f"predictions_{name}")
                    st.image(img_stream, caption="Heatmap of Predictions", use_container_width=True)
                if model_info["task"] == "rating" and model_info["model_name"] == "bert_regression":
                    img_stream = heatmap(filtered_reviews, "rating", f"predictions_{name}")
                    st.image(img_stream, caption="Heatmap of Predictions", use_container_width=True)
                if model_info["model_name"]=="bert_sentiment_prediction":
                    img_stream = heatmap(filtered_reviews,"star_based_sentiment",f"predictions_{name}")
                    st.image(img_stream, caption="Heatmap of Predictions", use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")
    
    if "selected_model_path" in st.session_state and "filtered_reviews" in st.session_state and st.session_state.selected_model_name is not None:   
            name = st.session_state.selected_model_name 
            img_stream = heatmap(filtered_reviews, f"target_{name}", f"predictions_{name}")
            st.image(img_stream, caption="Heatmap of Predictions", use_column_width=True)
    else:
        st.warning("No filtered data or model results available. Please go to 'Filter Products' page first and apply filters.")

    if st.button("Back to Menu"):
        st.session_state.page = "Menu"
        st.rerun()
