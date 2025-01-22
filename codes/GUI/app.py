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

def process_file_in_chunks(file):
    chunk_size = 10000  # Adjust the chunk size based on your needs
    chunk_list = []

    for chunk in pd.read_csv(file, chunksize=chunk_size):
        chunk_list.append(chunk)
    
    df = pd.concat(chunk_list)
    return df

# Set page configuration
st.set_page_config(page_title="Amazon Products", layout="wide")

# Inside styles css is the custom colors for the app
css_file_path = os.path.join(project_root, "static", "styles.css")
with open(css_file_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="
        background-color: #141824;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    ">
        <h1 style="color: #F6B17A; font-size: 50px; margin: 0;">
            AMAZON PRODUCTS ANALYSIS
        </h1>
    </div>
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
st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #F6B17A; /* Default background color */
        color: #1C2138; /* Default text color */
        padding: 15px 30px;
        font-size: 18px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s, transform 0.2s, color 0.3s;
    }
    div.stButton > button:hover {
        background-color: #E89D60; /* Hover background color */
        color: #FFFFFF; /* Hover text color */
        transform: scale(1.05);
    }
    div.stButton > button:focus {
        background-color: #D97B4E; /* Focus (clicked) background color */
        color: #FFFFFF; /* Focus (clicked) text color */
        outline: none;
        transform: scale(1.05);
    }
    div.stButton > button:active {
        background-color: #D97B4E; /* Active background color */
        color: #FFFFFF; /* Active text color */
        transform: scale(0.95);
    }
    </style>
    """,
    unsafe_allow_html=True,
)





# Menu Page (Main Entry Point)
if st.session_state.page == "Menu":
    st.write("")
    cols = st.columns([1,5])
    with cols[0]:
        st.write("")
        st.write("")
        st.write("### Main Menu")
        st.write("")
        if st.button("Go to Filter Products"):
            st.session_state.page = "Filter Products"
            st.rerun()

        if st.button("Go to ratings and words analysis"):
            st.session_state.page = "Ratings and words analysis"
            st.rerun()

        if st.button("Go to Models Results"):
            st.session_state.page = "Models Results"
            st.rerun()
    with cols[1]:
        st.markdown("""
        <div style="
            background-color: #1C2138; 
            color: #E2E8F0; 
            padding: 20px; 
            border-radius: 10px; 
            text-align: center; 
            font-family: 'sans-serif'; 
            line-height: 1.6;">
            <h1 style="color: #F6B17A; font-size: 35px; margin-bottom: 10px;">
                Welcome to the Amazon Products Analysis App!
            </h1>
            <p style="font-size: 18px; margin-bottom: 20px;">
                This interactive application provides powerful tools to analyze Amazon product reviews using 
                advanced Natural Language Processing (NLP) and machine learning. Here's what you can do:
            </p>
            <ul style="text-align: left; max-width: 800px; margin: 0 auto; padding-left: 20px; font-size: 16px;">
                <li><strong>Filter and Explore Reviews:</strong> Load and filter review data based on criteria like rating, review length, date range, and product categories.</li>
                <li><strong>Analyze Ratings and Trends:</strong> Visualize rating distributions, trends, and create insightful word clouds segmented by rating.</li>
                <li><strong>Model Predictions:</strong> Train and evaluate machine learning models for sentiment analysis, rating prediction, and more using your data or pre-trained models.</li>
                <li><strong>Topic Modeling:</strong> Discover topics from reviews using Latent Dirichlet Allocation (LDA).</li>
                <li><strong>Custom Models:</strong> Configure and train models with adjustable parameters for specific predictions.</li>
            </ul>
            <h2 style="color: #F6B17A; font-size: 24px; margin-top: 30px;">
                Navigation Guide
            </h2>
            <ul style="text-align: left; max-width: 800px; margin: 0 auto; padding-left: 20px; font-size: 16px;">
                <li><strong>Filter Products:</strong> Filter Amazon data or upload external data. Choose pre-trained models or train your own with selected hyperparameters. 
                <span style="color: #E89D60;"><strong>Warning:</strong></span> Returning to this page will reset all filtering, models, and analysis data. After filtering, 
                you can navigate freely between other pages without losing progress.</li>
                <li><strong>Ratings and Words Analysis:</strong> Explore and analyze ratings (if available) and textual data. Perform LDA topic modeling for deeper insights.</li>
                <li><strong>Models Results:</strong> Evaluate and inspect the results of your selected or trained models, with detailed performance metrics and insights.</li>
            </ul>
            <p style="margin-top: 20px; font-size: 18px; color: #F6B17A;">
                Get started now to uncover insights from your data!
            </p>
        </div>
        """, unsafe_allow_html=True)



# Page 1: Filter Products
elif st.session_state.page == "Filter Products":
    st.write("## Using external data")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if "df_external" not in st.session_state:
        st.session_state.external = True
    

    if uploaded_file:
        try:
            df_external = process_file_in_chunks(uploaded_file)
            st.success("Data loaded successfully!")
            st.write("#### Preview of the Data")
            st.dataframe(df_external.head(100), height=400)
            
            rating_column = None
            # Display columns for selection
            columns = df_external.columns.tolist()
            text_column = st.selectbox("Select the text column:", columns)
            rating_column = st.selectbox("Select the rating column:", [None] + columns)

            # Save column selections to session state
            df_external.rename(columns={text_column: 'text'}, inplace=True)

            st.session_state.text_column = 'text'
            # if "rating_column" not in st.session_state:
            #     st.session_state.rating_column = None

            st.success(f"Text column set to: {text_column}")
            if rating_column:
                positive_threshold = st.radio(
                    "Select the positive threshold for ratings:", 
                    options=[3, 4], 
                    index=1  # Default to 4
                )
                st.success(f"Rating column set to: {rating_column}")
                df_external.rename(columns={rating_column: 'rating'}, inplace=True)
                st.session_state.rating_column = 'rating'
                df_external = map_ratings_into_sentiment(df_external, positive_threshold)
                df_external = df_external[pd.to_numeric(df_external['rating'], errors='coerce').notnull()]
                df_external['rating'] = df_external['rating'].astype(int)

            else:
                st.info("No rating column selected.")
            text_column_type = df_external['text'].dtype
            

            # Convert the column to string, handling missing values
            df_external = df_external[df_external['text'].notna()]  
                
            df_external['text'] = df_external['text'].astype('string')
            df_external = df_external[df_external['text'].str.strip() != '']  # Drop rows where 'text' is an empty string

            if "predykty" not in st.session_state:
                st.session_state.df_external = df_external

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

        st.write("### Available trained models for prediction:")
        predict_on_roberta_selected = st.checkbox("Predict on RoBERTa", value=False)
        predict_on_vader_selected = st.checkbox("Predict on VADER", value=False)
        text_column_type = df_external['text'].dtype
        st.session_state.predict_on_roberta_selected = predict_on_roberta_selected
        st.session_state.predict_on_vader_selected = predict_on_vader_selected
        st.write("### Available local trained models for prediction:")
        st.write("### Model Selection")

        # Step 1: Choose task type
        # Step 1: Choose task type 
        task_type = st.radio("Select the task type:", ("classification", "regression", "sentiment_prediction"), key="task_type_radio")

        # Step 2: Show available models dynamically
        st.session_state.available_models = get_available_models(task_type, parent_dir="models")
        #print("Available models:", st.session_state.available_models)

        if st.session_state.get("available_models"):
            st.write(f"#### Available {task_type.replace('_', ' ').capitalize()} Models:")

            # Keep the selection box visible until a model is confirmed
            selected_available_model_name = st.selectbox(
                "Select a pre-trained model:", 
                list(st.session_state.available_models.keys()), 
                key="model_selectbox"
            )

            # Store the selection immediately in the session state
            st.session_state["current_selected_model"] = selected_available_model_name
            #print(f'Selected model from dropdown: {selected_available_model_name}')

            # Add a confirm button to finalize the selection
            if st.button("Confirm Selected Model"):
                st.session_state.selected_available_model_name = st.session_state["current_selected_model"]
                st.session_state.selected_model_path = st.session_state.available_models[st.session_state["current_selected_model"]]
                st.success(f"Selected model: {st.session_state.selected_available_model_name}")
                st.write(f"Selected Model Path: {st.session_state.selected_model_path}")

        # Debugging Output
        st.write(f"Selected model name: {st.session_state.get('selected_available_model_name')}")
        st.write(f"Selected model path: {st.session_state.get('selected_model_path')}")

        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        if st.button("Predict"):

            if "selected_model_path" in st.session_state and st.session_state.selected_model_path is not None:
                try:
                    name = st.session_state.selected_available_model_name
                    with st.spinner(f"Predicting on {name} using the chosen pre-trained model..."):
                        df_external[f"predictions_{name}"] = predict_on_tuned_model(df_external, st.session_state.selected_model_path)
                        st.success("Prediction completed successfully!")
                except Exception as e:
                    st.error(f"An error occurred while predicting: {e}")

            if predict_on_vader_selected or predict_on_roberta_selected:
                if predict_on_roberta_selected:
                        try: 
                            with st.spinner("Predicting on RoBERTa..."):
                                text_column_type = df_external['text'].dtype
                                df_external["predictions_roberta"] = predict_on_roberta(df_external)
                                st.session_state.df_external = df_external

                            st.success("Prediction on RoBERTa completed successfully!")
                        except Exception as e:  
                            st.error(f"An error occurred while predicting on RoBERTa: {e}")
                if predict_on_vader_selected:
                    try: 
                        with st.spinner("Predicting on VADER..."):
                            df_external["predictions_vader"] = predict_on_vader(df_external)
                            st.session_state.df_external = df_external

                        st.success("Prediction on VADER completed successfully!")
                        #st.dataframe(df_external.head(100), height=400)
                    
                    except Exception as e:  
                        st.error(f"An error occurred while predicting on VADER: {e}")
            st.session_state.predykty = 1
        
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        df_external = st.session_state.df_external

        if st.button("Back to Menu"):
            st.session_state.page = "Menu"
            st.rerun()

    else:
        st.write("## Using local data")
        if "filtered_reviews" not in st.session_state:
            st.session_state.filtered_reviews = pd.DataFrame()      
        cols = st.columns([3, 5])  
        with cols[0]:

            st.session_state.external = False
            st.session_state.predict_on_roberta_selected = False
            st.session_state.predict_on_vader_selected = False
            selected_category = st.selectbox("Choose category:", categories, index=categories.index("CDs_and_Vinyl"))

            stores = list(load_store_data(selected_category).keys())

            selected_stores = st.multiselect("Filter by stores (optional):", stores)

            min_text_length = st.number_input("Minimal number of words in review", min_value=1, value=20)
            start_date = st.date_input("Start date:", value=pd.to_datetime("2018-01-01"))
            end_date = st.date_input("End date:", value=pd.to_datetime("2018-03-01"))
            min_reviews_per_product = st.number_input("Minimal number of reviews per product", min_value=1, value=10)
            min_average_rating = st.slider("Minimal average rating of product:", 1.0, 5.0, 1.0)
            positive_threshold = st.radio(
                "Select the positive threshold for ratings:", 
                options=[3, 4], 
                index=1  # Default to 4
            )
            search_value = st.text_input("Filter products by name (optional)")

        with cols[1]:
            if st.button("Apply Filters:"):
                try:
                    with st.spinner("Loading data and applying filters..."):
                        if selected_stores:
                            filtered_reviews = pd.concat([
                                filter(
                                    category=selected_category,
                                    min_text_length=min_text_length,
                                    start_date=start_date,
                                    end_date=end_date,
                                    min_reviews_per_product=min_reviews_per_product,
                                    min_average_rating=min_average_rating,
                                    store=store,
                                    search_value=search_value if search_value else None
                                ) for store in selected_stores
                            ])
                        else:
                            filtered_reviews = filter(
                                category=selected_category,
                                min_text_length=min_text_length,
                                start_date=start_date,
                                end_date=end_date,
                                min_reviews_per_product=min_reviews_per_product,
                                min_average_rating=min_average_rating,
                                search_value=search_value if search_value else None
                            )

                    if filtered_reviews.empty:
                        st.write("No data matches these filters.")
                    else:
                        st.success("Filtering successful!")
                        st.write(f"Number of records after filtering: {len(filtered_reviews)}")
                        st.dataframe(filtered_reviews, height=485)

                        filtered_reviews = map_ratings_into_sentiment(filtered_reviews, positive_threshold)
                        st.session_state.filtered_reviews = filtered_reviews
                except Exception as e:
                    st.error(f"An error occurred in filtering data: {e}")
            
        st.write("## Models Selection")
        st.write("### Available trained models for prediction:")


        # Step 1: Choose task type
        task_type = st.radio("Select the task type:", ("classification", "regression", "sentiment_prediction"), key="task_type_radio")

        # Step 2: Show available models dynamically
        st.session_state.available_models = get_available_models(task_type, parent_dir="models")

        if st.session_state.get("available_models"):
            st.write(f"#### Available {task_type.replace('_', ' ').capitalize()} Models:")

            # Keep the selection box visible until a model is confirmed
            selected_available_model_name = st.selectbox(
                "Select a pre-trained model:", 
                list(st.session_state.available_models.keys()), 
                key="model_selectbox"
            )

            # Add a confirm button to finalize the selection
            if st.button("Confirm Selected Model"):
                st.session_state.selected_available_model_name = selected_available_model_name
                st.session_state.selected_model_path = st.session_state.available_models[selected_available_model_name]
                st.success(f"Selected model: {st.session_state.selected_available_model_name}")
                st.write(f"Selected Model Path: {st.session_state.selected_model_path}")

            predict_on_roberta_selected = st.checkbox("Predict on RoBERTa", value=False)
            predict_on_vader_selected = st.checkbox("Predict on VADER", value=False)

            st.session_state.predict_on_roberta_selected = predict_on_roberta_selected
            st.session_state.predict_on_vader_selected = predict_on_vader_selected

        st.write("## Train new model")

        # Adding multiple model selection functionality
        if "selected_models" not in st.session_state:
            st.session_state.selected_models = []

        model_descriptions = {
            "bert_classification": "Predicts ratings using classification.",
            "bert_regression": "Predicts ratings using regression.",
            "bert_sentiment_prediction": "Predicts sentiment."
        }

        # Display descriptions for the models
        st.write("#### Models Descriptions")
        st.write("- **bert_classification**: Predicts ratings using classification.")
        st.write("- **bert_regression**: Predicts ratings using regression.")
        st.write("- **bert_sentiment_prediction**: Predicts sentiment.")

        # Allow the user to select a model
        model_name = st.selectbox("Choose a model:", ["bert_classification", "bert_regression", "bert_sentiment_prediction"], key="model_name")

        # Display the selected model
        st.write(f"Selected Model: {model_name}")

        # Show training parameters below model and prediction target selection
        st.write("#### Training Parameters")
        max_epochs = st.number_input("Max Epochs", min_value=1, value=3, step=1, key="max_epochs")
        batch_size = st.number_input("Batch Size", min_value=1, value=16, step=1, key="batch_size")
        lr = st.number_input("Learning Rate", min_value=1e-6, value=2e-5, step=1e-6, format="%e", key="lr")
        max_len = st.number_input("Max Length", min_value=1, value=128, step=1, key="max_len")
        val_split = st.slider("Validation Split", min_value=0.0, max_value=1.0, value=0.2, step=0.01, key="val_split")
        localname = st.text_input("Model name", value='model_new',key="localname")
        early_stopping = st.checkbox("Enable Early Stopping", value=True, key="early_stopping")
        patience = st.number_input("Early Stopping Patience", min_value=1, value=3, step=1, key="patience")
        dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=1.0, value=0.0, step=0.05, key="dropout_rate")

        if st.button("Add Model with selected parameters"):
            st.session_state.selected_models.append({
                "model_name": model_name,
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
            st.success(f"Added model: {model_name} with parameters.")

        # Display selected models and parameters
        if st.session_state.selected_models:
            st.write("### Selected Models")
            for idx, model_info in enumerate(st.session_state.selected_models):
                st.write(f"{idx + 1}. Model: {model_info['model_name']}")
                st.json(model_info['parameters'])
        
        filtered_reviews = st.session_state.filtered_reviews
        st.write("")
        st.write("")
        if filtered_reviews.empty==False:
            if st.session_state.selected_models:
                if st.button("Train and Predict"):
                    try: 
                        for idx, model_info in enumerate(st.session_state.selected_models):
                            name = model_info["parameters"]["localname"]
                            if model_info["model_name"] == "bert_classification":
                                model_path = rf".\models\classification\{name}"
                                filtered_reviews[f'target_{name}'] = [int(x)-1 for x in filtered_reviews["rating"]]
                                with st.spinner(f"Training model {name}..."):    
                                    train_model(filtered_reviews, 'classification',f'target_{name}',5, **model_info["parameters"])
                                with st.spinner(f"Predicting on model {name}..."):
                                    filtered_reviews[f"predictions_{name}"]=predict_on_tuned_model(filtered_reviews,model_path)+1
                                st.success(f"Prediction with {name} completed successfully!")
                            if model_info["model_name"] == "bert_regression":
                                model_path = rf".\models\regression\{name}"
                                filtered_reviews[f'target_{name}'] = filtered_reviews["rating"]
                                with st.spinner(f"Training model {name}..."):    
                                    train_model(filtered_reviews, 'regression',f'target_{name}',5, **model_info["parameters"])
                                with st.spinner(f"Predicting on model {name}..."):
                                    filtered_reviews[f"predictions_{name}"]=predict_on_tuned_model(filtered_reviews,model_path)
                                st.success(f"Prediction with {name} completed successfully!")
                            if model_info["model_name"]=="bert_sentiment_prediction":
                                model_path = rf".\models\sentiment_prediction\{name}"
                                with st.spinner(f"Training model {name}..."):    
                                    train_model(filtered_reviews,target="star_based_sentiment",num_classes=2, **model_info["parameters"])
                                with st.spinner(f"Predicting on model {name}..."):
                                    filtered_reviews[f"predictions_{name}"]=predict_on_tuned_model(filtered_reviews,model_path)
                                st.success(f"Prediction with {name} completed successfully!")
                            

                    except Exception as e:
                        st.error(f"An error occurred during training and predicting: {e}")
            
            if (
                ("selected_model_path" in st.session_state and st.session_state.selected_model_path is not None) or
                st.session_state.get("predict_on_roberta_selected", False) or
                st.session_state.get("predict_on_vader_selected", False)
            ):
                if st.button("Predict"):
                    if "selected_model_path" in st.session_state and st.session_state.selected_model_path is not None:
                        try:
                            selected_model_path = st.session_state.selected_model_path
                            name = st.session_state.selected_available_model_name
                            if "classification" in selected_model_path:
                                filtered_reviews[f'target_{name}'] = [int(x)-1 for x in filtered_reviews["rating"]]
                                with st.spinner(f"Predicting on pre-trained model {name}..."):
                                    filtered_reviews[f"predictions_{name}"]=predict_on_tuned_model(filtered_reviews,selected_model_path)+1
                                st.success("Prediction completed successfully!")

                            elif "regression" in selected_model_path:
                                filtered_reviews[f'target_{name}'] = filtered_reviews["rating"]
                                with st.spinner(f"Predicting on pre-trained model {name}..."):
                                    filtered_reviews[f"predictions_{name}"]=predict_on_tuned_model(filtered_reviews,selected_model_path)
                                st.success("Prediction completed successfully!")
                            elif "sentiment_prediction" in selected_model_path:
                                with st.spinner(f"Predicting on pre-trained model {name}..."):
                                    filtered_reviews[f"predictions_{name}"]=predict_on_tuned_model(filtered_reviews,selected_model_path)
                                st.success("Prediction completed successfully!")
                            
                        except Exception as e:
                            st.error(f"An error occurred while predicting: {e}")
                    
                    
                    if st.session_state.predict_on_roberta_selected and st.session_state.predict_on_roberta_selected==True:
                        try: 
                            with st.spinner("Predicting on RoBERTa..."):
                                filtered_reviews["predictions_roberta"] = predict_on_roberta(filtered_reviews)
                                st.success("Prediction on RoBERTa completed successfully!")
                        except Exception as e:  
                            st.error(f"An error occurred while predicting on RoBERTa: {e}")

                    if st.session_state.predict_on_vader_selected:
                        try: 
                            with st.spinner("Predicting on VADER..."):
                                filtered_reviews["predictions_vader"] = predict_on_vader(filtered_reviews)
                                st.success("Prediction on VADER completed successfully!")
                        except Exception as e:  
                            st.error(f"An error occurred while predicting on VADER: {e}")
        else:
            st.warning('You have to filter data before training and predicting.')

        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        print(st.session_state.selected_model_path)
        if st.button("Back to Menu"):
            st.session_state.page = "Menu"
            st.rerun()  

# Page 2: Ratings and words analysis
elif st.session_state.page == "Ratings and words analysis":
    st.write("### Ratings and words analysis")
    if ("filtered_reviews" in st.session_state and not st.session_state.filtered_reviews.empty) or ("df_external" in st.session_state and "rating_column" in st.session_state and st.session_state.rating_column is not None):
        if "filtered_reviews" in st.session_state and not st.session_state.filtered_reviews.empty:
            reviews = st.session_state.filtered_reviews
        if "df_external" in st.session_state and "rating_column" in st.session_state:
            reviews = st.session_state.df_external

        st.write("#### Word Clouds by Star Rating")
        try:
            # Only generate word clouds if not already cached
            if "word_clouds_by_rating" not in st.session_state:
                st.write("Generating word clouds...")
                st.session_state.word_clouds_by_rating = create_tfidf_wordcloud(reviews)
            
            records_per_rating = reviews.groupby('rating').size().to_dict()
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
                        st.write(f"{rating}-Star Reviews ({records_per_rating.get(rating, 0)} records)")
                        st.image(word_clouds_available[rating], caption=f"{rating}-Star Reviews")
            else:
                st.warning("No word clouds available for the selected reviews.")
        
        except Exception as e:
            st.error(f"An error occurred while generating word clouds: {str(e)}")

        st.write("#### Rating Distribution and Trends")
        try:
            cols = st.columns(2)
            
            with cols[0]:
                st.write("**Rating Distribution**")
                distrib_img_stream = distribiution_of_rating_for_app(reviews)
                st.image(distrib_img_stream, caption="Rating Distribution")

            with cols[1]:
                if "df_external" in st.session_state:
                    st.write('')
                else:
                    st.write("**Rating Trend Over Time**")
                    trend_img_stream = plot_monthly_avg_app(reviews)
                    st.image(trend_img_stream, caption="Average Rating Over Time")
            cols = st.columns([9,1,7])
            with cols[0]:
                st.write("**Cumulative Distribution of Words Length**")
                word_length_img_stream = plot_number_of_words_percentages_for_app(reviews)
                st.image(word_length_img_stream, caption="Cumulative Distribution of Words Length of Reviews")
            with cols[1]:
                st.write("")
            with cols[2]:
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write('Cumulative distribiuton describes what percentage of texts has same or lower number of words in review. For example if in plot value above 150 Words per review is 90, that means 90% of texts are has maximum 150 words. If too many texts are very short it may indicate a small amount of information that could be obtain from chosen dataset. Moreover one of parameters in training models describes maximum text lenght. Analylizing this plot you can find balance between information and training speed.')

        except Exception as e:
            st.error(f"An error occurred while generating distribution or trends: {str(e)}")
        
        

        # LDA Parameters and Analysis
        st.write("#### LDA Analysis Parameters")
        col1, col2 = st.columns([1,1])
        with col1:
            with st.form("lda_parameters_form"):
                n_topics = st.number_input("Number of Topics", min_value=1, max_value=20, value=9, step=1)
                chunksize = st.number_input("Chunksize", min_value=10, max_value=5000, value=1000, step=10)
                passes = st.number_input("Number of Passes", min_value=1, max_value=100, value=10, step=1)
                iterations = st.number_input("Iterations", min_value=10, max_value=500, value=200, step=10)
                update_every = st.selectbox("Update Every", options=[1, 0], index=0)
                submit_button = st.form_submit_button("Perform LDA Analysis")

        with col2:
            st.markdown("""
            ### What is LDA?
            Latent Dirichlet Allocation (LDA) is a statistical model used for topic modeling. It discovers hidden topics in a collection of text documents and assigns each document a distribution over these topics. It’s widely used for extracting meaningful insights from large text datasets.

            ### Key Parameters Explained:
            - **n_topics**: The number of topics to discover in the text data.
            - **chunksize**: The number of documents processed at a time during training. Larger values speed up training but require more memory.
            - **passes**: The number of complete iterations over the entire dataset. More passes improve accuracy but increase computation time.
            - **iterations**: The maximum number of iterations for the model’s optimization process during each pass.
            - **update_every**: How often the model’s parameters are updated during training. A value of `1` updates after each chunk, while `0` delays updates until all chunks are processed.

            LDA helps uncover topics in your reviews, providing insights into recurring themes or patterns.
            """)


        if "lda_model" not in st.session_state:
            st.session_state.lda_model = None
            st.session_state.topic_word_menu = None
            st.session_state.review_topic_matrix = None
            st.session_state.selected_topic_id = None
        st.write("")
        if submit_button:
            try:
                if "texts_bow" in st.session_state and "dictionary" in st.session_state and "id2word" in st.session_state:
                    texts_bow = st.session_state.texts_bow
                    dictionary = st.session_state.dictionary
                    id2word = st.session_state.id2word
                else:
                    texts_bow, dictionary, id2word = preprocess_text(reviews, 100, 0.85)
                    st.session_state.texts_bow = texts_bow
                    st.session_state.dictionary = dictionary
                    st.session_state.id2word = id2word

                lda_model = LDA_training(
                    reviews, texts_bow, dictionary, id2word,
                    n_topics=n_topics, chunksize=chunksize, passes=passes,
                    iterations=iterations, update_every=update_every
                )

                st.session_state.lda_model = lda_model
                st.session_state.n_topics = n_topics
                st.session_state.topic_word_menu = create_topic_word_menu(lda_model, n_topics)
                
                review_topic_image = create_review_topic_matrix_stars_for_app_new_2(
                    reviews, lda_model, texts_bow, n_topics
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

    elif "df_external" in st.session_state and "rating_column" not in st.session_state:
        st.warning('Since your external data does not have a rating column (or you didnt select one), you can only use the Model Results Page for your selected models.')

    else:
        st.warning("No filtered data available. Please filter data on the home page first.")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    if st.button("Back to Menu"):
        st.session_state.page = "Menu"
        st.rerun()


# Page 3: Models Results
elif st.session_state.page == "Models Results":
    st.write("# Models Results")

    if ("filtered_reviews" in st.session_state and not st.session_state.filtered_reviews.empty) or ("df_external" in st.session_state and "rating_column" in st.session_state and st.session_state.rating_column is not None):
        if "df_external" in st.session_state and "rating_column" in st.session_state and st.session_state.rating_column is not None:
            df_external = st.session_state.df_external
            #st.dataframe(df_external.head(100), height=400)
            reviews = st.session_state.df_external
            #st.dataframe(reviews.head(100), height=400)
        if ("rating_column" in st.session_state and st.session_state.rating_column) or not st.session_state.filtered_reviews.empty:
            if not st.session_state.filtered_reviews.empty:
                reviews = st.session_state.filtered_reviews
                filtered_reviews = st.session_state.filtered_reviews
                if "selected_models" in st.session_state:
                    for idx, model_info in enumerate(st.session_state.selected_models):
                        name = model_info["parameters"]["localname"]
                        st.write(f"#### Results for Model: {name}")
                        st.write("")
                        st.write("")
                        st.write("")
                        if model_info["model_name"] == "bert_classification":
                            col1, col2 = st.columns([3, 4])

                            results = calculate_metrics(filtered_reviews, "rating", f"predictions_{name}")
                            
                            with col1:
                                st.write("##### Metrics:")
                                st.markdown(f"""
                                <div style="
                                    font-size: 17px; 
                                    font-weight: bold; 
                                    color: #F6B17A; 
                                    margin-bottom: 10px;">
                                    MAE: {results['MAE']}
                                </div>
                                """, unsafe_allow_html=True)
                                st.markdown(f"""
                                <div style="
                                    font-size: 17px; 
                                    font-weight: bold; 
                                    color: #F6B17A; 
                                    margin-bottom: 10px;">
                                    Average Accuracy: {results['Average Accuracy']}
                                </div>
                                """, unsafe_allow_html=True)
                                st.write(" ")
                                st.write("##### Metrics Per Label:")
                                metrics_df = results["Metrics Per Label"]
                                st.table(metrics_df)

                                st.markdown("""
                                <div style="
                                    background-color: #424769; 
                                    color: #E2E8F0; 
                                    padding: 13px; 
                                    border-radius: 10px; 
                                    font-family: 'sans-serif'; 
                                    line-height: 1.6; 
                                    margin-top: 10px;">
                                    <h2 style="color: #F6B17A; font-size: 20px; margin-bottom: 1px;">Metrics Explanation</h2>
                                    <p style="font-size: 16px; margin-bottom: 8px;">
                                        <strong>Accuracy:</strong> Accuracy is the simplest metric to evaluate any prediction. It is computed by dividing the 
                                        number of correct predictions by the number of all observations.
                                    </p>
                                    <p style="font-size: 16px; margin-bottom: 8px;">
                                        <strong>F-score:</strong> F-score is the harmonic mean between precision and recall. However, accuracy is unreliable in 
                                        the case of unbalanced data. The F-score punishes more for assigning all values to one class.
                                    </p>
                                    <p style="font-size: 16px; margin-bottom: 8px;">
                                        <strong>MAE (Mean Absolute Error):</strong> MAE is a typical regression metric that is interpretable and easy to understand by many 
                                        people. Despite performing classification, in the context of our data, this metric is still valuable because every class 
                                        can be interpreted as a value in the range of 1 to 5. The mean absolute error provides information about the average 
                                        mistake the model made during predictions.
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            with col2:
                                st.write("")
                                st.write("")
                                img_stream = heatmap(filtered_reviews, "rating", f"predictions_{name}")
                                st.image(img_stream, caption="Heatmap of Predictions", use_container_width=True)


                            col1, col2 = st.columns([2, 3])
                            with col1:
                                st.write("**Rating Distribution**")
                                distrib_img_stream = distribiution_of_rating_for_app(filtered_reviews, f"predictions_{name}")
                                st.image(distrib_img_stream, caption="Rating Distribution")
                            with col2:
                                st.write("")
                                plot_stream = plot_monthly_avg_app(filtered_reviews, label=f"predictions_{name}")   
                                st.image(plot_stream, caption="Monthly Average Rating", use_container_width=True)
                            st.write("#### Word Clouds by Prediction")
                            try:
                                if "word_clouds_by_prediction" not in st.session_state:
                                    st.session_state.word_clouds_by_prediction = create_tfidf_wordcloud(filtered_reviews, rating_column=f"predictions_{name}")
                                records_per_prediction = filtered_reviews.groupby(f"predictions_{name}").size().to_dict()
                                word_clouds_available = {
                                    pred: st.session_state.word_clouds_by_prediction[pred] 
                                    for pred in st.session_state.word_clouds_by_prediction
                                    if pred in st.session_state.word_clouds_by_prediction
                                }
                                unique_predictions = sorted(word_clouds_available.keys())
                                if unique_predictions:
                                    cols = st.columns(len(unique_predictions))
                                    for i, pred in enumerate(unique_predictions):
                                        with cols[i]:
                                            st.write(f"{pred} Predictions ({records_per_prediction.get(pred, 0)} records)")
                                            st.image(word_clouds_available[pred], caption=f"{pred} Predictions")
                                else:
                                    st.warning("No word clouds available for the selected predictions.")
                            except Exception as e:
                                st.error(f"Error generating word clouds: {e}")
                            
                        if model_info["model_name"] == "bert_regression":

                            col1, col2 = st.columns([3, 4])
                            results = calculate_metrics(filtered_reviews, "rating", f"predictions_{name}")
                            with col1:
                                st.write("##### Metrics:")
                                st.markdown(f"""
                                <div style="
                                    font-size: 17px; 
                                    font-weight: bold; 
                                    color: #F6B17A; 
                                    margin-bottom: 10px;">
                                    MAE: {results['MAE']}
                                </div>
                                """, unsafe_allow_html=True)
                                st.markdown(f"""
                                <div style="
                                    font-size: 17px; 
                                    font-weight: bold; 
                                    color: #F6B17A; 
                                    margin-bottom: 10px;">
                                    Average Accuracy: {results['Average Accuracy']}
                                </div>
                                """, unsafe_allow_html=True)
                                st.write(" ")
                                st.write("##### Metrics Per Label:")
                                metrics_df = results["Metrics Per Label"]
                                st.table(metrics_df)

                                st.markdown("""
                                <div style="
                                    background-color: #424769; 
                                    color: #E2E8F0; 
                                    padding: 13px; 
                                    border-radius: 10px; 
                                    font-family: 'sans-serif'; 
                                    line-height: 1.6; 
                                    margin-top: 10px;">
                                    <h2 style="color: #F6B17A; font-size: 20px; margin-bottom: 1px;">Metrics Explanation</h2>
                                    <p style="font-size: 16px; margin-bottom: 8px;">
                                        <strong>Accuracy:</strong> Accuracy is the simplest metric to evaluate any prediction. It is computed by dividing the 
                                        number of correct predictions by the number of all observations.
                                    </p>
                                    <p style="font-size: 16px; margin-bottom: 8px;">
                                        <strong>F-score:</strong> F-score is the harmonic mean between precision and recall. However, accuracy is unreliable in 
                                        the case of unbalanced data. The F-score punishes more for assigning all values to one class.
                                    </p>
                                    <p style="font-size: 16px; margin-bottom: 8px;">
                                        <strong>MAE (Mean Absolute Error):</strong> MAE is a typical regression metric that is interpretable and easy to understand by many 
                                        people. Despite performing classification, in the context of our data, this metric is still valuable because every class 
                                        can be interpreted as a value in the range of 1 to 5. The mean absolute error provides information about the average 
                                        mistake the model made during predictions.
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            with col2:
                                st.write("")
                                st.write("")
                                img_stream = heatmap(filtered_reviews, "rating", f"predictions_{name}")
                                st.image(img_stream, caption="Heatmap of Predictions", use_container_width=True)


                            col1, col2 = st.columns([2, 3])
                            with col1:
                                st.write("**Rating Distribution**")
                                distrib_img_stream = distribiution_of_rating_for_app(filtered_reviews, f"predictions_{name}")
                                st.image(distrib_img_stream, caption="Rating Distribution")
                            with col2:
                                st.write("")
                                st.write("")
                                plot_stream = plot_monthly_avg_app(filtered_reviews, label=f"predictions_{name}")   
                                st.image(plot_stream, caption="Monthly Average Rating", use_container_width=True)
                            st.write("##### Word Clouds by Prediction")
                            try:
                                if "word_clouds_by_prediction" not in st.session_state:
                                    st.session_state.word_clouds_by_prediction = create_tfidf_wordcloud(filtered_reviews, rating_column=f"predictions_{name}")
                                records_per_prediction = filtered_reviews.groupby(f"predictions_{name}").size().to_dict()
                                word_clouds_available = {
                                    pred: st.session_state.word_clouds_by_prediction[pred] 
                                    for pred in st.session_state.word_clouds_by_prediction
                                    if pred in st.session_state.word_clouds_by_prediction
                                }
                                unique_predictions = sorted(word_clouds_available.keys())
                                if unique_predictions:
                                    cols = st.columns(len(unique_predictions))
                                    for i, pred in enumerate(unique_predictions):
                                        with cols[i]:
                                            st.write(f"{pred} Predictions ({records_per_prediction.get(pred, 0)} records)")
                                            st.image(word_clouds_available[pred], caption=f"{pred} Predictions")
                                else:
                                    st.warning("No word clouds available for the selected predictions.")
                            except Exception as e:
                                st.error(f"Error generating word clouds: {e}")

                        if model_info["model_name"]=="bert_sentiment_prediction":
                            col1, col2 = st.columns([3, 4])
                            results = calculate_metrics(filtered_reviews, "star_based_sentiment", f"predictions_{name}")
                            with col1:
                                st.write("##### Metrics:")
                                st.markdown(f"""
                                <div style="
                                    font-size: 17px; 
                                    font-weight: bold; 
                                    color: #F6B17A; 
                                    margin-bottom: 10px;">
                                    MAE: {results['MAE']}
                                </div>
                                """, unsafe_allow_html=True)
                                st.markdown(f"""
                                <div style="
                                    font-size: 17px; 
                                    font-weight: bold; 
                                    color: #F6B17A; 
                                    margin-bottom: 10px;">
                                    Average Accuracy: {results['Average Accuracy']}
                                </div>
                                """, unsafe_allow_html=True)
                                st.write(" ")
                                st.write("##### Metrics Per Label:")
                                metrics_df = results["Metrics Per Label"]
                                st.table(metrics_df)

                                st.markdown("""
                                <div style="
                                    background-color: #424769; 
                                    color: #E2E8F0; 
                                    padding: 13px; 
                                    border-radius: 10px; 
                                    font-family: 'sans-serif'; 
                                    line-height: 1.6; 
                                    margin-top: 10px;">
                                    <h2 style="color: #F6B17A; font-size: 20px; margin-bottom: 1px;">Metrics Explanation</h2>
                                    <p style="font-size: 16px; margin-bottom: 8px;">
                                        <strong>Accuracy:</strong> Accuracy is the simplest metric to evaluate any prediction. It is computed by dividing the 
                                        number of correct predictions by the number of all observations.
                                    </p>
                                    <p style="font-size: 16px; margin-bottom: 8px;">
                                        <strong>F-score:</strong> F-score is the harmonic mean between precision and recall. However, accuracy is unreliable in 
                                        the case of unbalanced data. The F-score punishes more for assigning all values to one class.
                                    </p>
                                    <p style="font-size: 16px; margin-bottom: 8px;">
                                        <strong>MAE (Mean Absolute Error):</strong> MAE is a typical regression metric that is interpretable and easy to understand by many 
                                        people. Despite performing classification, in the context of our data, this metric is still valuable because every class 
                                        can be interpreted as a value in the range of 1 to 5. The mean absolute error provides information about the average 
                                        mistake the model made during predictions.
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            with col2:
                                st.write("")
                                st.write("")
                                img_stream = heatmap(filtered_reviews, "star_based_sentiment", f"predictions_{name}")
                                st.image(img_stream, caption="Heatmap of Predictions", use_container_width=True)


                            col1, col2 = st.columns([2, 3])
                            with col1:
                                st.write("**Rating Distribution**")
                                distrib_img_stream = distribiution_of_rating_for_app(filtered_reviews, f"predictions_{name}")
                                st.image(distrib_img_stream, caption="Rating Distribution")
                            with col2:
                                st.write("")
                                st.write("")
                                plot_stream = plot_monthly_avg_app(filtered_reviews, label=f"predictions_{name}")   
                                st.image(plot_stream, caption="Monthly Average Rating", use_container_width=True)
                            st.write("##### Word Clouds by Prediction")
                            try:
                                if "word_clouds_by_prediction" not in st.session_state:
                                    st.session_state.word_clouds_by_prediction = create_tfidf_wordcloud(filtered_reviews, rating_column=f"predictions_{name}")
                                records_per_prediction = filtered_reviews.groupby(f"predictions_{name}").size().to_dict()
                                word_clouds_available = {
                                    pred: st.session_state.word_clouds_by_prediction[pred] 
                                    for pred in st.session_state.word_clouds_by_prediction
                                    if pred in st.session_state.word_clouds_by_prediction
                                }
                                unique_predictions = sorted(word_clouds_available.keys())
                                if unique_predictions:
                                    cols = st.columns(len(unique_predictions))
                                    for i, pred in enumerate(unique_predictions):
                                        with cols[i]:
                                            st.write(f"{pred} Predictions ({records_per_prediction.get(pred, 0)} records)")
                                            st.image(word_clouds_available[pred], caption=f"{pred} Predictions")
                                else:
                                    st.warning("No word clouds available for the selected predictions.")
                            except Exception as e:
                                st.error(f"Error generating word clouds: {e}")                            

            else: 
                reviews = st.session_state.df_external
            #st.dataframe(reviews.head(100), height=400)
            if st.session_state.predict_on_roberta_selected and st.session_state.predict_on_roberta_selected==True:
                st.write('#### Results for Model: RoBERTa')

                col1, col2 = st.columns([3, 4])
                results = calculate_metrics(reviews, "star_based_sentiment", "predictions_roberta")
                with col1:
                    st.write("##### Metrics:")
                    st.markdown(f"""
                    <div style="
                        font-size: 17px; 
                        font-weight: bold; 
                        color: #F6B17A; 
                        margin-bottom: 10px;">
                        MAE: {results['MAE']}
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style="
                        font-size: 17px; 
                        font-weight: bold; 
                        color: #F6B17A; 
                        margin-bottom: 10px;">
                        Average Accuracy: {results['Average Accuracy']}
                    </div>
                    """, unsafe_allow_html=True)
                    st.write(" ")
                    st.write("##### Metrics Per Label:")
                    metrics_df = results["Metrics Per Label"]
                    st.table(metrics_df)

                    st.markdown("""
                    <div style="
                        background-color: #424769; 
                        color: #E2E8F0; 
                        padding: 13px; 
                        border-radius: 10px; 
                        font-family: 'sans-serif'; 
                        line-height: 1.6; 
                        margin-top: 10px;">
                        <h2 style="color: #F6B17A; font-size: 20px; margin-bottom: 1px;">Metrics Explanation</h2>
                        <p style="font-size: 16px; margin-bottom: 8px;">
                            <strong>Accuracy:</strong> Accuracy is the simplest metric to evaluate any prediction. It is computed by dividing the 
                            number of correct predictions by the number of all observations.
                        </p>
                        <p style="font-size: 16px; margin-bottom: 8px;">
                            <strong>F-score:</strong> F-score is the harmonic mean between precision and recall. However, accuracy is unreliable in 
                            the case of unbalanced data. The F-score punishes more for assigning all values to one class.
                        </p>
                        <p style="font-size: 16px; margin-bottom: 8px;">
                            <strong>MAE (Mean Absolute Error):</strong> MAE is a typical regression metric that is interpretable and easy to understand by many 
                            people. Despite performing classification, in the context of our data, this metric is still valuable because every class 
                            can be interpreted as a value in the range of 1 to 5. The mean absolute error provides information about the average 
                            mistake the model made during predictions.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.write("")
                    st.write("")
                    img_stream = heatmap(reviews, "star_based_sentiment", "predictions_roberta")
                    st.image(img_stream, caption="Heatmap of Predictions", use_container_width=True)
                
                if not st.session_state.filtered_reviews.empty:
                    col1, col2 = st.columns([2, 3])
                    with col1:
                        st.write("**Rating Distribution**")
                        distrib_img_stream = distribiution_of_rating_for_app(reviews, "predictions_roberta")
                        st.image(distrib_img_stream, caption="Rating Distribution")
                    with col2:
                        plot_stream = plot_monthly_avg_app(reviews, label="predictions_roberta")   
                        st.image(plot_stream, caption="Monthly Average Rating", use_container_width=True)
                else:
                    col1, col2, col3 = st.columns([1, 4, 1])
                    with col1:
                         st.write(" ")
                    with col2:
                        st.write("**Rating Distribution**")
                        distrib_img_stream = distribiution_of_rating_for_app(reviews, "predictions_roberta")
                        st.image(distrib_img_stream, caption="Rating Distribution")
                    with col3:
                        st.write("")
                st.write("##### Word Clouds by Prediction")
                try:
                    if "word_clouds_by_prediction" not in st.session_state:
                        st.session_state.word_clouds_by_prediction = create_tfidf_wordcloud(reviews, rating_column="predictions_roberta")
                    records_per_prediction = reviews.groupby("predictions_roberta").size().to_dict()
                    word_clouds_available = {
                        pred: st.session_state.word_clouds_by_prediction[pred] 
                        for pred in st.session_state.word_clouds_by_prediction
                        if pred in st.session_state.word_clouds_by_prediction
                    }
                    unique_predictions = sorted(word_clouds_available.keys())
                    if unique_predictions:
                        cols = st.columns(len(unique_predictions))
                        for i, pred in enumerate(unique_predictions):
                            with cols[i]:
                                st.write(f"{pred} Predictions ({records_per_prediction.get(pred, 0)} records)")
                                st.image(word_clouds_available[pred], caption=f"{pred} Predictions")
                    else:
                        st.warning("No word clouds available for the selected predictions.")
                except Exception as e:
                    st.error(f"Error generating word clouds: {e}")
                
            
            if st.session_state.predict_on_vader_selected and st.session_state.predict_on_vader_selected==True:
                st.write('## Results for Model: VADER')

                col1, col2 = st.columns([3, 4])
                results = calculate_metrics(reviews, "star_based_sentiment", "predictions_vader")
                with col1:
                    st.write("##### Metrics:")
                    st.markdown(f"""
                    <div style="
                        font-size: 17px; 
                        font-weight: bold; 
                        color: #F6B17A; 
                        margin-bottom: 10px;">
                        MAE: {results['MAE']}
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style="
                        font-size: 17px; 
                        font-weight: bold; 
                        color: #F6B17A; 
                        margin-bottom: 10px;">
                        Average Accuracy: {results['Average Accuracy']}
                    </div>
                    """, unsafe_allow_html=True)
                    st.write(" ")
                    st.write("##### Metrics Per Label:")
                    metrics_df = results["Metrics Per Label"]
                    st.table(metrics_df)

                    st.markdown("""
                    <div style="
                        background-color: #424769; 
                        color: #E2E8F0; 
                        padding: 13px; 
                        border-radius: 10px; 
                        font-family: 'sans-serif'; 
                        line-height: 1.6; 
                        margin-top: 10px;">
                        <h2 style="color: #F6B17A; font-size: 20px; margin-bottom: 1px;">Metrics Explanation</h2>
                        <p style="font-size: 16px; margin-bottom: 8px;">
                            <strong>Accuracy:</strong> Accuracy is the simplest metric to evaluate any prediction. It is computed by dividing the 
                            number of correct predictions by the number of all observations.
                        </p>
                        <p style="font-size: 16px; margin-bottom: 8px;">
                            <strong>F-score:</strong> F-score is the harmonic mean between precision and recall. However, accuracy is unreliable in 
                            the case of unbalanced data. The F-score punishes more for assigning all values to one class.
                        </p>
                        <p style="font-size: 16px; margin-bottom: 8px;">
                            <strong>MAE (Mean Absolute Error):</strong> MAE is a typical regression metric that is interpretable and easy to understand by many 
                            people. Despite performing classification, in the context of our data, this metric is still valuable because every class 
                            can be interpreted as a value in the range of 1 to 5. The mean absolute error provides information about the average 
                            mistake the model made during predictions.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.write("")
                    st.write("")
                    img_stream = heatmap(reviews, "star_based_sentiment", 'predictions_vader')
                    st.image(img_stream, caption="Heatmap of Predictions", use_container_width=True)
                
                if not st.session_state.filtered_reviews.empty:
                    col1, col2 = st.columns([2, 3])
                    with col1:
                        st.write("**Rating Distribution**")
                        distrib_img_stream = distribiution_of_rating_for_app(reviews, "predictions_vader")
                        st.image(distrib_img_stream, caption="Rating Distribution")
                    with col2:
                        st.write("")
                        plot_stream = plot_monthly_avg_app(filtered_reviews, label="predictions_vader")   
                        st.image(plot_stream, caption="Monthly Average Rating", use_container_width=True)
                    
                else:
                    col1, col2, col3 = st.columns([1, 4, 1])
                    with col1:
                         st.write(" ")
                    with col2:
                        st.write("**Rating Distribution**")
                        distrib_img_stream = distribiution_of_rating_for_app(reviews, "predictions_vader")
                        st.image(distrib_img_stream, caption="Rating Distribution")
                    with col3:
                        st.write("")
                st.write("##### Word Clouds by Prediction")
                try:
                    if "word_clouds_by_prediction" not in st.session_state:
                        st.session_state.word_clouds_by_prediction = create_tfidf_wordcloud(reviews, rating_column="predictions_vader")
                    records_per_prediction = reviews.groupby("predictions_vader").size().to_dict()
                    word_clouds_available = {
                        pred: st.session_state.word_clouds_by_prediction[pred] 
                        for pred in st.session_state.word_clouds_by_prediction
                        if pred in st.session_state.word_clouds_by_prediction
                    }
                    unique_predictions = sorted(word_clouds_available.keys())
                    if unique_predictions:
                        cols = st.columns(len(unique_predictions))
                        for i, pred in enumerate(unique_predictions):
                            with cols[i]:
                                st.write(f"{pred} Predictions ({records_per_prediction.get(pred, 0)} records)")
                                st.image(word_clouds_available[pred], caption=f"{pred} Predictions")
                    else:
                        st.warning("No word clouds available for the selected predictions.")
                except Exception as e:
                    st.error(f"Error generating word clouds: {e}")

            # The pre-trained models        
            if "selected_model_path" in st.session_state and st.session_state.selected_available_model_name is not None:
                selected_model_path = st.session_state.selected_model_path
                name = st.session_state.selected_available_model_name
                st.write(f'## Results for Model: {name}')
                if "classification" in selected_model_path:
                    col1, col2 = st.columns([3, 4])

                    results = calculate_metrics(reviews, "rating", f"predictions_{name}")
                    
                    with col1:
                        st.write("##### Metrics:")
                        st.markdown(f"""
                        <div style="
                            font-size: 17px; 
                            font-weight: bold; 
                            color: #F6B17A; 
                            margin-bottom: 10px;">
                            MAE: {results['MAE']}
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"""
                        <div style="
                            font-size: 17px; 
                            font-weight: bold; 
                            color: #F6B17A; 
                            margin-bottom: 10px;">
                            Average Accuracy: {results['Average Accuracy']}
                        </div>
                        """, unsafe_allow_html=True)
                        st.write(" ")
                        st.write("##### Metrics Per Label:")
                        metrics_df = results["Metrics Per Label"]
                        st.table(metrics_df)

                        st.markdown("""
                        <div style="
                            background-color: #424769; 
                            color: #E2E8F0; 
                            padding: 13px; 
                            border-radius: 10px; 
                            font-family: 'sans-serif'; 
                            line-height: 1.6; 
                            margin-top: 10px;">
                            <h2 style="color: #F6B17A; font-size: 20px; margin-bottom: 1px;">Metrics Explanation</h2>
                            <p style="font-size: 16px; margin-bottom: 8px;">
                                <strong>Accuracy:</strong> Accuracy is the simplest metric to evaluate any prediction. It is computed by dividing the 
                                number of correct predictions by the number of all observations.
                            </p>
                            <p style="font-size: 16px; margin-bottom: 8px;">
                                <strong>F-score:</strong> F-score is the harmonic mean between precision and recall. However, accuracy is unreliable in 
                                the case of unbalanced data. The F-score punishes more for assigning all values to one class.
                            </p>
                            <p style="font-size: 16px; margin-bottom: 8px;">
                                <strong>MAE (Mean Absolute Error):</strong> MAE is a typical regression metric that is interpretable and easy to understand by many 
                                people. Despite performing classification, in the context of our data, this metric is still valuable because every class 
                                can be interpreted as a value in the range of 1 to 5. The mean absolute error provides information about the average 
                                mistake the model made during predictions.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.write("")
                        st.write("")
                        img_stream = heatmap(reviews, "rating", f"predictions_{name}")
                        st.image(img_stream, caption="Heatmap of Predictions", use_container_width=True)


                    col1, col2 = st.columns([2, 3])
                    with col1:
                        st.write("**Rating Distribution**")
                        distrib_img_stream = distribiution_of_rating_for_app(reviews, f"predictions_{name}")
                        st.image(distrib_img_stream, caption="Rating Distribution")
                    with col2:
                        st.write("")
                        plot_stream = plot_monthly_avg_app(reviews, label=f"predictions_{name}")   
                        st.image(plot_stream, caption="Monthly Average Rating", use_container_width=True)
                    st.write("#### Word Clouds by Prediction")
                    try:
                        if "word_clouds_by_prediction" not in st.session_state:
                            st.session_state.word_clouds_by_prediction = create_tfidf_wordcloud(reviews, rating_column=f"predictions_{name}")
                        records_per_prediction = reviews.groupby(f"predictions_{name}").size().to_dict()
                        word_clouds_available = {
                            pred: st.session_state.word_clouds_by_prediction[pred] 
                            for pred in st.session_state.word_clouds_by_prediction
                            if pred in st.session_state.word_clouds_by_prediction
                        }
                        unique_predictions = sorted(word_clouds_available.keys())
                        if unique_predictions:
                            cols = st.columns(len(unique_predictions))
                            for i, pred in enumerate(unique_predictions):
                                with cols[i]:
                                    st.write(f"{pred} Predictions ({records_per_prediction.get(pred, 0)} records)")
                                    st.image(word_clouds_available[pred], caption=f"{pred} Predictions")
                        else:
                            st.warning("No word clouds available for the selected predictions.")
                    except Exception as e:
                        st.error(f"Error generating word clouds: {e}")                    
                    

                elif "regression" in selected_model_path:
                    col1, col2 = st.columns([3, 4])
                    results = calculate_metrics(reviews, "rating", f"predictions_{name}")
                    with col1:
                        st.write("##### Metrics:")
                        st.markdown(f"""
                        <div style="
                            font-size: 17px; 
                            font-weight: bold; 
                            color: #F6B17A; 
                            margin-bottom: 10px;">
                            MAE: {results['MAE']}
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"""
                        <div style="
                            font-size: 17px; 
                            font-weight: bold; 
                            color: #F6B17A; 
                            margin-bottom: 10px;">
                            Average Accuracy: {results['Average Accuracy']}
                        </div>
                        """, unsafe_allow_html=True)
                        st.write(" ")
                        st.write("##### Metrics Per Label:")
                        metrics_df = results["Metrics Per Label"]
                        st.table(metrics_df)

                        st.markdown("""
                        <div style="
                            background-color: #424769; 
                            color: #E2E8F0; 
                            padding: 13px; 
                            border-radius: 10px; 
                            font-family: 'sans-serif'; 
                            line-height: 1.6; 
                            margin-top: 10px;">
                            <h2 style="color: #F6B17A; font-size: 20px; margin-bottom: 1px;">Metrics Explanation</h2>
                            <p style="font-size: 16px; margin-bottom: 8px;">
                                <strong>Accuracy:</strong> Accuracy is the simplest metric to evaluate any prediction. It is computed by dividing the 
                                number of correct predictions by the number of all observations.
                            </p>
                            <p style="font-size: 16px; margin-bottom: 8px;">
                                <strong>F-score:</strong> F-score is the harmonic mean between precision and recall. However, accuracy is unreliable in 
                                the case of unbalanced data. The F-score punishes more for assigning all values to one class.
                            </p>
                            <p style="font-size: 16px; margin-bottom: 8px;">
                                <strong>MAE (Mean Absolute Error):</strong> MAE is a typical regression metric that is interpretable and easy to understand by many 
                                people. Despite performing classification, in the context of our data, this metric is still valuable because every class 
                                can be interpreted as a value in the range of 1 to 5. The mean absolute error provides information about the average 
                                mistake the model made during predictions.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.write("")
                        st.write("")
                        img_stream = heatmap(reviews, "rating", f"predictions_{name}")
                        st.image(img_stream, caption="Heatmap of Predictions", use_container_width=True)


                    col1, col2 = st.columns([2, 3])
                    with col1:
                        st.write("**Rating Distribution**")
                        distrib_img_stream = distribiution_of_rating_for_app(reviews, f"predictions_{name}")
                        st.image(distrib_img_stream, caption="Rating Distribution")
                    with col2:
                        st.write("")
                        st.write("")
                        plot_stream = plot_monthly_avg_app(reviews, label=f"predictions_{name}")   
                        st.image(plot_stream, caption="Monthly Average Rating", use_container_width=True)
                    st.write("##### Word Clouds by Prediction")
                    try:
                        if "word_clouds_by_prediction" not in st.session_state:
                            st.session_state.word_clouds_by_prediction = create_tfidf_wordcloud(reviews, rating_column=f"predictions_{name}")
                        records_per_prediction = reviews.groupby(f"predictions_{name}").size().to_dict()
                        word_clouds_available = {
                            pred: st.session_state.word_clouds_by_prediction[pred] 
                            for pred in st.session_state.word_clouds_by_prediction
                            if pred in st.session_state.word_clouds_by_prediction
                        }
                        unique_predictions = sorted(word_clouds_available.keys())
                        if unique_predictions:
                            cols = st.columns(len(unique_predictions))
                            for i, pred in enumerate(unique_predictions):
                                with cols[i]:
                                    st.write(f"{pred} Predictions ({records_per_prediction.get(pred, 0)} records)")
                                    st.image(word_clouds_available[pred], caption=f"{pred} Predictions")
                        else:
                            st.warning("No word clouds available for the selected predictions.")
                    except Exception as e:
                        st.error(f"Error generating word clouds: {e}")

                    
                elif "sentiment_prediction" in selected_model_path:
                    col1, col2 = st.columns([3, 4])
                    results = calculate_metrics(filtered_reviews, "star_based_sentiment", f"predictions_{name}")
                    with col1:
                        st.write("##### Metrics:")
                        st.markdown(f"""
                        <div style="
                            font-size: 17px; 
                            font-weight: bold; 
                            color: #F6B17A; 
                            margin-bottom: 10px;">
                            MAE: {results['MAE']}
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"""
                        <div style="
                            font-size: 17px; 
                            font-weight: bold; 
                            color: #F6B17A; 
                            margin-bottom: 10px;">
                            Average Accuracy: {results['Average Accuracy']}
                        </div>
                        """, unsafe_allow_html=True)
                        st.write(" ")
                        st.write("##### Metrics Per Label:")
                        metrics_df = results["Metrics Per Label"]
                        st.table(metrics_df)

                        st.markdown("""
                        <div style="
                            background-color: #424769; 
                            color: #E2E8F0; 
                            padding: 13px; 
                            border-radius: 10px; 
                            font-family: 'sans-serif'; 
                            line-height: 1.6; 
                            margin-top: 10px;">
                            <h2 style="color: #F6B17A; font-size: 20px; margin-bottom: 1px;">Metrics Explanation</h2>
                            <p style="font-size: 16px; margin-bottom: 8px;">
                                <strong>Accuracy:</strong> Accuracy is the simplest metric to evaluate any prediction. It is computed by dividing the 
                                number of correct predictions by the number of all observations.
                            </p>
                            <p style="font-size: 16px; margin-bottom: 8px;">
                                <strong>F-score:</strong> F-score is the harmonic mean between precision and recall. However, accuracy is unreliable in 
                                the case of unbalanced data. The F-score punishes more for assigning all values to one class.
                            </p>
                            <p style="font-size: 16px; margin-bottom: 8px;">
                                <strong>MAE (Mean Absolute Error):</strong> MAE is a typical regression metric that is interpretable and easy to understand by many 
                                people. Despite performing classification, in the context of our data, this metric is still valuable because every class 
                                can be interpreted as a value in the range of 1 to 5. The mean absolute error provides information about the average 
                                mistake the model made during predictions.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.write("")
                        st.write("")
                        img_stream = heatmap(filtered_reviews, "star_based_sentiment", f"predictions_{name}")
                        st.image(img_stream, caption="Heatmap of Predictions", use_container_width=True)


                    col1, col2 = st.columns([2, 3])
                    with col1:
                        st.write("**Rating Distribution**")
                        distrib_img_stream = distribiution_of_rating_for_app(filtered_reviews, f"predictions_{name}")
                        st.image(distrib_img_stream, caption="Rating Distribution")
                    with col2:
                        st.write("")
                        st.write("")
                        plot_stream = plot_monthly_avg_app(filtered_reviews, label=f"predictions_{name}")   
                        st.image(plot_stream, caption="Monthly Average Rating", use_container_width=True)
                    st.write("##### Word Clouds by Prediction")
                    try:
                        if "word_clouds_by_prediction" not in st.session_state:
                            st.session_state.word_clouds_by_prediction = create_tfidf_wordcloud(filtered_reviews, rating_column=f"predictions_{name}")
                        records_per_prediction = filtered_reviews.groupby(f"predictions_{name}").size().to_dict()
                        word_clouds_available = {
                            pred: st.session_state.word_clouds_by_prediction[pred] 
                            for pred in st.session_state.word_clouds_by_prediction
                            if pred in st.session_state.word_clouds_by_prediction
                        }
                        unique_predictions = sorted(word_clouds_available.keys())
                        if unique_predictions:
                            cols = st.columns(len(unique_predictions))
                            for i, pred in enumerate(unique_predictions):
                                with cols[i]:
                                    st.write(f"{pred} Predictions ({records_per_prediction.get(pred, 0)} records)")
                                    st.image(word_clouds_available[pred], caption=f"{pred} Predictions")
                        else:
                            st.warning("No word clouds available for the selected predictions.")
                    except Exception as e:
                        st.error(f"Error generating word clouds: {e}")           
                    


    if (("rating_column" not in st.session_state or st.session_state.rating_column is None) and ("df_external" in st.session_state)):
        df_external = st.session_state.df_external
        if st.session_state.predict_on_vader_selected==True:
                st.write('#### Results for Model: VADER')
                st.write("")
                st.write("")
                st.write("")
                col1, col2, col3 = st.columns([1, 4, 1])
                with col1:
                    st.write("")
                with col2:
                    st.write("**Rating Distribution**")
                    distrib_img_stream = distribiution_of_rating_for_app(df_external, "predictions_vader")
                    st.image(distrib_img_stream, caption="Rating Distribution")
                with col3:
                    st.write("")
                st.write("##### Word Clouds by Prediction")
                try:
                    if "word_clouds_by_prediction" not in st.session_state:
                        st.session_state.word_clouds_by_prediction = create_tfidf_wordcloud(df_external, rating_column="predictions_vader")
                    records_per_prediction =df_external.groupby("predictions_vader").size().to_dict()
                    word_clouds_available = {
                        pred: st.session_state.word_clouds_by_prediction[pred] 
                        for pred in st.session_state.word_clouds_by_prediction
                        if pred in st.session_state.word_clouds_by_prediction
                    }
                    unique_predictions = sorted(word_clouds_available.keys())
                    if unique_predictions:
                        cols = st.columns(len(unique_predictions))
                        for i, pred in enumerate(unique_predictions):
                            with cols[i]:
                                st.write(f"{pred} Predictions ({records_per_prediction.get(pred, 0)} records)")
                                st.image(word_clouds_available[pred], caption=f"{pred} Predictions")
                    else:
                        st.warning("No word clouds available for the selected predictions.")
                except Exception as e:
                    st.error(f"Error generating word clouds: {e}")
        if st.session_state.predict_on_roberta_selected==True:
                st.write('#### Results for Model: RoBERTa')
                col1, col2, col3 = st.columns([1, 4, 1])
                with col1:
                    st.write("")
                with col2:
                    st.write("**Rating Distribution**")
                    distrib_img_stream = distribiution_of_rating_for_app(df_external, "predictions_roberta")
                    st.image(distrib_img_stream, caption="Rating Distribution")
                with col3:
                    st.write("")
                st.write("##### Word Clouds by Prediction")
                try:
                    if "word_clouds_by_prediction" not in st.session_state:
                        st.session_state.word_clouds_by_prediction = create_tfidf_wordcloud(df_external, rating_column="predictions_roberta")
                    records_per_prediction =df_external.groupby("predictions_roberta").size().to_dict()
                    word_clouds_available = {
                        pred: st.session_state.word_clouds_by_prediction[pred] 
                        for pred in st.session_state.word_clouds_by_prediction
                        if pred in st.session_state.word_clouds_by_prediction
                    }
                    unique_predictions = sorted(word_clouds_available.keys())
                    if unique_predictions:
                        cols = st.columns(len(unique_predictions))
                        for i, pred in enumerate(unique_predictions):
                            with cols[i]:
                                st.write(f"{pred} Predictions ({records_per_prediction.get(pred, 0)} records)")
                                st.image(word_clouds_available[pred], caption=f"{pred} Predictions")
                    else:
                        st.warning("No word clouds available for the selected predictions.")
                except Exception as e:
                    st.error(f"Error generating word clouds: {e}")
    
    if "filtered_reviews" in st.session_state or "df_external" not in st.session_state:
        if "filtered_reviews" in st.session_state:
            reviews = st.session_state.filtered_reviews
            if "df_external" not in st.session_state and reviews.empty:
                st.warning("No filtered data or model results available. Please go to 'Filter Products' page first and apply filters.")
        if 'filtered_reviews' not in st.session_state:
            st.warning("No filtered data or model results available. Please go to 'Filter Products' page first and apply filters.")
    

    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    if st.button("Back to Menu"):
        st.session_state.page = "Menu"
        st.rerun()
