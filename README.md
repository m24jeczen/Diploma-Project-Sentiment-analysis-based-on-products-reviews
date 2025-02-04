# Sentiment analysis based on reviews for products from online stores

This is our Bachelor's diploma thesis project.

## Project describtion

## How to launch and close the app:
### 0. Running the app for the first time 
if you have access to a CUDA-capable graphics card we recommend install CUDA and use it for faster models training. Instructions:
https://pytorch.org/get-started/locally/

Be sure that python satisfy < 3.12 (we recomend version 3.11.9) and to run comands:
```
pip install --upgrade pip
pip install -r requirements.txt
```
Some packages could not install properly. Application can show errors with particular one. In this case try for example:
```
pip install pyLDAvis
```
or
```
pip install wordcloud
```

To set up the necessary models before launching the app, run the following commands in your terminal:

```python
python
from codes.deep_learning.download_model import download_and_save_hugging_face_models
download_and_save_hugging_face_models()

import spacy
from spacy.cli import download

download("en_core_web_sm")

import nltk
nltk.download('wordnet')
```
### 1. Running the app
There are 2 ways to run the app:

1. Call the `main()` function directly by running:
```bash
python main.py
```

2. Alternatively, run the following command in your terminal:
```bash
streamlit run .\codes\GUI\app.py --server.runOnSave=true
```

### 2. Stopping the app
To stop the app, simply press `Ctrl + C` in your console or terminal.

## Repository contents
### .streamlit:

In this directory, there is the main app theme implementation

### codes:
#### GUI:
- `app.py`: here is the implementation of GUI, done with streamlit
#### deep_learning:
- `download_model.py`:
- `predict_on_model.py`:
- `preprocessing.py`:
- `rating_analysis.py`:
#### loading_and_filtering:
- `data_loader.py`:
- `filter.py`:
- `filter_test.py`:
- `parameters.py`:
- `prepare_category.py`:
#### topic_modeling:
- `LDA.py`:
- `text_preprocessing.py`:
#### output_data:
- `sephora_1000.csv`:
#### visualizations.py:




## Authors:
- [Magdalena Jeczeń](https://github.com/m24jeczen)  
- [Michal Iwicki](https://github.com/Michal-Iwicki)
