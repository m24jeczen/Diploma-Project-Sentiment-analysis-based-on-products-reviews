# Sentiment analysis based on reviews for products from online stores

This is our Bachelor's diploma thesis project.

## Project describtion

## Code describtion
### 0. Running the app for the first time 

To set up the necessary models before launching the app, run the following commands in your terminal:

```python
python
from codes.deep_learning.download_model import download_and_save_hugging_face_models
download_and_save_hugging_face_models()
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
#### .streamlit:

In this directory, there is the main app theme implementation

#### codes:
##### GUI:
- `app.py`: here is the implementation of GUI, done with streamlit
##### deep_learning:
- `download_model.py`:
- `predict_on_model.py`:
- `preprocessing.py`:
- `rating_analysis.py`:
##### loading_and_filtering:
##### topic_modeling:
##### visualizations.py:




## Authors:
- [Magdalena Jeczeń](https://github.com/m24jeczen)  
- [Michal Iwicki](https://github.com/Michal-Iwicki)
