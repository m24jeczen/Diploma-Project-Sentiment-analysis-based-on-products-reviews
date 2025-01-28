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

2. Alternatively, run the following command in your terminal:
```
streamlit run .\codes\GUI\app.py --server.runOnSave=true
```
