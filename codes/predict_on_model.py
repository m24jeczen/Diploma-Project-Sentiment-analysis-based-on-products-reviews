from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.sentiment import SentimentIntensityAnalyzer
from codes.parameters import device, roberta_model
from torch.utils.data import DataLoader, Dataset
import torch, nltk
import numpy as np

class TextDataset(Dataset):
    # Dataset for text input data.
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {key: val.squeeze(0) for key, val in inputs.items()}
    

def predict_on_local_model(data, local_path, batch_size=128):
    try:
        # Initialize the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        model = AutoModelForSequenceClassification.from_pretrained(local_path)
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    model.to(device)
    model.eval()

    # Create a dataset and DataLoader
    dataset = TextDataset(data.text.tolist(), tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []
    for batch in dataloader:
        # Move the batch to the device
        inputs = {key: val.to(device) for key, val in batch.items()}

        # Perform prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        predictions.extend(logits.tolist())

    return predictions


# Function to predict on twitter model. There are fixed 3 labels which we would not change
def predict_on_roberta(data):
    predictions = predict_on_local_model(data, roberta_model.local_path)
    mapped_predictions = [
        1 if logit[2] > logit[0] else 0 
        for logit in predictions
    ]
    return mapped_predictions


# Function to predict starts for data on saved model. Will be changed to accept any saved loccaly model
def predict_stars(data, model_local_path):
    logits = predict_on_local_model(data, model_local_path)
    return np.argmax(logits,axis = 1) +1

# Function to predict sentiment on trained bert model
def predict_sentiment_bert(data, model_local_path):
    logits = predict_on_local_model(data, model_local_path)
    return np.argmax(logits,axis = 1)

def predict_on_vader(data):

    # Make sure to download the VADER lexicon if not already downloaded
    if not nltk.downloader.Downloader().is_installed('vader_lexicon'):
        nltk.download('vader_lexicon')

    sia = SentimentIntensityAnalyzer()
    # Apply sentiment analysis to the dataset
    data['sentiment'] = data['text'].apply(lambda x:1 if sia.polarity_scores(x)['compound']>=0 else 0)

    return data