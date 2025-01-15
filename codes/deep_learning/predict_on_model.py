from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.sentiment import SentimentIntensityAnalyzer
from codes.loading_and_filtering.parameters import device, roberta_model
from codes.deep_learning.download_model import BertForTask
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
    

def predict_on_tuned_model(data, local_path, batch_size=256):
    # Checking if choosen model is for star prediction or sentiment analysis
    if "sentiment_prediction" in local_path:
        num_classes = 2
    else:
        num_classes = 5

    # Checking if model was trained to do regression or classification
    if "regression" in local_path:
        task = "regression"
    else:
        task = "classification"

    try:
        # Initialize the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        model = BertForTask.load_model(task,num_classes,local_path)
    except Exception as e:
        print(f"Error loading the model: {e}")
        return
    texts = list(data.text)
    all_predictions = []
    model.to(device)
    model.eval()
    dataset = TextDataset(
        texts=texts,
        tokenizer=tokenizer
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in dataloader:
            # Move input data to the same device as the model
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            if task == "classification":
                predictions = torch.argmax(outputs, dim=1)  # Get class indices
            elif task == "regression":
                predictions = torch.round(outputs.squeeze(-1))  # Flatten for regression and clamping border results
                predictions = torch.clamp(predictions, min = 1, max = 5)

            all_predictions.extend(predictions.cpu().numpy())  # Move predictions to CPU for further processing

    return np.array(all_predictions)


# Function to predict on twitter model. There are fixed 3 labels but we care about only positive and negative
def predict_on_roberta(data, batch_size = 256):
    try:
        # Initialize the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(roberta_model.local_path)
        model = AutoModelForSequenceClassification.from_pretrained(roberta_model.local_path)
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

    mapped_predictions = [
        1 if logit[2] > logit[0] else 0 
        for logit in predictions
    ]
    return mapped_predictions


# Function to predict starts for data on saved model. Will be changed to accept any saved loccaly model
def predict_stars(data, model_local_path):
    predictions = predict_on_tuned_model(data, model_local_path)
    return predictions +1

# Function to predict sentiment on trained bert model
def predict_sentiment_bert(data, model_local_path):
    predictions = predict_on_tuned_model(data, model_local_path)
    return predictions

def predict_on_vader(data):

    # Make sure to download the VADER lexicon if not already downloaded
    if not nltk.downloader.Downloader().is_installed('vader_lexicon'):
        nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    # Apply sentiment analysis to the dataset
    return data['text'].apply(lambda x:1 if sia.polarity_scores(x)['compound']>=0 else 0)
