from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertForSequenceClassification
from nltk.sentiment import SentimentIntensityAnalyzer
from codes.parameters import device, roberta_model,bert_model
import torch, nltk


# Function to predict on twitter model. There are fixed 3 labels which we would not change
def predict_on_roberta(data):
    try:
        tokenizer = AutoTokenizer.from_pretrained(roberta_model.local_path)
        model = AutoModelForSequenceClassification.from_pretrained(roberta_model.local_path)
        inputs = tokenizer(list(data.text), padding=True, truncation=True, return_tensors="pt", max_length=128)
    except:
        print("Invalid input data or lack of model")
        return
    model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    mapped_predictions = [
        "positive" if logit[2] > logit[0] else "negative" 
        for logit in logits
    ]

    return mapped_predictions

# Function to predict starts for data on saved model. Will be changed to accept any saved loccaly model
def predict_on_bert(data):
    try:
        tokenizer = AutoTokenizer.from_pretrained(bert_model.local_path)
        model = AutoModelForSequenceClassification.from_pretrained(bert_model.local_path)
        model.to(device)
        inputs = tokenizer(list(data.text), padding=True, truncation=True, return_tensors="pt", max_length=128)
    except:
        print("Invalid input data or lack of model")
    inputs = {key: val.to(device) for key, val in inputs.items()}

    model.eval()  
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

    return [x+1 for x in predictions.tolist()]

def predict_on_vader(data):

    # Make sure to download the VADER lexicon if not already downloaded
    if not nltk.downloader.Downloader().is_installed('vader_lexicon'):
        nltk.download('vader_lexicon')

    sia = SentimentIntensityAnalyzer()
    # Apply sentiment analysis to the dataset
    data['sentiment'] = data['text'].apply(lambda x:1 if sia.polarity_scores(x)['compound']>=0 else 0)

    return data