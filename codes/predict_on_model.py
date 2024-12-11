from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertForSequenceClassification
from codes.parameters import device, roberta_model,bert_model
import torch

def predict_on_roberta(data):

    tokenizer = AutoTokenizer.from_pretrained(roberta_model.local_path)
    model = AutoModelForSequenceClassification.from_pretrained(roberta_model.local_path)
    try:
        inputs = tokenizer(list(data.text), padding=True, truncation=True, return_tensors="pt", max_length=128)
    except:
        print("Invalid input data")
        return
    model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

    label_map = {0: "negative", 1: "neutral", 2: "positive"}

    mapped_predictions = [label_map[label] for label in predictions.tolist()]

    return mapped_predictions

def predict_on_bert(data):
    tokenizer = BertTokenizer.from_pretrained(bert_model.local_path)
    model = BertForSequenceClassification.from_pretrained(bert_model.local_path)
    model.to(device)
    try:
        inputs = tokenizer(list(data.text), padding=True, truncation=True, return_tensors="pt", max_length=128)
    except:
        print("Invalid input data")
    inputs = {key: val.to(device) for key, val in inputs.items()}

    model.eval()  
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

    return [x+1 for x in predictions.tolist()]

