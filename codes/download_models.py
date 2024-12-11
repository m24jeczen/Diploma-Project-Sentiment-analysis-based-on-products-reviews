from transformers import AutoTokenizer, AutoModelForSequenceClassification
from codes.parameters import device
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch

def download_and_save_twitter_model(model_name, model_local_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.save_pretrained(model_local_path)
        tokenizer.save_pretrained(model_local_path)
        print(f"Model {model_name} downloaded and saved.")
    except ValueError:
            print(f"Can't download and save {model_name}.")
            return 
            

def download_and_tune_bert(model_name, model_local_path, data, min_length = 1):
    try:
         data=data[data["text"].str.split().str.len() >= min_length]
         texts = list(data.text)
    except:
         return "Invalid text data"
    
    try:
         labels = [int(x)-1 for x in data.rating]
    except:
         return "Invalid rating data"
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_encodings = tokenizer(train_texts, padding="max_length", truncation=True, max_length=128)
    val_encodings = tokenizer(val_texts, padding="max_length", truncation=True, max_length=128)


    def prepare_dataset(encodings, labels):
        return Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        })

    train_dataset = prepare_dataset(train_encodings, train_labels)
    val_dataset = prepare_dataset(val_encodings, val_labels)

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5)
    model.to(device) 


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=1).to(device)
        accuracy = (predictions == torch.tensor(labels).to(device)).float().mean().item()
        return {"accuracy": accuracy}

    training_args = TrainingArguments(
        output_dir="../results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir="../logs",
        logging_steps=10,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )


    trainer.train()

    results = trainer.evaluate()

    trainer.save_model(model_local_path)
        
