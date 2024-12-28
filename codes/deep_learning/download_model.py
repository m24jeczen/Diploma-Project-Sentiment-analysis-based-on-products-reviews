from transformers import AutoTokenizer, AutoModelForSequenceClassification
from codes.loading_and_filtering.parameters import device
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from codes.loading_and_filtering.parameters import roberta_model, bert_model_name, available_models
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch

# Function for downloading twitter model from hugging face. Can be used for other models but need to be tested.
# model_name is for hugging face system downloading. model_local_path is for loading them later from local files.

def download_and_save_twitter_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(roberta_model.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(roberta_model.model_name)
        model.save_pretrained(roberta_model.local_path)
        tokenizer.save_pretrained(roberta_model.local_path)
        available_models.append(roberta_model)
        print(f"Model {roberta_model.model_name} downloaded and saved.")
    except:
            print(f"Can't download {roberta_model.model_name}.")
            return 
            
# Function depends on added label tune to predict it and save it locally(genereally only for stars and sentiment 0 or 1)
def download_and_tune_bert(model_local_path, data, label = "rating"):
    # Handling errors of wrong input data
    try:
         data=data[data["text"].str.split().str.len() >= 1]
         texts = list(data.text)
    except:
         return "Invalid text data"
    try:
        if label == "rating":
            number_of_labels = 5
            labels = [int(x)-1 for x in data.rating]
        else:
             labels = data[label]
             number_of_labels = 2
    except:
         return "Invalid data"

    # Train val splitting
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    # Tokenizing texts
    train_encodings = tokenizer(train_texts, padding="max_length", truncation=True, max_length=128)
    val_encodings = tokenizer(val_texts, padding="max_length", truncation=True, max_length=128)

    def prepare_dataset(encodings, labels):
        return Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        })
    # Needed in case of entering ratings out of range 1-5
    try:
        train_dataset = prepare_dataset(train_encodings, train_labels)
        val_dataset = prepare_dataset(val_encodings, val_labels)

        model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=number_of_labels)
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
        available_models.append()

    except:
        return "Wrong parameters or input data"
        
