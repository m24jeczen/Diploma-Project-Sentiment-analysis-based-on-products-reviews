from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertModel, BertTokenizer
from codes.parameters import device
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from codes.parameters import roberta_model, bert_model
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch, os
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from codes.parameters import bert_model, device

def download_and_save_hugging_face_models():
    try:
        tokenizer = AutoTokenizer.from_pretrained(roberta_model.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(roberta_model.model_name)
        model.save_pretrained(roberta_model.local_path)
        tokenizer.save_pretrained(roberta_model.local_path)
        print(f"Model {roberta_model.model_name} downloaded and saved.")
    except:
            print(f"Can't download {roberta_model.model_name}.")
            return 
    try:
        tokenizer = BertTokenizer.from_pretrained(bert_model.model_name)
        model = BertModel.from_pretrained(bert_model.model_name)
        model.save_pretrained(bert_model.local_path)
        tokenizer.save_pretrained(bert_model.local_path)
        print(f"Model {bert_model.model_name} downloaded and saved.")
    except:
            print(f"Can't download {bert_model.model_name}.")
            return         


class BertForTask(nn.Module):
    def __init__(self, task, num_classes, dropout_rate = 0):
        super(BertForTask, self).__init__()
        self.task = task
        self.bert = BertModel.from_pretrained(bert_model.local_path)
        self.dropout = nn.Dropout(dropout_rate)

        if task == "classification":
            self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
            self.criterion = nn.CrossEntropyLoss()
        elif task == "regression":
            self.regressor = nn.Linear(self.bert.config.hidden_size, 1)
            self.criterion = nn.MSELoss()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        if self.task == "classification":
            return self.classifier(pooled_output)
        elif self.task == "regression":
            return self.regressor(pooled_output)
        
    def save_model(self, path):
        """Save the model to the specified path."""
        self.bert.save_pretrained(path)
        torch.save(self.state_dict(), os.path.join(path, "model_state.pth"))

    @staticmethod
    def load_model(task, num_classes, path, dropout_rate=0):
        """Load a model from the specified path."""
        model = BertForTask(task, num_classes, dropout_rate)
        model.bert = BertModel.from_pretrained(path)
        model.load_state_dict(torch.load(os.path.join(path, "model_state.pth")))
        return model
        
        
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float if isinstance(label, float) else torch.long)
        }
    
def evaluate_model(model, dataloader, task, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            if task == "regression":
                ratings = torch.round(labels * 4 + 1)

            with torch.amp.autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            if task == "classification":
                loss = criterion(outputs, labels)
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
            elif task == "regression":
                outputs = outputs.squeeze(-1)
                loss = criterion(outputs, labels)
                predictions = torch.round(outputs * 4 + 1)  # Apply scaling and rounding
                correct += (predictions == ratings).sum().item()
                total += labels.size(0)

            total_loss += loss.item()

    accuracy = correct / total
    return total_loss / len(dataloader), accuracy
    

# Training function
def train_model(dataframe, task = "classification",num_classes=5, max_epochs=3, batch_size=16, lr=2e-5, max_len=128, val_split=0.2, localname = None):
    tokenizer = BertTokenizer.from_pretrained(bert_model.local_path)
    dataset = TextDataset(
        texts=dataframe['text'].tolist(),
        labels=dataframe['label'].tolist(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model and assign loss function depend on choosen regression or classification
    model = BertForTask(task, num_classes=num_classes).to(device)
    criterion = model.criterion
    optimizer = AdamW(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            if task == "regression":
                outputs = outputs.squeeze(-1)
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        val_loss, metric = evaluate_model(model, val_loader, task, criterion, device)

        print(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {metric:.4f}")
        if localname == None:
            current_time =datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = os.path.join("models", "bert_"+task+"_trained_at_"+current_time)
        else:
            path = os.path.join("models", localname)

    model.save_model(path)
    tokenizer.save_pretrained(path)
        















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

    tokenizer = BertTokenizer.from_pretrained(bert_model.local_path)

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

        model = BertForSequenceClassification.from_pretrained(bert_model.local_path, num_labels=number_of_labels)
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

    except:
        return "Wrong parameters or input data"
        
