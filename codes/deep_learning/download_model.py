from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertModel, BertTokenizer
from codes.loading_and_filtering.parameters import device, roberta_model, bert_model
from datasets import Dataset
import torch, os
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW

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
            # logits = self.regressor(pooled_output)
            # return torch.clamp(logits,min = 1, max=5)
            return self.regressor(pooled_output)
        
        # Function for saving models in local folders
    def save_model(self, path):
        self.bert.save_pretrained(path)
        torch.save(self.state_dict(), os.path.join(path, "model_state.pth"))

        # Function for loading tuned models
    @staticmethod
    def load_model(task, num_classes, path):
        """Load a model from the specified path."""
        model = BertForTask(task, num_classes)
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
                predictions = torch.round(outputs)  # Apply scaling and rounding
                predictions = torch.clamp(predictions, min = 1, max = 5)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            total_loss += loss.item()

    accuracy = correct / total
    return total_loss / len(dataloader), accuracy
    

# Training function
def train_model(dataframe, task = "classification", target="rating",num_classes=5, max_epochs=3, batch_size=16, lr=2e-5, max_len=128, val_split=0.2, localname = None, early_stopping = True, patience = 3, dropout_rate = 0):
    if task == "regression":
        dataframe["label"] = [float(x) for x in dataframe[target]]
    else:
        dataframe["label"] = dataframe[target]
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

    # Initialize model and assign loss function depend on choosen regression or classification approach
    model = BertForTask(task, num_classes=num_classes, dropout_rate = dropout_rate).to(device)
    criterion = model.criterion
    optimizer = AdamW(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

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
        
        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()  # Save the current best state
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epochs. Patience: {patience}")

            # Stop training if patience is exceeded
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load the best model state into the model
    if best_model_state:
        model.load_state_dict(best_model_state)
    folder = "models"
    # path for star prediction and (classification and regression approaches)
    if num_classes == 5:
        # for star prediction
        folder = os.path.join(folder, task)
    else:
        # for sentiment prediction
        folder = os.path.join(folder, "sentiment_prediction")
    
    if localname == None:
        current_time =datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(folder, "bert_trained_at_"+current_time)
    else:
        path = os.path.join(folder, localname)

    model.save_model(path)
    tokenizer.save_pretrained(path)
        

def get_available_models(task_type, parent_dir="models"):

    models_dir = f"./{parent_dir}"
    available_models = {}

    for root, dirs, files in os.walk(models_dir):
        if "config.json" in files:
            relative_path = os.path.relpath(root, models_dir)
            model_name = os.path.basename(root)
            # Check if the model is in the desired task type directory
            path_parts = relative_path.split(os.sep)
            if task_type in path_parts and model_name != "bert-base-uncased":
                available_models[model_name] = os.path.join(models_dir, relative_path)

    return available_models

