import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertModel, BertConfig
from torch.optim import AdamW
from codes.parameters import bert_model, device

# Dataset
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
    all_predictions = []
    all_labels = []

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
    

# Model
class LocalBert(nn.Module):
    def __init__(self, task, num_classes=None):
        super(LocalBert, self).__init__()
        self.task = task
        self.bert = BertModel.from_pretrained(bert_model.local_path)
        self.dropout = nn.Dropout(0.3)
        if task == "classification":
            self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        elif task == "regression":
            self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        if self.task == "classification":
            return self.classifier(pooled_output)
        elif self.task == "regression":
            return self.regressor(pooled_output)

# Training function
def train_model(dataframe, task = "classification",label="rating", num_classes=None, max_epochs=3, batch_size=16, lr=2e-5, max_len=128, val_split=0.2):
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

    # Initialize model
    model = LocalBert(task, num_classes=num_classes).to(device)

    # Define optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=lr)
    if task == "classification":
        criterion = nn.CrossEntropyLoss()
    elif task == "regression":
        criterion = nn.MSELoss()

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
            
            if task == "classification":
                loss = criterion(outputs, labels)
            elif task == "regression":
                outputs = outputs.squeeze(-1)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # Evaluate on validation set
        val_loss, metric = evaluate_model(model, val_loader, task, criterion, device)

        print(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {metric:.4f}")
        



