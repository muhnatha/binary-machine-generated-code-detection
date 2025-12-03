import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler

# --- Import your RawCodeDataset class ---
try:
    from src.dataset import RawCodeDataset
except ImportError:
    from dataset import RawCodeDataset

# --- Configuration ---
OUTPUT_DIR = "output"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "model.pth")
METRICS_SAVE_PATH = os.path.join(OUTPUT_DIR, "metrics.json")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# --- UNIXCODER CONFIGURATION ---
MODEL_NAME = "microsoft/unixcoder-base"
MAX_LEN = 512            
SAFE_MAX_LEN = 256        

# --- OPTIMIZED SETTINGS ---
BATCH_SIZE = 2            
ACCUMULATION_STEPS = 8   
EPOCHS = 5              
PATIENCE = 3              
LEARNING_RATE = 1e-5      
CLASS_NAMES = ['Human-Written', 'Machine-Generated']

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# --- Preprocessing Functions ---
def remove_comments(code):
    """Remove comments from code."""
    if pd.isna(code) or code == '':
        return ''
    code = str(code)
    # Remove single-line comments (// and #)
    code = re.sub(r'//.*', '', code)
    code = re.sub(r'#.*', '', code)
    # Remove multi-line comments (/* */ and ''' ''')
    code = re.sub(r'/\*[\s\S]*?\*/', '', code)
    code = re.sub(r"'''[\s\S]*?'''", '', code)
    code = re.sub(r'"""[\s\S]*?"""', '', code)
    return code.strip()

def clean_code(code):
    if pd.isna(code) or code == '':
        return ''
    code = str(code)
    code = code.replace('\r\n', '\n').replace('\r', '\n')
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        cleaned_line = line.rstrip()
        cleaned_lines.append(cleaned_line)
    code = '\n'.join(cleaned_lines)
    while '\n\n\n' in code:
        code = code.replace('\n\n\n', '\n\n')
    return code.strip()

# --- Dataset Wrapper ---
class CodeDataset(Dataset):
    def __init__(self, raw_dataset, tokenizer, max_len):
        self.dataset = raw_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        code_text = str(item['code'])
        label = item['label']

        code_text = remove_comments(code_text)
        code_text = clean_code(code_text)

        encoding = self.tokenizer.encode_plus(
            code_text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            # CHANGED: Label must be float for BCE
            'labels': torch.tensor(label, dtype=torch.float)
        }

class UniXcoderClassifier(nn.Module):
    def __init__(self, base_model):
        super(UniXcoderClassifier, self).__init__()
        self.bert = base_model
        self.drop = nn.Dropout(p=0.3)
        hidden_size = self.bert.config.hidden_size 
        # CHANGED: Output dimension is 1 for Binary Classification
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        output = self.drop(pooled_output)
        return self.out(output)

# --- TRAINING FUNCTIONS ---
def train_epoch(model, data_loader, loss_fn, optimizer, scaler, device, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_targets = []
    
    fallback_len = SAFE_MAX_LEN 
    
    optimizer.zero_grad()

    for i, d in tqdm(enumerate(data_loader), total=len(data_loader), desc="Training", leave=False):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)

        try:
            with autocast(): 
                outputs = model(input_ids, attention_mask)
                
                # CHANGED: Flatten outputs and targets for BCE
                outputs = outputs.view(-1)
                
                # CHANGED: Prediction logic (Sigmoid + Threshold)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                loss = loss_fn(outputs, targets)
                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            loss_val = loss.item() * ACCUMULATION_STEPS 
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss_val)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        except torch.cuda.OutOfMemoryError:
            print(f"\n[WARNING] OOM detected. Retrying with truncation (Len: {fallback_len})...")
            
            del outputs, loss
            gc.collect()
            torch.cuda.empty_cache()
            
            input_ids_safe = input_ids[:, :fallback_len]
            attention_mask_safe = attention_mask[:, :fallback_len]

            with autocast():
                outputs = model(input_ids_safe, attention_mask_safe)
                
                # CHANGED: Flatten and Predict for OOM block as well
                outputs = outputs.view(-1)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                loss = loss_fn(outputs, targets)
                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item() * ACCUMULATION_STEPS)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    epoch_f1 = f1_score(all_targets, all_preds, average='binary') # Changed average to binary
    return correct_predictions.double() / n_examples, sum(losses) / len(losses), epoch_f1

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for d in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)

            with autocast():
                outputs = model(input_ids, attention_mask)
                
                # CHANGED: Flatten + Sigmoid + Threshold
                outputs = outputs.view(-1)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    epoch_f1 = f1_score(all_targets, all_preds, average='binary') # Changed average to binary
    return correct_predictions.double() / n_examples, sum(losses) / len(losses), epoch_f1

# --- INFERENCE HELPERS ---
def get_predictions(model, data_loader, device):
    model = model.eval()
    predictions = []
    real_values = []

    with torch.no_grad():
        for d in tqdm(data_loader, desc="Predicting", leave=False):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)
            
            with autocast():
                outputs = model(input_ids, attention_mask)
                # CHANGED: Flatten + Sigmoid + Threshold
                outputs = outputs.view(-1)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

            predictions.extend(preds.cpu())
            real_values.extend(targets.cpu())

    return torch.stack(predictions), torch.stack(real_values)

def save_confusion_matrix(y_true, y_pred, class_names, save_dir):
    try:
        if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
        cm = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        plt.figure(figsize=(8, 6))
        sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        plt.close()
    except Exception as e:
        print(f"Confusion Matrix plotting failed: {e}")

def save_plots_safe(history, save_dir):
    try:
        epochs = range(1, len(history['train_loss']) + 1)
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, history['train_loss'], label='Training Loss')
        plt.plot(epochs, history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # NOTE: Set this to False for the real run!
    USE_SUBSAMPLE = True 
    
    print(f"Loading datasets (Subsample={USE_SUBSAMPLE})...")
    train_raw = RawCodeDataset(split='train', subsample=USE_SUBSAMPLE, sample_size=5000)
    val_raw = RawCodeDataset(split='validation', subsample=USE_SUBSAMPLE, sample_size=2000)
    # Load Test Data for Final Inference
    test_raw = RawCodeDataset(split='test', subsample=False)
    
    print(f"Initializing Tokenizer ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_set = CodeDataset(train_raw, tokenizer, MAX_LEN)
    val_set = CodeDataset(val_raw, tokenizer, MAX_LEN)
    test_set = CodeDataset(test_raw, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    # Test Loader
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    print(f"Initializing Model ({MODEL_NAME})...")
    base_model = AutoModel.from_pretrained(MODEL_NAME)
    
    # CHANGED: Removed n_classes arg, model defaults to output dim 1 now
    model = UniXcoderClassifier(base_model)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # CHANGED: Using BCEWithLogitsLoss
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    scaler = GradScaler()

    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
    best_f1 = 0 
    patience_counter = 0

    print(f"Starting training for {EPOCHS} epochs...")
    print(f"Effective Batch Size: {BATCH_SIZE * ACCUMULATION_STEPS}")
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        
        train_acc, train_loss, train_f1 = train_epoch(model, train_loader, loss_fn, optimizer, scaler, device, len(train_set))
        val_acc, val_loss, val_f1 = eval_model(model, val_loader, loss_fn, device, len(val_set))

        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['train_f1'].append(float(train_f1))
        history['val_f1'].append(float(val_f1))

        print(f"Train | Loss: {train_loss:.4f} | F1: {train_f1:.4f}")
        print(f"Val   | Loss: {val_loss:.4f} | F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            best_f1 = val_f1
            patience_counter = 0
            print(f"New best model saved to {MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    with open(METRICS_SAVE_PATH, 'w') as f:
        json.dump(history, f, indent=4)
    save_plots_safe(history, FIGURES_DIR)
    
    # --- FINAL INFERENCE ON TEST SET ---
    print("\nRunning final evaluation on Test Set...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    y_pred, y_test = get_predictions(model, test_loader, device)

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    save_confusion_matrix(y_test, y_pred, CLASS_NAMES, FIGURES_DIR)

    print("Pipeline complete.")

if __name__ == "__main__":
    main()