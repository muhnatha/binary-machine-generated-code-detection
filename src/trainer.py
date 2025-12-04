import os
# --- MEMORY OPTIMIZATION CONFIGURATION (Must be before torch import) ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
import torch.nn as nn
import numpy as np
import gc
import re
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
from collections import Counter 

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

# --- OPTIMIZED SETTINGS ---
BATCH_SIZE = 32            
ACCUMULATION_STEPS = 1   
EPOCHS = 5           
PATIENCE = 3              
LEARNING_RATE = 5e-5      
CLASS_NAMES = ['Human-Written', 'Machine-Generated']

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# --- Preprocessing Functions ---

def clean_code_strict(code):
    if pd.isna(code) or code == '':
        return ''
    code = str(code)
    code = re.sub(r'<[^>]+>', '', code)
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'#.*', '', code)
    code = re.sub(r'//.*', '', code)
    code = code.replace('\r\n', '\n').replace('\r', '\n')
    lines = [line.rstrip() for line in code.split('\n') if line.strip()]
    return '\n'.join(lines)

def filter_dataset_codet_m4(raw_dataset, tokenizer):
    print("Running CoDet-M4 Quality Assurance Pipeline...")
    seen_hashes = set()
    token_counts = []
    temp_list = []
    
    range_iter = range(len(raw_dataset))
    
    for i in tqdm(range_iter, desc="Cleaning & Deduplicating"):
        item = raw_dataset[i]
        original_code = item['code']
        label = item['label']
        language = item['language'] 
        
        processed_code = clean_code_strict(original_code)
        
        if not processed_code.strip():
            continue

        code_hash = hash(processed_code)
        if code_hash in seen_hashes:
            continue
        seen_hashes.add(code_hash)
        
        tokens = tokenizer.tokenize(processed_code)
        token_len = len(tokens)
        
        token_counts.append(token_len)
        temp_list.append({
            'code': processed_code,
            'label': label,
            'language': language, 
            'length': token_len
        })
        
    if not token_counts:
        print("Warning: Dataset empty after cleaning!")
        return []

    p5 = np.percentile(token_counts, 5)
    p95 = np.percentile(token_counts, 95)
    
    final_dataset = []
    for item in temp_list:
        if p5 <= item['length'] <= p95:
            final_dataset.append(item)
            
    print(f"Final Size: {len(final_dataset)}")
    return final_dataset

def compute_language_weights(dataset_list):
    languages = [item['language'] for item in dataset_list]
    count = Counter(languages)
    total_samples = len(languages)
    n_languages = len(count)
    
    weights = {}
    print("\n--- Language Weights ---")
    for lang, freq in count.items():
        w = total_samples / (n_languages * freq)
        weights[lang] = w
        print(f"{lang}: {freq} samples -> Weight: {w:.4f}")
    
    return weights

# --- Dataset Wrapper ---
class CodeDataset(Dataset):
    def __init__(self, data_source, tokenizer, max_len, lang_weights=None):
        self.data_source = data_source 
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.lang_weights = lang_weights

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, index):
        item = self.data_source[index]
        code_text = str(item['code'])
        label = item['label']
        language = item['language']

        # Determine weight
        weight = 1.0
        if self.lang_weights:
            weight = self.lang_weights.get(language, 1.0)

        encoding = self.tokenizer.encode_plus(
            code_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float),
            'sample_weight': torch.tensor(weight, dtype=torch.float)
        }

class UniXcoderClassifier(nn.Module):
    def __init__(self, base_model):
        super(UniXcoderClassifier, self).__init__()
        self.bert = base_model
        self.drop = nn.Dropout(p=0.3)
        hidden_size = self.bert.config.hidden_size 
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        output = self.drop(pooled_output)
        return self.out(output)

# --- ROBUST TRAINING FUNCTION ---
def train_epoch(model, data_loader, loss_fn, optimizer, scaler, device, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_targets = []
    
    optimizer.zero_grad()

    # Define ladder of fallback lengths
    fallback_ladder = [512, 256, 128]

    for i, d in tqdm(enumerate(data_loader), total=len(data_loader), desc="Training", leave=False):
        input_ids_full = d["input_ids"].to(device)
        attention_mask_full = d["attention_mask"].to(device)
        targets = d["labels"].to(device)
        sample_weights = d["sample_weight"].to(device)
        
        success = False
        
        for seq_len in fallback_ladder:
            try:
                # Slice input to current attempt length
                input_ids = input_ids_full[:, :seq_len]
                attention_mask = attention_mask_full[:, :seq_len]

                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids, attention_mask)
                    outputs = outputs.view(-1)
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()
                    
                    # Weighted Loss
                    loss_unreduced = loss_fn(outputs, targets) 
                    loss_weighted = loss_unreduced * sample_weights
                    loss = loss_weighted.mean()
                    
                    loss = loss / ACCUMULATION_STEPS

                scaler.scale(loss).backward()
                
                # Check for NaNs
                if torch.isnan(loss):
                     raise ValueError("Loss is NaN")

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
                
                success = True
                break # Break out of ladder loop if successful

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                # If OOM, clear cache and try next length in ladder
                if "out of memory" in str(e):
                    if seq_len == fallback_ladder[-1]:
                        print(f"\n[CRITICAL] Skipping batch {i} - OOM even at length {seq_len}")
                    else:
                        # Clean up variable references before GC
                        if 'outputs' in locals(): del outputs
                        if 'loss' in locals(): del loss
                        gc.collect()
                        torch.cuda.empty_cache()
                        continue # Try next length
                else:
                    raise e # Re-raise other errors
        
        if not success:
            # If we failed all ladder steps, we must still clean memory
            gc.collect()
            torch.cuda.empty_cache()
            optimizer.zero_grad() # Reset optimizer for next batch

    epoch_f1 = f1_score(all_targets, all_preds, average='binary')
    return correct_predictions.double() / n_examples, sum(losses) / (len(losses) + 1e-9), epoch_f1

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
            
            # For validation, if we OOM, we just crash currently. 
            # Usually Val uses less memory (no gradients).
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids, attention_mask)
                outputs = outputs.view(-1)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                loss_unreduced = loss_fn(outputs, targets)
                loss = loss_unreduced.mean() 

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    epoch_f1 = f1_score(all_targets, all_preds, average='binary')
    return correct_predictions.double() / n_examples, sum(losses) / (len(losses) + 1e-9), epoch_f1

def get_predictions(model, data_loader, device):
    model = model.eval()
    predictions = []
    real_values = []

    with torch.no_grad():
        for d in tqdm(data_loader, desc="Predicting", leave=False):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids, attention_mask)
                outputs = outputs.view(-1)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

            predictions.extend(preds.cpu())
            real_values.extend(targets.cpu())

    return torch.stack(predictions), torch.stack(real_values)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    USE_SUBSAMPLE = True 
    
    print(f"Loading tokenizer ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print(f"Loading datasets (Subsample={USE_SUBSAMPLE})...")
    train_raw = RawCodeDataset(split='train', subsample=USE_SUBSAMPLE, sample_size=50000)
    val_raw = RawCodeDataset(split='validation', subsample=USE_SUBSAMPLE, sample_size=10000)
    test_raw = RawCodeDataset(split='test', subsample=False)

    train_data_filtered = filter_dataset_codet_m4(train_raw, tokenizer)
    val_data_filtered = filter_dataset_codet_m4(val_raw, tokenizer)

    lang_weights = compute_language_weights(train_data_filtered)

    train_set = CodeDataset(train_data_filtered, tokenizer, MAX_LEN, lang_weights=lang_weights)
    val_set = CodeDataset(val_data_filtered, tokenizer, MAX_LEN) 
    test_set = CodeDataset(test_raw, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    print(f"Initializing Model ({MODEL_NAME})...")
    base_model = AutoModel.from_pretrained(MODEL_NAME)
    
    # --- CRITICAL: ENABLE GRADIENT CHECKPOINTING ---
    # This drastically reduces VRAM usage by not storing intermediate activations
    base_model.gradient_checkpointing_enable()
    print("Gradient Checkpointing Enabled (Memory Saver)")

    model = UniXcoderClassifier(base_model).to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none').to(device)
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
    
    print("\nRunning final evaluation on Test Set...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    y_pred, y_test = get_predictions(model, test_loader, device)

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

if __name__ == "__main__":
    main()
