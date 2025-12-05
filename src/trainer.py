import os
# --- 1. MEMORY CONFIGURATION ---
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
import torch.nn as nn
import numpy as np
import gc
import re
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
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

# --- AST Parser Imports ---
try:
    from tree_sitter import Language, Parser
    import tree_sitter_languages
    TREE_SITTER_AVAILABLE = True
except ImportError:
    print("⚠️ WARNING: tree-sitter not installed. AST features will be empty.")
    print("👉 Run: pip install tree-sitter tree-sitter-languages")
    TREE_SITTER_AVAILABLE = False

# --- Configuration ---
OUTPUT_DIR = "output"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "model.pth")
METRICS_SAVE_PATH = os.path.join(OUTPUT_DIR, "metrics.json")
MODEL_NAME = "microsoft/unixcoder-base"

# --- TRAINING SETTINGS ---
MAX_LEN = 512             
BATCH_SIZE = 1            
ACCUMULATION_STEPS = 16   
EPOCHS = 3           
PATIENCE = 3              
LEARNING_RATE = 5e-5      
CLASS_NAMES = ['Human-Written', 'Machine-Generated']

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 2. AST PARSER UTILITY
# ---------------------------------------------------------
class ASTParser:
    def __init__(self):
        self.parsers = {}
        # Mapping dataset language names to tree-sitter language names
        self.lang_map = {
            'Python': 'python',
            'Java': 'java',
            'C++': 'cpp',
            'Go': 'go',
            'PHP': 'php',
            'JavaScript': 'javascript',
            'C': 'c',
            'C#': 'c_sharp',
            'Ruby': 'ruby'
        }

    def get_parser(self, lang_name):
        if not TREE_SITTER_AVAILABLE:
            return None
            
        ts_lang = self.lang_map.get(lang_name)
        if not ts_lang:
            return None
        
        if ts_lang not in self.parsers:
            try:
                parser = Parser()
                language = tree_sitter_languages.get_language(ts_lang)
                parser.set_language(language)
                self.parsers[ts_lang] = parser
            except Exception as e:
                # Fail silently for languages not installed
                self.parsers[ts_lang] = None
        
        return self.parsers[ts_lang]

    def parse_to_flattened_ast(self, code, lang_name):
        """
        Parses code and returns a string sequence of AST node types.
        Example output: "function_definition identifier parameters block return_statement"
        """
        parser = self.get_parser(lang_name)
        if not parser:
            return ""

        try:
            # Tree-sitter expects bytes
            tree = parser.parse(bytes(code, "utf8"))
            cursor = tree.walk()
            
            tokens = []
            
            # Efficient Depth-First Traversal
            visited_children = False
            while True:
                if not visited_children:
                    # We only keep 'named' nodes (structural elements), ignoring punctuation
                    if cursor.node.is_named:
                        tokens.append(cursor.node.type)
                    
                    if cursor.goto_first_child():
                        continue
                
                if cursor.goto_next_sibling():
                    visited_children = False
                elif cursor.goto_parent():
                    visited_children = True
                else:
                    break
            
            return " ".join(tokens)
            
        except Exception:
            return ""

# ---------------------------------------------------------
# 3. DATASET CLASS (Cross-Modal)
# ---------------------------------------------------------
def clean_code_strict(code):
    if pd.isna(code) or code == '': return ''
    code = str(code)
    code = re.sub(r'<[^>]+>', '', code)
    lines = [line.rstrip() for line in code.split('\n') if line.strip()]
    return '\n'.join(lines)

class CodeDataset(Dataset):
    def __init__(self, data_source, tokenizer, max_len, lang_weights=None):
        self.data_source = data_source 
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.lang_weights = lang_weights
        self.ast_parser = ASTParser() # Initialize Parser

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, index):
        item = self.data_source[index]
        code_text = str(item['code'])
        label = item['label']
        language = item['language']

        # 1. Clean Code
        code_text = clean_code_strict(code_text)
        
        # 2. Generate AST
        ast_text = self.ast_parser.parse_to_flattened_ast(code_text, language)

        # 3. Tokenize Separately to manage length budget
        # We allocate ~70% of tokens to Code, ~30% to AST
        max_code_len = int(self.max_len * 0.70)
        max_ast_len = self.max_len - max_code_len - 3 # Reserve space for special tokens

        code_tokens = self.tokenizer.tokenize(code_text)
        ast_tokens = self.tokenizer.tokenize(ast_text)
        
        # Truncate
        code_tokens = code_tokens[:max_code_len]
        ast_tokens = ast_tokens[:max_ast_len]

        # 4. Construct Input: <s> Code </s> AST </s>
        # [CLS] is <s>, [SEP] is </s> in UniXcoder tokenizer
        input_tokens = [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token] + ast_tokens + [self.tokenizer.sep_token]
        
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        attention_mask = [1] * len(input_ids)

        weight = 1.0
        if self.lang_weights:
            weight = self.lang_weights.get(language, 1.0)

        # Return Tensors (No Padding Here)
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.float),
            'sample_weight': torch.tensor(weight, dtype=torch.float)
        }

def dynamic_collate_fn(batch):
    """
    Pads batch to the longest sequence IN THE BATCH.
    """
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = torch.stack([item['labels'] for item in batch])
    weights = torch.stack([item['sample_weight'] for item in batch])

    # Pad ID 1 (UniXcoder pad token)
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=1)
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'labels': labels,
        'sample_weight': weights
    }

# ---------------------------------------------------------
# 4. MODEL & UTILS
# ---------------------------------------------------------
class UniXcoderClassifier(nn.Module):
    def __init__(self, base_model):
        super(UniXcoderClassifier, self).__init__()
        self.bert = base_model
        self.drop = nn.Dropout(p=0.3)
        hidden_size = self.bert.config.hidden_size 
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # Encoder-Only Mode: Attention mask is all 1s (handled by dataset)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Pool the [CLS] token (index 0)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        output = self.drop(pooled_output)
        return self.out(output)

def compute_language_weights(raw_data):
    langs = [item['language'] for item in raw_data]
    count = Counter(langs)
    total = len(langs)
    weights = {l: total / (len(count) * freq) for l, freq in count.items()}
    print(f"Language Weights: {list(weights.items())[:3]}...")
    return weights

def freeze_layers(model, num_layers=8):
    print(f"❄️ Freezing bottom {num_layers} layers...")
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    if hasattr(model.bert, 'encoder'):
        for i in range(num_layers):
            for param in model.bert.encoder.layer[i].parameters():
                param.requires_grad = False

# ---------------------------------------------------------
# 5. TRAINING LOOP
# ---------------------------------------------------------
def train_epoch(model, data_loader, loss_fn, optimizer, scaler, device, n_examples):
    model = model.train()
    losses = []
    optimizer.zero_grad(set_to_none=True)

    for i, d in tqdm(enumerate(data_loader), total=len(data_loader), desc="Train", leave=False):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)
        weights = d["sample_weight"].to(device)

        try:
            with autocast():
                outputs = model(input_ids, attention_mask).view(-1)
                loss = (loss_fn(outputs, targets) * weights).mean() / ACCUMULATION_STEPS

            scaler.scale(loss).backward()
            
            # Immediate cleanup
            loss_item = loss.item() * ACCUMULATION_STEPS
            del loss, outputs
            
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            losses.append(loss_item)

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                continue
            raise e

    return np.mean(losses) if losses else 0

def eval_model(model, data_loader, loss_fn, device):
    model = model.eval()
    losses, preds, targets = [], [], []
    
    with torch.no_grad():
        for d in tqdm(data_loader, desc="Eval", leave=False):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            y = d["labels"].to(device)
            
            with autocast():
                out = model(input_ids, attention_mask).view(-1)
                loss = loss_fn(out, y).mean()
                
            losses.append(loss.item())
            preds.extend((torch.sigmoid(out) > 0.5).float().cpu().numpy())
            targets.extend(y.cpu().numpy())
            
    return np.mean(losses), f1_score(targets, preds, average='binary')

# ---------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.cuda.empty_cache()

    # 1. Load Data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Set subsample=False for full run
    USE_SUBSAMPLE = True
    train_raw = RawCodeDataset('train', subsample=USE_SUBSAMPLE, sample_size=5000)
    val_raw = RawCodeDataset('validation', subsample=USE_SUBSAMPLE, sample_size=1000)
    test_raw = RawCodeDataset('test', subsample=False)

    train_data = [{'code': x['code'], 'label': x['label'], 'language': x['language']} for x in train_raw]
    val_data = [{'code': x['code'], 'label': x['label'], 'language': x['language']} for x in val_raw]
    test_data = [{'code': x['code'], 'label': x['label'], 'language': x['language']} for x in test_raw]

    lang_weights = compute_language_weights(train_data)

    # 2. Datasets (Now with AST parsing inside)
    train_set = CodeDataset(train_data, tokenizer, MAX_LEN, lang_weights)
    val_set = CodeDataset(val_data, tokenizer, MAX_LEN)
    test_set = CodeDataset(test_data, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, collate_fn=dynamic_collate_fn)
    val_loader = DataLoader(val_set, BATCH_SIZE, collate_fn=dynamic_collate_fn)
    test_loader = DataLoader(test_set, BATCH_SIZE, collate_fn=dynamic_collate_fn)

    # 3. Model
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.gradient_checkpointing = True
    base_model = AutoModel.from_pretrained(MODEL_NAME, config=config)
    base_model.gradient_checkpointing_enable()
    
    model = UniXcoderClassifier(base_model).to(device)
    
    # 4. Freeze Bottom Layers (Preserve Multilingual Knowledge)
    freeze_layers(model, num_layers=8)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none').to(device)
    scaler = GradScaler()

    best_f1 = 0
    patience_cnt = 0

    print("Starting Training...")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, scaler, device, len(train_set))
        val_loss, val_f1 = eval_model(model, val_loader, loss_fn, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_cnt = 0
            print("Saved Best Model.")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print("Early stopping.")
                break

    print("Done.")

if __name__ == "__main__":
    main()
