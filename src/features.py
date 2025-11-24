import torch
import numpy as np
import logging
import re
import math
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from src.dataset import RawCodeDataset
except ImportError:
    from dataset import RawCodeDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. USE A LIGHTWEIGHT, MULTILINGUAL MODEL
# SmolLM2-135M is tiny (135M params) but trained on 2 Trillion tokens (smart).
PPL_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Loading {PPL_MODEL_ID} on {DEVICE}...")
try:
    ppl_tokenizer = AutoTokenizer.from_pretrained(PPL_MODEL_ID)
    ppl_model = AutoModelForCausalLM.from_pretrained(PPL_MODEL_ID).to(DEVICE)
    ppl_model.eval()
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

def extract_code_block(raw_text: str) -> str:
    """Extracts code and cleans lazy language tags."""
    if not raw_text: return ""
    
    # Pattern 1: Markdown Code Blocks
    pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(pattern, raw_text, re.DOTALL)
    if matches: 
        code = "\n".join(matches).strip()
    else:
        code = raw_text.strip()

    # Pattern 2: Lazy Language Tags (The "Infinity" Fix)
    # Removes "python def..." which crashes perplexity models
    common_langs = ["python", "java", "cpp", r"c\+\+", "c", "go", "javascript", "js"]
    lang_pattern = r"^\s*(" + "|".join(common_langs) + r")\s+"
    code = re.sub(lang_pattern, "", code, count=1, flags=re.IGNORECASE)
    
    return code

def get_line_losses(text: str) -> list:
    """Calculates Cross-Entropy LOSS (Log-Perplexity) for each line."""
    lines = text.split('\n')
    valid_lines = [line.strip() for line in lines if len(line.strip()) > 0]
    
    if not valid_lines:
        return []
    
    losses = []
    
    # Handle different config names for context length
    if hasattr(ppl_model.config, 'n_positions'):
        max_len = ppl_model.config.n_positions
    else:
        max_len = getattr(ppl_model.config, 'max_position_embeddings', 2048)

    for line in valid_lines:
        encodings = ppl_tokenizer(line, return_tensors='pt')
        input_ids = encodings.input_ids.to(DEVICE)

        # Truncate long lines to prevent crash
        if input_ids.size(1) > max_len:
            input_ids = input_ids[:, :max_len]

        with torch.no_grad():
            outputs = ppl_model(input_ids, labels=input_ids)
            # USE LOSS (Linear), NOT EXP (Exponential)
            loss = outputs.loss.item()
            losses.append(loss)
            
    return losses

def extract_features(code: str, mode='full') -> np.ndarray:
    """
    The Main Function called by your Collator.
    Returns: [Avg Loss, Std Loss, Burstiness]
    """
    cleaned_code = extract_code_block(code)

    if len(cleaned_code) == 0:
        return np.array([0.0] * 3, dtype=np.float32)
    
    line_losses = get_line_losses(cleaned_code)
    
    # Filter out any random Infs/NaNs
    valid_losses = [x for x in line_losses if math.isfinite(x)]
    
    if len(valid_losses) > 0:
        avg_loss = np.mean(valid_losses)
        std_loss = np.std(valid_losses)
        # Burstiness: Max Loss / Avg Loss
        burstiness = np.max(valid_losses) / (avg_loss + 1e-5)
    else:
        avg_loss, std_loss, burstiness = 0.0, 0.0, 0.0
    
    # Mode Selector
    if mode == 'avg_only':
        return np.array([avg_loss], dtype=np.float32)
    elif mode == 'std_only':
        return np.array([std_loss], dtype=np.float32)
    elif mode == 'burst_only':
        return np.array([burstiness], dtype=np.float32)
    
    # Default 'full'
    return np.array([avg_loss, std_loss, burstiness], dtype=np.float32)

if __name__ == '__main__':
    logger.info("--- TESTING ROBUST FEATURE EXTRACTOR ---")
    try:
        ds = RawCodeDataset(split='train', subsample=True, sample_size=6)
        for i in range(len(ds)):
            item = ds[i]
            feats = extract_features(item['code'])
            print(f"Label: {item['label']} | Loss: {feats[0]:.2f} | Std: {feats[1]:.2f}")
        print("TEST PASSED")
    except Exception as e:
        print(e)