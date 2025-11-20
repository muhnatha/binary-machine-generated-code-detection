import torch
import numpy as np
import logging
import re
import math
from collections import Counter
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PPL_MODEL_ID = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"loading {PPL_MODEL_ID} on {DEVICE} for feature extraction")
try:
    ppl_tokenizer = GPT2TokenizerFast.from_pretrained(PPL_MODEL_ID)
    ppl_model = GPT2LMHeadModel.from_pretrained(PPL_MODEL_ID).to(DEVICE)
    ppl_model.eval()
except Exception as e:
    logger.error(f"Failed to load GPT2 model: {e}")
    raise

def extract_code_block(raw_text: str) -> str:
    """ Extracts code from markdown blocks"""
    if not raw_text: return""
    pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(pattern, raw_text, re.DOTALL)
    if matches: return "\n".join(matches).strip()
    return raw_text.strip()

def get_line_perplexities(text: str) -> list:
    """Calculates perplexity for each line in the text."""
    lines = text.split('\n')
    valid_lines = [line.strip() for line in lines if len(line.strip()) > 0] # remove empty lines

    # if no valid lines, return empty list
    if not valid_lines:
        return []
    
    perplexities = []

    for line in valid_lines:
        encodings = ppl_tokenizer(line, return_tensors='pt')
        input_ids = encodings.input_ids.to(DEVICE)

        if input_ids.size(1)>ppl_model.config.n_positions:
            continue

        with torch.no_grad():
            outputs = ppl_model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
            perplexities.append(perplexity)
    return perplexities

def extract_features(code: str) -> np.ndarray:
    """Extracts features from the given code snippet."""
    cleaned_code = extract_code_block(code)

    if len(cleaned_code) == 0:
        return np.array([0.0] * 6, dtype=np.float32)
    
    log_length = math.log(len(cleaned_code) + 1)
    lines_raw = cleaned_code.split('\n')
    line_count = float(len(lines_raw))

    line_ppls = get_line_perplexities(cleaned_code)
    if len(line_ppls) > 0:
        avg_ppl = np.mean(line_ppls)
        std_ppl = np.std(line_ppls)
        burstiness = np.max(line_ppls) / (avg_ppl + 1e-5)
    else:
        avg_ppl = 0.0
        std_ppl = 0.0
        burstiness = 0.0
    
    return np.array([
        avg_ppl,
        std_ppl,
        burstiness
    ], dtype=np.float32)

if __name__ == '__main__':
    logger.info("--- TESTING RESEARCH FEATURES ---")
    
    human_code = """
    def weird_algo(x):
        # chaotic human logic
        temp = x * 999
        if temp > 5000: return "Big"
        return {k:v for k,v in enumerate("nonsense")}
    """
    
    ai_code = """
    def calculate_area(radius):
        if radius < 0:
            return 0
        pi = 3.14159
        return pi * (radius ** 2)
    """

    print("\n--- Human Code ---")
    f_h = extract_features(human_code)
    print(f"Avg PPL: {f_h[0]:.2f} | STD: {f_h[1]:.2f} | Burstiness: {f_h[2]:.2f}")

    print("\n--- AI Code ---")
    f_a = extract_features(ai_code)
    print(f"Avg PPL: {f_a[0]:.2f} | STD: {f_a[1]:.2f} | Burstiness: {f_a[2]:.2f}")
    
    print("\nNote: Humans typically have higher STD and Burstiness.")

