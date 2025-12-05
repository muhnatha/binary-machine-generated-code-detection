import torch
import logging
import numpy as np
from transformers import RobertaTokenizer
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import RawCodeDataset

try:
    from src.features import extract_features
except ImportError:
    # fallback for direct script execution
    from features import extract_features

logger = logging.getLogger(__name__)


class FusedDataCollator:
    """
    Custom data collator that processes a batch of raw code

    It connects raw code inputs to feature extraction and tokenization,
    1. Path A: Tokenizes code for CodeBERT (Semantic).
    2. Path B: Extracts statistical features via GPT-2 (Intrinsic).
    """

    def __init__(self, tokenizer_name="microsoft/codebert-base", max_length=512, mode="full"):
        """
        Args:
            tokenizer_name (str): HuggingFace model name for Path A.
            max_length (int): Max token length for Path A.
            mode (str): The experiment mode ('full', 'avg_only', 'std_only', 'burst_only').
                        This controls which features are extracted.
        """
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.mode = mode

        logger.info(f"collator initialized in {self.mode} mode")

    def __call__(self, batch):
        texts = [item["code"] for item in batch]
        labels = [item["label"] for item in batch]

        # PATH A: Tokenize for CodeBERT
        # return_tensors='pt' gives us PyTorch tensors directly
        # tokenized = self.tokenizer(
        #     texts,
        #     padding=True,
        #     truncation=True,
        #     max_length=self.max_length,
        #     return_tensors="pt"
        # )

        input_ids = None
        attention_mask = None

        # PATH B: Extract Meta-Features (The Research Part)
        # We loop through the batch and calculate features for each sample.
        # The 'mode' argument determines if we get a vector of size 1, 3, or 5.
        meta_features_list = []
        for text in texts:
            # get ALL features: [avg_ppl, std_ppl, burstiness]
            full_feats = extract_features(text)

            if self.mode == "full":
                # Use all 5 features
                selected_feats = full_feats
            elif self.mode == "avg_only":
                # Keep only Average Perplexity
                selected_feats = np.array([full_feats[0]], dtype=np.float32)
            elif self.mode == "std_only":
                # Keep only Std Dev Perplexity
                selected_feats = np.array([full_feats[1]], dtype=np.float32)
            elif self.mode == "burst_only":
                # Keep only Burstiness (Index 4)
                selected_feats = np.array([full_feats[2]], dtype=np.float32)
            elif self.mode == "baseline":
                # empty feature vector for baseline
                selected_feats = np.array([], dtype=np.float32)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            meta_features_list.append(selected_feats)

        # Stack into a single tensor (Batch Size, Feature Count)
        meta_features_tensor = torch.tensor(np.array(meta_features_list), dtype=torch.float32)

        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "meta_features": meta_features_tensor,
            "labels": labels_tensor,
            "raw_code": texts,
        }


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Output directory
    os.makedirs("processed_data", exist_ok=True)

    def save_split(split_name, filename, sample_size=None):
        logger.info(f"--- Processing {split_name} set ---")

        # Load Dataset
        # subsample=False means use the WHOLE dataset if sample_size is None
        is_subsample = sample_size is not None
        ds = RawCodeDataset(split=split_name, subsample=is_subsample, sample_size=sample_size)

        # Create Loader
        # Use batch_size=8 to prevent GPU OOM with the 3B model
        collator = FusedDataCollator(mode="full")
        loader = DataLoader(ds, batch_size=8, collate_fn=collator)

        all_data = []

        # Loop and Extract
        for batch in tqdm(loader, desc=f"Extracting {split_name}"):
            codes = batch["raw_code"]
            features = batch["meta_features"].numpy()
            labels = batch["labels"].numpy()

            for i in range(len(codes)):
                all_data.append(
                    {
                        "label": labels[i],
                        "avg_loss": features[i][0],
                        "std_loss": features[i][1],
                        "burstiness": features[i][2],
                        "code": codes[i],
                    }
                )

        # Save to CSV
        df = pd.DataFrame(all_data)
        output_path = os.path.join("processed_data", filename)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} rows to {output_path}")

    # You can change these numbers or set sample_size=None for the full data
    save_split("train", "train_featureshk.csv", sample_size=100000)
    save_split("validation", "validation_featureshk.csv", sample_size=10000)
    # save_split('test', 'test_features.csv', sample_size=None)
