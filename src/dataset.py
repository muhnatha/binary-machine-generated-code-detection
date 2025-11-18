import logging
from datasets import load_dataset
from torch.utils.data import Dataset

DATASET_NAME = "Daniil Or/SemEval-2026-Task13"
DATASET_TASK = "task_a"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RawCodeDataset(Dataset):
    """
    A simple PyTorch Dataset that loads the raw 'code' and 'label'
    from the hugging face dataset.
    """
    def __init__(self, split='train', subsample=True, sample_size=5000):
        """
        Args:
            split (str): The dataset split to load ('train', 'validation', 'test').
            subsample (bool): Whether to subsample the dataset.
            sample_size (int): The number of samples to keep if subsampling.
        """
        logger.info(f"Loading raw {split} dataset from hugging face")

        try:
            full_dataset = load_dataset(DATASET_NAME, DATASET_TASK, split=split)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

        if subsample:
            logger.warning(f"Subsampling {split} data to {sample_size}")
            logger.warning("To run on full data, set subsample=False")

            logger.info("Creating stratification column")
            full_dataset = full_dataset.map(create_stratification_column)

            if len(full_dataset) > sample_size:
                proportion = sample_size / len(full_dataset)

                self.raw_dataset = full_dataset.train_test_split(
                    test_size=proportion,
                    shuffle=True,
                    seed=10,
                    stratify_by_column="label"
                )['test'] 
                # clearn up the stratify column after balancing
                self.raw_dataset = self.raw_dataset.remove_columns(['stratify_col']) 
            else:
                logger.warning("Sample size is larger than dataset size; using full dataset.")
                self.raw_dataset = full_dataset
        else:
            self.raw_dataset = full_dataset
        
        logger.info(f"Loaded dataset with {len(self.raw_dataset)} examples")
    
    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        item = self.raw_dataset[idx]
        return{
            'code': item['code'],
            'label': item['label']
        }
            
def create_stratification_column(example):
    """
    combines 'label' and 'programming_language' into a single column for stratification
    """
    return {
        'stratify_col': f"{example['label']}_{example['programming_language']}"
    }

if __name__ == '__main__':
    """
    This block allows you to run this file directly to test it
    """
    logger.info("Running dataset.py as a script for testing...")
    
    # Test the 'train' split with 5000 samples
    train_dataset = RawCodeDataset(
        split='train', 
        subsample=True, 
        sample_size=5000
    )
    
    print("\n" + "="*30)
    print("  TESTING COMPLETE ")
    print("="*30)
    print(f"Train dataset size: {len(train_dataset)}")
    
    # --- Check the balance of the training sample ---
    print("\nChecking balance of 5000-sample training set:")
    labels = [train_dataset[i]['label'] for i in range(len(train_dataset))]
    label_0_count = sum(1 for label in labels if label == 0)
    label_1_count = sum(1 for label in labels if label == 1)
    
    print(f"Label 0 (Human): {label_0_count} samples")
    print(f"Label 1 (Machine): {label_1_count} samples")
    print("These should be very close to 2500 each.")
    print("\nNote: This sample is now also stratified by programming language.")
    print("="*30)
