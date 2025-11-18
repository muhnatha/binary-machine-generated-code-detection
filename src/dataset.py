import logging
from datasets import load_dataset, ClassLabel
from torch.utils.data import Dataset

DATASET_NAME = "DaniilOr/SemEval-2026-Task13"
DATASET_TASK = "A"

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

            try:
                full_dataset = full_dataset.class_encode_column("stratify_col")
            except AttributeError:
                # Fallback for older datasets versions
                full_dataset = full_dataset.cast_column("stratify_col", ClassLabel(names=list(set(full_dataset["stratify_col"]))))

            if len(full_dataset) > sample_size:
                proportion = sample_size / len(full_dataset)

                self.raw_dataset = full_dataset.train_test_split(
                    test_size=proportion,
                    shuffle=True,
                    seed=10,
                    stratify_by_column="stratify_col"
                )['test'] 
                # clean up the stratify column after balancing
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
    combines 'label' and 'language' into a single column for stratification
    """
    return {
        'stratify_col': f"{example['label']}_{example['language']}"
    }

if __name__ == '__main__':
    """
    This block allows you to run this file directly to test it
    """
    logger.info("--- TESTING DATASET LOADING ---")
    
    # Training Set (Subsampled & Stratified)
    print("\n1. Loading Training Set...")
    train_dataset = RawCodeDataset(split='train', subsample=True, sample_size=5000)
    
    # Validation Set (Subsampled & Stratified)
    print("\n2. Loading Validation Set...")
    val_dataset = RawCodeDataset(split='validation', subsample=True, sample_size=2000)
    
    # Test Set (Full - No Subsampling)
    print("\n3. Loading Test Set...")
    test_dataset = RawCodeDataset(split='test', subsample=False)

    print("\n" + "="*30)
    print("  SUMMARY ")
    print("="*30)
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size:   {len(val_dataset)}")
    print(f"Test size:  {len(test_dataset)}")
    print("="*30)
