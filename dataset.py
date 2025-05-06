import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from datasets import load_dataset

class AGNewsDataset(Dataset):
    def __init__(self, split, tokenizer, max_length=128):
        """
        Dataset for AG News classification
        
        Args:
            split: 'train', 'validation', or 'test'
            tokenizer: GPT2Tokenizer for encoding the text
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load dataset from HuggingFace
        # AG News has 4 classes: World (1), Sports (2), Business (3), Sci/Tech (4)
        if split == 'validation':
            # Create validation split from train
            full_train = load_dataset("ag_news", split="train")
            train_val = full_train.train_test_split(test_size=0.1, seed=42)
            self.dataset = train_val['test']
        elif split == 'train':
            full_train = load_dataset("ag_news", split="train")
            train_val = full_train.train_test_split(test_size=0.1, seed=42)
            self.dataset = train_val['train']
        else:
            self.dataset = load_dataset("ag_news", split="test")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        # AG_NEWS labels are 0-indexed in HuggingFace dataset
        label = item['label']
        
        encoding = self.tokenizer(
            text, 
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        
        # Remove batch dimension added by tokenizer
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }

def create_dataloaders(tokenizer, batch_size=16):
    """Create and return dataloaders for train, validation, and test sets"""
    # Create datasets
    train_dataset = AGNewsDataset('train', tokenizer)
    val_dataset = AGNewsDataset('validation', tokenizer)
    test_dataset = AGNewsDataset('test', tokenizer)
    
    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size
    )
    
    return train_loader, val_loader, test_loader