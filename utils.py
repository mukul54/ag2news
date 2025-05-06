import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

def count_trainable_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_optimizer(model, config_name):
    """Configure the optimizer based on the training configuration"""
    if config_name == "full_finetuning":
        optimizer = optim.AdamW(model.parameters(), lr=5e-5)
        return optimizer, {"learning_rate": 5e-5}
    
    elif config_name == "prompt_tuning":
        # Fix: Don't call .parameters() on prompt_embeddings since it's already a Parameter
        optimizer = optim.AdamW([
            model.prompt_embeddings,  # This is already a nn.Parameter
            *model.classifier.parameters()  # Unpack the parameters from classifier
        ], lr=5e-4)
        return optimizer, {"learning_rate": 5e-4, "num_prompt_tokens": 20}
    
    elif config_name == "partial_finetuning":
        # Create parameter groups with different learning rates
        base_lr = 5e-5
        top_lr = 5 * base_lr  # 5x larger LR for top layers
        
        # Define optimizer with parameter groups
        optimizer_grouped_parameters = [
            {'params': model.classifier.parameters(), 'lr': top_lr},
            {'params': [p for n, p in model.named_parameters() if 'classifier' not in n and p.requires_grad], 'lr': base_lr}
        ]
        
        optimizer = optim.AdamW(optimizer_grouped_parameters)
        return optimizer, {"base_lr": base_lr, "top_lr": top_lr, "freeze_percentage": 0.5}
    
    else:
        raise ValueError(f"Unknown configuration: {config_name}")

def get_criterion():
    """Return the loss function"""
    return nn.CrossEntropyLoss()

def compute_average_results(all_results):
    """Compute average results across multiple runs"""
    configs = list(set([r['config_name'] for r in all_results]))
    averaged_results = []
    
    for config in configs:
        config_results = [r for r in all_results if r['config_name'] == config]
        
        avg_result = {
            'config_name': config,
            'train_accs': np.mean([r['train_accs'] for r in config_results], axis=0).tolist(),
            'val_accs': np.mean([r['val_accs'] for r in config_results], axis=0).tolist(),
            'test_acc': np.mean([r['test_acc'] for r in config_results]),
            'trainable_params': config_results[0]['trainable_params'],
            'train_time': np.mean([r['train_time'] for r in config_results])
        }
        
        averaged_results.append(avg_result)
    
    return averaged_results

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)