import os
import torch
from transformers import GPT2Tokenizer
import wandb
import argparse

# Import from our modules
from dataset import create_dataloaders
from models import (
    GPT2ForSequenceClassification, 
    GPT2PromptTuning, 
    GPT2PartialFineTuning
)
from train import train_model
from utils import get_optimizer, get_criterion, compute_average_results, set_seed
from visualization import log_final_results
from config import (
    RANDOM_SEED, BATCH_SIZE, NUM_EPOCHS,
    WANDB_PROJECT, WANDB_MODE
)

def initialize_model(config_name, device):
    """Initialize model based on configuration name"""
    if config_name == "full_finetuning":
        return GPT2ForSequenceClassification(num_classes=4).to(device)
    
    elif config_name == "prompt_tuning":
        return GPT2PromptTuning(num_classes=4, num_prompt_tokens=20).to(device)
    
    elif config_name == "partial_finetuning":
        return GPT2PartialFineTuning(num_classes=4, freeze_percentage=0.5).to(device)
    
    else:
        raise ValueError(f"Unknown configuration: {config_name}")

def run_configuration(config_name, train_loader, val_loader, test_loader, device, num_epochs, run_id=1):
    """Run a single configuration"""
    # Initialize model
    model = initialize_model(config_name, device)
    
    # Get optimizer and criterion
    optimizer, optim_config = get_optimizer(model, config_name)
    criterion = get_criterion()
    
    # Run training
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config_name=config_name,
        num_epochs=num_epochs,
        run_id=run_id
    )
    
    return results

def run_single_configuration(config_name, train_loader, val_loader, test_loader, device, num_epochs=5):
    """Run one specific configuration"""
    print(f"\n=== Running {config_name} configuration ===")
    result = run_configuration(
        config_name=config_name,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=num_epochs
    )
    
    # Return as a list for compatibility with visualization code
    return [result]

def run_all_configurations(train_loader, val_loader, test_loader, device, num_epochs=5):
    """Run all three configurations (one run each)"""
    all_results = []
    
    configs = [
        "full_finetuning",
        "prompt_tuning",
        "partial_finetuning"
    ]
    
    for config in configs:
        print(f"\n--- Running {config} ---")
        result = run_configuration(
            config_name=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            num_epochs=num_epochs
        )
        all_results.append(result)
    
    return all_results

def parse_args():
    parser = argparse.ArgumentParser(description='GPT-2 Fine-tuning for AG News Classification')
    parser.add_argument('--config', type=str, default='all', 
                        choices=['all', 'full_finetuning', 'prompt_tuning', 'partial_finetuning'],
                        help='Configuration to run (default: all)')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f'Number of training epochs (default: {NUM_EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help=f'Random seed (default: {RANDOM_SEED})')
    parser.add_argument('--wandb', type=str, default=WANDB_MODE,
                        choices=['online', 'offline', 'disabled'],
                        help=f'Weights & Biases mode (default: {WANDB_MODE})')
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize wandb
    os.environ["WANDB_MODE"] = args.wandb
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(tokenizer, args.batch_size)
    
    # Run configurations based on command-line argument
    if args.config == 'all':
        # Run all three configurations
        results = run_all_configurations(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            num_epochs=args.epochs
        )
    else:
        # Run a single configuration
        results = run_single_configuration(
            config_name=args.config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            num_epochs=args.epochs
        )
    
    # Log and visualize final results
    log_final_results(results)

if __name__ == "__main__":
    main()