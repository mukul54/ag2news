import os
import torch
from transformers import GPT2Tokenizer
import wandb

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
    RANDOM_SEED, BATCH_SIZE, NUM_EPOCHS, NUM_RUNS,
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

def run_configuration(config_name, train_loader, val_loader, test_loader, device, num_epochs, run_id):
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

def run_all_configurations(train_loader, val_loader, test_loader, device, num_epochs=5, num_runs=3):
    """Run all three configurations with multiple runs"""
    all_results = []
    
    configs = [
        "full_finetuning",
        "prompt_tuning",
        "partial_finetuning"
    ]
    
    for config in configs:
        config_results = []
        for run in range(num_runs):
            print(f"\n--- {config} Run {run+1}/{num_runs} ---")
            result = run_configuration(
                config_name=config,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                num_epochs=num_epochs,
                run_id=run+1
            )
            config_results.append(result)
        
        # Store all individual run results
        all_results.extend(config_results)
    
    # Compute average results
    avg_results = compute_average_results(all_results)
    
    return avg_results

def main():
    # Set random seed for reproducibility
    set_seed(RANDOM_SEED)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize wandb
    os.environ["WANDB_MODE"] = WANDB_MODE
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(tokenizer, BATCH_SIZE)
    
    # Run all configurations
    avg_results = run_all_configurations(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=NUM_EPOCHS,
        num_runs=NUM_RUNS
    )
    
    # Log and visualize final results
    log_final_results(avg_results)

if __name__ == "__main__":
    main()