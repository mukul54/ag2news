"""
Configuration settings for GPT-2 fine-tuning experiments
"""

# General configuration
RANDOM_SEED = 42
BATCH_SIZE = 128
NUM_EPOCHS = 5
NUM_RUNS = 3
MAX_SEQ_LENGTH = 256
NUM_CLASSES = 4

# Optimizer settings
WEIGHT_DECAY = 0.01  # AdamW weight decay

# Model configurations
MODEL_CONFIGS = {
    "full_finetuning": {
        "description": "Full fine-tuning with no layer freezing",
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
    },
    "prompt_tuning": {
        "description": "Prompt tuning with frozen model weights",
        "num_prompt_tokens": 20,
        "learning_rate": 5e-4,
        "weight_decay": 0.01,
    },
    "partial_finetuning": {
        "description": "Partial fine-tuning with lower layers frozen",
        "freeze_percentage": 0.5,
        "base_lr": 5e-5,
        "top_lr_multiplier": 5,
        "weight_decay": 0.01,
    }
}

# Wandb configuration
WANDB_PROJECT = "gpt2-agnews-classification"
WANDB_ENTITY = "mukul54"  # Set to your username if you have a wandb account
WANDB_MODE = "online"  # Set to "disabled" to turn off wandb logging