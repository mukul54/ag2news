# GPT-2 Fine-tuning for AG News Classification

This project implements and compares three different fine-tuning strategies for the GPT-2 model on the AG News Classification dataset.

## Overview

The AG News dataset consists of news articles categorized into 4 classes: World, Sports, Business, and Sci/Tech. We compare:

1. **Full Fine-tuning**: Update all parameters in GPT-2 with a single learning rate
2. **Prompt Tuning**: Freeze GPT-2 weights and only train soft prompt tokens + classification head
3. **Partial Fine-tuning**: Freeze lower 50% of GPT-2 layers and use different learning rates for different layers

## Installation

```bash
# Clone the repository
git clone https://github.com/mukul54/gpt2-agnews.git
cd gpt2-agnews

# Install requirements
pip install -r requirements.txt

# Optional: Login to wandb for experiment tracking
wandb login
```

## Usage

Run the experiment with:

```bash
python main.py
```

This will:
- Train all three configurations for 5 epochs
- Perform 3 runs per configuration to ensure statistical significance
- Track metrics with Weights & Biases
- Generate accuracy plots
- Report final test accuracy, parameter counts, and training times

## Configuration

You can customize experiment settings in `config.py`:

```python
# General configuration
RANDOM_SEED = 42
BATCH_SIZE = 128
NUM_EPOCHS = 5
NUM_RUNS = 3
MAX_SEQ_LENGTH = 256 
```

## Project Structure

```
├── main.py            # Main execution script
├── dataset.py         # Dataset loading and preprocessing
├── models.py          # Model implementations
├── train.py           # Training and evaluation logic
├── utils.py           # Helper functions
├── visualization.py   # Visualization and logging utilities 
├── config.py          # Configuration parameters
└── requirements.txt   # Dependencies
```

## Fine-tuning Approaches

### 1. Full Fine-tuning
- Updates all parameters in the GPT-2 model
- Learning rate: 5e-5
- Parameter efficient: No

### 2. Prompt Tuning
- Freezes all GPT-2 model parameters
- Only trains 20 soft prompt tokens and classifier head
- Learning rate: 5e-4
- Parameter efficient: Yes

### 3. Partial Fine-tuning
- Freezes embedding layer and lower 50% of transformer layers
- Uses differential learning rates:
  - Base LR: 5e-5
  - Top layers: 25e-5 (5x larger)
- Parameter efficient: Moderate

## Results

The experiment produces:
- Training and validation accuracy curves
- Final test accuracy for each approach
- Number of trainable parameters
- Total training time

Results are logged to Weights & Biases and saved locally as `accuracy_curves.png`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.