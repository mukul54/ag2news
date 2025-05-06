import matplotlib.pyplot as plt
import wandb

def plot_accuracy_curves(results):
    """
    Plot training and validation accuracy curves for all configurations
    
    Args:
        results: List of dictionaries containing results for each configuration
    """
    plt.figure(figsize=(12, 8))
    
    epochs = range(1, len(results[0]['train_accs']) + 1)
    
    for result in results:
        plt.plot(epochs, result['train_accs'], 'o-', label=f"{result['config_name']} - Train")
        plt.plot(epochs, result['val_accs'], 's--', label=f"{result['config_name']} - Val")
    
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    # Save figure
    plt.savefig('accuracy_curves.png')
    
    return 'accuracy_curves.png'

def log_final_results(results):
    """
    Log final results to wandb and print a summary table
    
    Args:
        results: List of dictionaries containing results for each configuration
    """
    # Initialize a new wandb run for final results
    wandb.init(project="gpt2-agnews-classification", name="final_results", reinit=True)
    
    # Log the accuracy curves figure
    fig_path = plot_accuracy_curves(results)
    wandb.log({"accuracy_curves": wandb.Image(fig_path)})
    
    # Create and log a table of results
    columns = ["Configuration", "Test Accuracy", "Trainable Parameters", "Avg Train Time (s)"]
    data = []
    
    for result in results:
        data.append([
            result['config_name'], 
            result['test_acc'],
            result['trainable_params'],
            result['train_time']
        ])
    
    results_table = wandb.Table(columns=columns, data=data)
    wandb.log({"results": results_table})
    
    # Finish wandb run
    wandb.finish()
    
    # Print final metrics table
    print("\n=== Final Metrics ===")
    print("{:<20} {:<15} {:<25} {:<15}".format(
        "Configuration", "Test Accuracy", "Trainable Parameters", "Avg Train Time (s)"))
    print("-" * 75)
    
    for result in results:
        print("{:<20} {:<15.4f} {:<25,} {:<15.2f}".format(
            result['config_name'], 
            result['test_acc'],
            result['trainable_params'],
            result['train_time']))