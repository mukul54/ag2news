import torch
from tqdm import tqdm
import wandb
import time

def train_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Log batch metrics to wandb
        wandb.log({"batch/train_loss": loss.item()})
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def train_model(model, train_loader, val_loader, test_loader, optimizer, criterion, 
                device, config_name, num_epochs, run_id):
    """Complete training loop for a model"""
    
    print(f"\n=== Training {config_name} configuration (Run {run_id}) ===")
    
    # Initialize wandb run
    run = wandb.init(
        project="gpt2-agnews-classification",
        name=f"{config_name}_run_{run_id}",
        config={
            "config_name": config_name,
            "epochs": num_epochs,
            "run_id": run_id
        },
        reinit=True
    )
    
    # Log model parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    wandb.config.update({
        "trainable_parameters": trainable_params,
        "total_parameters": total_params
    })
    
    # Training loop
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    total_train_time = 0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        epoch_time = time.time() - start_time
        total_train_time += epoch_time
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Epoch time: {epoch_time:.2f} seconds")
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "epoch_time": epoch_time
        })
    
    # Evaluate on test set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    print(f"Total training time: {total_train_time:.2f} seconds")
    
    # Log final metrics
    wandb.log({
        "test/loss": test_loss,
        "test/accuracy": test_acc,
        "total_train_time": total_train_time
    })
    
    # Close wandb run
    wandb.finish()
    
    # Return metrics
    return {
        'config_name': config_name,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'test_acc': test_acc,
        'trainable_params': trainable_params,
        'train_time': total_train_time
    }