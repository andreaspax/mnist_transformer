import torch
import wandb
import encoder
import tqdm
import dataset
import utils

torch.manual_seed(2)

batch_size = 512
epochs = 40
initial_lr = 0.001
d_model = 64
dff = 1024
device = utils.get_device()
dropout = 0.1
total_samples = 60000

model = encoder.Classification(input_dim=196, dff=dff, d_model=d_model, causal=False)
model.to(device)

params = sum(p.numel() for p in model.parameters())
print("param count:", params)

wandb.init(
    project="mlx6-mnist-transformer",
    config={
        "learning_rate": 'scheduler',
        "epochs": epochs,
        "params": params,
        "encoder_layers": 3,
        "d_model": d_model,
        "dff": dff,
        "single": False,
        "dropout": dropout,
    },
)
#
#
#
train_dataset = dataset.MNISTDataset(train=True, single=False, total_samples=total_samples, seed=2)
test_dataset = dataset.MNISTDataset(train=False, single=False, total_samples=10000, seed=2)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

# Define a custom loss function
def sequence_loss(logits, labels):
    """Calculate loss for sequence prediction"""
    # Reshape logits to (batch_size * 5, num_classes)
    batch_size = logits.size(0)
    logits = logits.view(-1, 12)  # 12 classes (0-9 + special tokens)
    
    # Reshape labels to (batch_size * 5)
    labels = labels.view(-1)
    
    # Calculate cross entropy loss
    return torch.nn.functional.cross_entropy(logits, labels)

# Replace the loss_fn definition
loss_fn = sequence_loss

optimiser = torch.optim.Adam(model.parameters(), lr=initial_lr)

# ReduceLROnPlateau - reduces LR when validation loss stops improving
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser,
    mode='min',
    factor=0.5,  # multiply LR by this factor when reducing
    patience=2,   # number of epochs to wait before reducing LR
    verbose=True  # print message when LR is reduced
)

for epoch in range(epochs):
    print(f"Epoch {epoch + 1} started")
    total_loss = 0
    num_batches = 0
    total_correct = 0
    total_samples = 0
    prgs = tqdm.tqdm((train_dataloader), desc=f"Epoch {epoch+1}", leave=False)
    for _, flattened, _, labels in prgs:            
        flattened, labels = flattened.to(device), labels.to(device)
        
        optimiser.zero_grad()
        logits = model(flattened)  # Shape: (batch_size, 5, num_classes)
        
        # Calculate accuracy
        _, predicted = torch.max(logits, 2)  # Changed dim from 1 to 2
        correct = (predicted == labels).sum().item()
        total_correct += correct
        total_samples += labels.size(0) * labels.size(1)  # Count all numbers
        
        # Calculate loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),  # Reshape to (batch_size * 5, num_classes)
            labels.view(-1)  # Reshape to (batch_size * 5)
        )
        
        loss.backward()
        
        optimiser.step()

        total_loss += loss.item()
        num_batches += 1

        wandb.log({
            "batch_train_loss": loss.item(), 
            "batch_train_accuracy": correct / labels.size(0)
        })

    avg_loss = total_loss / num_batches
    epoch_accuracy = total_correct / total_samples

    
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Acc: {epoch_accuracy:.2%}")

    # Validation phase
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():  # Disable gradient computation for validation
        val_prgs = tqdm.tqdm(test_dataloader, desc="Validation", leave=False)
        for _, val_flattened, _, val_labels in val_prgs:
            val_flattened, val_labels = val_flattened.to(device), val_labels.to(device)
            
            # Forward pass
            val_logits = model(val_flattened)
            
            # Calculate validation loss
            batch_val_loss = loss_fn(val_logits, val_labels)
            val_loss += batch_val_loss.item()
            
            # Calculate validation accuracy
            _, val_predicted = torch.max(val_logits, 2)  # Change dim to 2 for sequence predictions
            val_correct += (val_predicted == val_labels).sum().item()
            val_total += val_labels.size(0) * val_labels.size(1)  # Count all numbers in sequence
    
    # Calculate average validation metrics
    avg_val_loss = val_loss / len(test_dataloader)
    val_accuracy = val_correct / val_total
    
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Acc: {val_accuracy:.2%}")
    
    # Step the scheduler
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(avg_val_loss)  # For ReduceLROnPlateau
    
    # Log learning rate
    current_lr = optimiser.param_groups[0]['lr']
    wandb.log({
        "learning_rate": current_lr,
        "train_loss": avg_loss,
        "train_accuracy": epoch_accuracy,
        "val_loss": avg_val_loss,
        "val_accuracy": val_accuracy,
        "epoch": epoch
    })

    # Set model back to training mode
    model.train()

    torch.save(model.state_dict(), "weights/2x2_encoder.pt")

print("Uploading...")
artifact = wandb.Artifact("model-weights", type="2x2_encoder")
artifact.add_file("./weights/2x2_encoder.pt")
wandb.log_artifact(artifact)

print("Done!")
wandb.finish()
