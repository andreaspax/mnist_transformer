import torch
# import wandb
import tqdm
import dataset
import utils
import transformer

torch.manual_seed(2)

batch_size = 512
epochs = 5
initial_lr = 0.001
d_model = 64
dff = 1024
device = utils.get_device()
dropout = 0.1
total_samples = 60000

model = transformer.Transformer(d_model, dff, vocab_size=12, seq_len_x=5, seq_len_y=16)
model.to(device)

params = sum(p.numel() for p in model.parameters())
print("param count:", params)

# wandb.init(
#     project="mlx6-mnist-transformer",
#     config={
#         "learning_rate": 'scheduler',
#         "epochs": epochs,
#         "params": params,
#         "encoder_layers": 3,
#         "decoder_layers": 3,
#         "d_model": d_model,
#         "dff": dff,
#         "dropout": dropout,
#     },
# )


train_dataset = dataset.MNISTDataset(train=True, single=False, total_samples=total_samples, seed=2)
test_dataset = dataset.MNISTDataset(train=False, single=False, total_samples=10000, seed=2)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0)


# Replace the loss_fn definition
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

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
    for _, flattened, input_labels, output_labels in prgs:            
        flattened, input_labels, output_labels = flattened.to(device), input_labels.to(device), output_labels.to(device)
        
        optimiser.zero_grad()
        logits = model(input_labels, flattened)  # Shape: (batch_size, seq_len, vocab_size)
        logits = logits.reshape(-1, 12)     # Reshape logits to (batch_size * seq_len, vocab_size)
        
        output_labels = output_labels.reshape(-1)   # Reshape labels t o (batch_size * seq_len)

        # Calculate loss using sequence loss
        loss = loss_fn(logits, output_labels)
        
        # Calculate accuracy
        _, predicted = torch.max(logits, dim=-1)  # Get predictions
        correct = (predicted == output_labels).sum().item()
        total_correct += correct
        total_samples += output_labels.numel()  # Count all elements in the sequence
        
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimiser.step()

        total_loss += loss.item()
        num_batches += 1

        # Update progress bar with current metrics
        prgs.set_postfix({
            'loss': loss.item(),
            'acc': correct / output_labels.numel()
        })

    avg_loss = total_loss / num_batches
    epoch_accuracy = total_correct / total_samples

        # Step the scheduler
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(avg_loss)  # For ReduceLROnPlateau

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Acc: {epoch_accuracy:.2%}")

print("Saving model...")
torch.save(model.state_dict(), "weights/transformer.pt")
