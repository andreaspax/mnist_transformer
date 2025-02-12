import torch
# import wandb
import encoder
import tqdm
import dataset

torch.manual_seed(2)

batch_size = 256
epochs = 2
initial_lr = 0.0001


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("device:", device)

# wandb.init(
#     project="mlx6-mnist-transformer",
#     config={
#         "learning_rate": initial_lr,
#         "epochs": epochs,
#     },
# )


model = encoder.Encoder(input_dim=49, dff=1024)
model.to(device)

print("param count:", sum(p.numel() for p in model.parameters()))




#
#
#
train_dataset = dataset.MNISTDataset(train=True, single=True, total_samples=100000, seed=2)
test_dataset = dataset.MNISTDataset(train=False, single=True, total_samples=10000, seed=2)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=2)

loss_fn = torch.nn.CrossEntropyLoss()

optimiser = torch.optim.Adam(model.parameters(), lr=initial_lr)

for epoch in range(epochs):
    print(f"Epoch {epoch + 1} started")
    total_loss = 0
    num_batches = 0
    # for step in range(steps_per_epoch):
    prgs = tqdm.tqdm((train_dataloader), desc=f"Epoch {epoch+1}", leave=False)
        # print(prgs)
    for _, flattened, labels in prgs:            
        flattened, labels = flattened.to(device), labels.to(device)
        
        optimiser.zero_grad()

        encoder_output = model(flattened)
        cls_token = encoder_output[:, 0, :]
        normalised_output = torch.nn.LayerNorm(64).to(device)(cls_token)
        
        logits = torch.nn.Linear(64, 10).to(device)(normalised_output)
        
        loss = loss_fn(logits, labels)

        loss.backward()
        optimiser.step()

        total_loss += loss.item()
        num_batches += 1

        # Calculate accuracy
        _, predicted = torch.max(logits, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / labels.size(0)


    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Acc: {accuracy:.2%}")

    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for test_flattened, test_labels in test_dataloader:
            test_flattened, test_labels = test_flattened.to(device), test_labels.to(device)
            outputs = model(test_flattened)
            test_loss += loss_fn(outputs, test_labels).item()
            test_correct += (torch.argmax(outputs, 1) == test_labels).sum().item()
            test_total += test_labels.size(0)

    print(f"Test Loss: {test_loss/len(test_dataloader):.4f}, Acc: {test_correct/test_total:.2%}")
#         # wandb.log({"loss": loss.item()})

# torch.save(model.state_dict(), "weights/encoder.pt")

# # print("Uploading...")
# # artifact = wandb.Artifact("model-weights", type="encoder")
# # artifact.add_file("./weights/encoder.pt")
# # wandb.log_artifact(artifact)

# print("Done!")
# # wandb.finish()
