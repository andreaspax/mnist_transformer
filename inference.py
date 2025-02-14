import torch
import dataset
import encoder
import utils
import matplotlib.pyplot as plt

def load_model(weights_path, device):
    # Initialize model with same parameters as training
    model = encoder.Classification(input_dim=196, dff=1024, d_model=64, seq_len=16)
    model.load_state_dict(torch.load(weights_path))
    model = model.to(device)
    model.eval()
    return model

correct = 0
total = 0

# Configuration
device = utils.get_device()
weights_path = "weights/2x2_encoder.pt"

# Initialize model and load weights
model = load_model(weights_path, device)

# Load test dataset
test_dataset = dataset.MNISTDataset(train=False, single=False, seed=4)  
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

print("\nSample Predictions:")
cmb, flattened, input_seq, target_seq = next(iter(test_dataloader))
flattened = flattened.view(flattened.size(0), -1, 196)
flattened, target_seq = flattened.to(device), target_seq.to(device)

# Get prediction
with torch.no_grad():
    logits = model(flattened)
    _, predicted = torch.max(logits, 1)

print("Labels:", target_seq)
print("Predictions:", predicted)

# Calculate accuracy
correct += (predicted == target_seq).sum().item()
total += target_seq.size(0)
accuracy = correct / total
print(f"Accuracy: {accuracy:.2%}")
