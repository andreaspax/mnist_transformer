import torch
import dataset
import encoder
import utils
import matplotlib.pyplot as plt

def load_model(weights_path, device):
    # Initialize model with same parameters as training
    model = encoder.Classification(input_dim=49)
    model.load_state_dict(torch.load(weights_path))
    model = model.to(device)
    model.eval()
    return model


# Configuration
device = utils.get_device()
weights_path = "weights/single_encoder.pt"

# Initialize model and load weights
model = load_model(weights_path, device)
model.eval()

# Load test dataset
test_dataset = dataset.MNISTDataset(train=False, single=True, seed=4)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)


print("\nSample Predictions:")
cmb, flattened, label = next(iter(test_dataloader))
flattened = flattened.view(flattened.size(0), -1, 49)  # 7x7 patches
flattened, label = flattened.to(device), label.to(device)

# Get prediction
with torch.no_grad():
    logits = model(flattened)
    _, predicted = torch.max(logits, 1)

print("Labels:", label)
print("Predictions:", predicted)
