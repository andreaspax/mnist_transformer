import torch
import dataset
import transformer
import utils
import matplotlib.pyplot as plt
import os

def load_model(weights_path, device):
    # Get absolute path to weights file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(base_dir, weights_path)
    
    # Initialize model with same parameters as training
    model = transformer.Transformer(d_model=64, dff=1024, vocab_size=12, seq_len_x=5, seq_len_y=16)
    model.load_state_dict(torch.load(weights_path))
    model.to(device)
    model.eval()
    return model

def visualize_prediction(combined_image, predicted_value):
    """Visualize the input image and prediction"""
    plt.figure(figsize=(10, 5))
    
    # Plot the input image
    plt.subplot(1, 2, 1)
    plt.imshow(combined_image.squeeze(), cmap='gray')
    plt.title('Input Image')
    plt.axis('off')
    
    # Show prediction
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.title('Prediction')
    plt.text(0.1, 0.5, f"Predicted first digit: {predicted_value}", fontsize=12)
    
    plt.tight_layout()
    plt.show()

def main():
    # Configuration
    device = utils.get_device()
    weights_path = "weights/transformer.pt"  # Path is now relative to script location
    
    # Initialize model and load weights
    model = load_model(weights_path, device)
    
    # Load test dataset
    test_dataset = dataset.MNISTDataset(train=False, single=False, seed=99)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    
    # Get a sample image
    combined_img, flattened, _, out_label = next(iter(test_dataloader))
    flattened = flattened.to(device)
    
    # Create start token (usually 10 for start token)
    start_token = torch.tensor([[10]], device=device)  # Shape: [1, 1]
    
    # Make prediction
    with torch.no_grad():
        logits = model(start_token, flattened)
        # Get prediction for first position only
        prediction = torch.argmax(logits[0, 0]).item()
    
    print(f"Correct label: {out_label}")
    print("\nPrediction Results:")
    print(f"Start token: {start_token.item()}")
    print(f"Predicted first digit: {prediction}")
    
    # Visualize results
    visualize_prediction(combined_img[0], prediction)

if __name__ == "__main__":
    main()