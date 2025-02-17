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
    plt.text(0.1, 0.5, f"Predicted sequence: {predicted_value}", fontsize=12)
    
    plt.tight_layout()
    plt.show()

def main():
    # Configuration
    device = utils.get_device()
    weights_path = "weights/transformer.pt"  # Path is now relative to script location
    
    # Initialize model and load weights
    model = load_model(weights_path, device)
    
    # Load test dataset
    test_dataset = dataset.MNISTDataset(train=False, single=False, seed=2)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    
    # Get a sample image
    combined_img, flattened, _, out_label = next(iter(test_dataloader))
    flattened = flattened.to(device)
    
    print(f"Correct label: {out_label}")

    #assume max 100 tokens (sequence length)
    max_tokens = 100
    prediction_sequence = []

    # Create start token (usually 10 for start token)
    token_seq = torch.tensor([[10]], device=device)  # Shape: [1, 1]
    
    with torch.no_grad():
        for n in range(max_tokens):
            print(f"\nCurrent sequence: {[t.item() for t in token_seq[0]]}")
            
            # Make prediction
            logits = model(token_seq, flattened)
            
            # Get prediction for next token only
            prediction = torch.argmax(logits[0, n]).item()
            prediction_sequence.append(prediction)
            print(f"Predicted so far: {prediction_sequence}")

            if prediction == 11: # End token
                break

            # Add prediction to token sequence
            token_seq = torch.cat([token_seq, torch.tensor([[prediction]], device=device)], dim=1)
            
            if token_seq.size(1) >= max_tokens:
                break
    
    prediction_sequence = [utils.vocab[t] for t in prediction_sequence]

    # Visualize results
    visualize_prediction(combined_img[0], prediction_sequence)

if __name__ == "__main__":
    main()