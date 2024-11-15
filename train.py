import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import io
import base64
from flask import Flask, render_template, jsonify, send_file
import threading
from model import MNISTNet
import numpy as np
from visualization import TrainingVisualizer

app = Flask(__name__)

# Global variables for tracking training progress
training_status = {
    'epoch': 0,
    'train_loss': 0.0,
    'val_acc': 0.0,
    'training_complete': False,
    'test_results': '',
    'losses': []
}

# Data preparation
print("Loading datasets...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)
print("Datasets loaded successfully")

# Initialize model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def evaluate_model():
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total

def train_model():
    num_epochs = 2
    print("\nStarting training...")
    print(f"Model architecture:\n{model}\n")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Create progress bar for each epoch
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Update progress bar every batch
            if batch_idx % 10 == 9:
                avg_loss = running_loss / 10
                training_status['train_loss'] = avg_loss
                training_status['losses'].append(avg_loss)
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}'
                })
                running_loss = 0.0
        
        # Evaluate after each epoch
        val_acc = evaluate_model()
        training_status['epoch'] = epoch + 1
        training_status['val_acc'] = val_acc
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Average Loss: {training_status['train_loss']:.4f}")
        print(f"Validation Accuracy: {val_acc*100:.2f}%\n")
    
    print("Training completed!")
    print("Generating test results...")
    generate_test_results()
    training_status['training_complete'] = True
    print("Test results generated. You can view them in the web interface.")

def generate_test_results():
    model.eval()
    test_images = []
    all_preds = []
    all_targets = []
    
    # Generate predictions and confusion matrix data
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.numpy())
            
            # Generate test image visualizations for first 10 samples only
            if len(test_images) < 10:
                for i in range(min(10 - len(test_images), len(data))):
                    img = data[i].cpu().numpy().squeeze()
                    pred_digit = pred[i].item()
                    true_digit = target[i].item()
                    
                    plt.figure(figsize=(2, 2))
                    plt.imshow(img, cmap='gray')
                    plt.axis('off')
                    plt.title(f'Pred: {pred_digit}\nTrue: {true_digit}')
                    
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    plt.close()
                    
                    buf.seek(0)
                    img_str = base64.b64encode(buf.getvalue()).decode()
                    test_images.append(f'<div class="test-image"><img src="data:image/png;base64,{img_str}"></div>')
    
    # Generate confusion matrix
    conf_matrix = np.zeros((10, 10), dtype=int)
    for p, t in zip(all_preds, all_targets):
        conf_matrix[t, p] += 1
    
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add labels
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(range(10))
    plt.yticks(range(10))
    
    # Add text annotations
    for i in range(10):
        for j in range(10):
            plt.text(j, i, conf_matrix[i, j],
                    ha="center", va="center")
    
    # Save confusion matrix
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    conf_matrix_str = base64.b64encode(buf.getvalue()).decode()
    
    # Update training status
    training_status['test_results'] = ''.join(test_images)
    training_status['confusion_matrix'] = conf_matrix_str

@app.route('/')
def home():
    return render_template('monitor.html')

@app.route('/status')
def status():
    return jsonify(training_status)

@app.route('/plot')
def plot():
    plt.figure(figsize=(10, 5))
    plt.plot(training_status['losses'])
    plt.title('Training Loss')
    plt.xlabel('Iterations (x10)')
    plt.ylabel('Loss')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    print("\nMNIST CNN Training Monitor")
    print("==========================")
    
    # Start training in a separate thread
    training_thread = threading.Thread(target=train_model)
    training_thread.start()
    
    # Start Flask server
    print("\nStarting web server...")
    print("Open http://localhost:5000 in your browser to monitor training")
    app.run(debug=False) 