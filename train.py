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
from model import MNISTNet32, MNISTNet8
import numpy as np
import signal
import sys
import os

# Initialize Flask app
app = Flask(__name__)

# Define hyperparameters at the top level
BATCH_SIZE_32 = 128  # for model32
BATCH_SIZE_8 = 64   # smaller batch size for model8
NUM_EPOCHS_32 = 2    # model32 epochs
NUM_EPOCHS_8 = 1     # model8 epochs
LEARNING_RATE_ADAM = 0.001
LEARNING_RATE_SGD = 0.01  # SGD typically needs higher learning rate
MOMENTUM = 0.9  # Momentum for SGD

# Training status for both models
training_status = {
    'model32': {
        'epoch': 0,
        'train_loss': 0.0,
        'val_acc': 0.0,
        'losses': [],
        'accuracies': [],
        'confusion_matrix': None
    },
    'model8': {
        'epoch': 0,
        'train_loss': 0.0,
        'val_acc': 0.0,
        'losses': [],
        'accuracies': [],
        'confusion_matrix': None
    },
    'training_complete': False,
    'test_results': ''
}

# Data preparation
print("Loading datasets...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader_32 = DataLoader(train_dataset, batch_size=BATCH_SIZE_32, shuffle=True)
test_loader_32 = DataLoader(test_dataset, batch_size=BATCH_SIZE_32)

train_loader_8 = DataLoader(train_dataset, batch_size=BATCH_SIZE_8, shuffle=True)
test_loader_8 = DataLoader(test_dataset, batch_size=BATCH_SIZE_8)
print("Datasets loaded successfully")

# Initialize both models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model32 = MNISTNet32().to(device)
model8 = MNISTNet8().to(device)
criterion = nn.CrossEntropyLoss()
optimizer32 = optim.Adam(model32.parameters(), lr=LEARNING_RATE_ADAM)
optimizer8 = optim.SGD(model8.parameters(), 
                      lr=LEARNING_RATE_SGD,
                      momentum=MOMENTUM)

def evaluate_model(model, model_name):
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0
    # Use appropriate test loader
    test_loader = test_loader_32 if model_name == 'model32' else test_loader_8
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total

def train_models():
    print("\nStarting training sequence...")
    print(f"\nModel Configurations:")
    print("1. 32-64 Model:")
    print(f"   - Epochs: {NUM_EPOCHS_32}")
    print(f"   - Batch Size: {BATCH_SIZE_32}")
    print(f"   - Optimizer: Adam")
    print(f"   - Learning Rate: {LEARNING_RATE_ADAM}")
    
    print("\n2. 8-8 Model:")
    print(f"   - Epochs: {NUM_EPOCHS_8}")
    print(f"   - Batch Size: {BATCH_SIZE_8}")
    print(f"   - Optimizer: SGD with momentum")
    print(f"   - Learning Rate: {LEARNING_RATE_SGD}")
    
    # Train model32
    for epoch in range(NUM_EPOCHS_32):
        train_epoch(model32, optimizer32, 'model32', epoch, NUM_EPOCHS_32)
    
    # Train model8
    for epoch in range(NUM_EPOCHS_8):
        train_epoch(model8, optimizer8, 'model8', epoch, NUM_EPOCHS_8)
    
    print("\nTraining completed for both models!")
    print("\nFinal Results:")
    print("-------------")
    print(f"32-64 Model (Adam) Accuracy: {training_status['model32']['val_acc']*100:.2f}%")
    print(f"8-8 Model (SGD) Accuracy: {training_status['model8']['val_acc']*100:.2f}%")
    
    generate_comparison_results()
    training_status['training_complete'] = True

def train_epoch(model, optimizer, model_name, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Use appropriate dataloader based on model
    loader = train_loader_32 if model_name == 'model32' else train_loader_8
    
    pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{total_epochs} - {model_name}')
    print(f"\nBatch size for {model_name}: {loader.batch_size}")  # Print batch size info
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        if batch_idx % 10 == 9:
            avg_loss = running_loss / 10
            accuracy = correct / total
            training_status[model_name]['train_loss'] = avg_loss
            training_status[model_name]['losses'].append(avg_loss)
            training_status[model_name]['accuracies'].append(accuracy)
            val_acc = evaluate_model(model, model_name)
            training_status[model_name]['val_acc'] = val_acc
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.4f}',
                'val_acc': f'{val_acc:.4f}'
            })
            running_loss = 0.0
            correct = 0
            total = 0

    # Evaluate at the end of epoch
    val_acc = evaluate_model(model, model_name)
    training_status[model_name]['epoch'] = epoch + 1
    training_status[model_name]['val_acc'] = val_acc
    
    print(f"\n{model_name} Epoch {epoch+1} Summary:")
    print(f"Average Loss: {training_status[model_name]['train_loss']:.4f}")
    print(f"Validation Accuracy: {val_acc*100:.2f}%")

@app.route('/')
def home():
    return render_template('monitor.html')

@app.route('/status')
def status():
    return jsonify(training_status)

@app.route('/plot')
def plot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(training_status['model32']['losses'], 'b-', label='32-64 Model (Adam)')
    ax1.plot(training_status['model8']['losses'], 'r-', label='8-8 Model (SGD)')
    ax1.set_title('Training Loss Comparison')
    ax1.set_xlabel('Iterations (x10)')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(training_status['model32']['accuracies'], 'b-', label='32-64 Model (Adam)')
    ax2.plot(training_status['model8']['accuracies'], 'r-', label='8-8 Model (SGD)')
    ax2.set_title('Training Accuracy Comparison')
    ax2.set_xlabel('Iterations (x10)')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return send_file(buf, mimetype='image/png')

def generate_comparison_results():
    # Generate confusion matrices for both models
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    for model, name, ax in [(model32, '32-64 Model', ax1), (model8, '8-8 Model', ax2)]:
        conf_matrix = get_confusion_matrix(model)
        im = ax.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
        ax.set_title(f'Confusion Matrix - {name}')
        plt.colorbar(im, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        
        # Add text annotations
        for i in range(10):
            for j in range(10):
                ax.text(j, i, conf_matrix[i, j],
                       ha="center", va="center")
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    training_status['confusion_matrices'] = base64.b64encode(buf.getvalue()).decode()

def get_confusion_matrix(model):
    model.eval()
    conf_matrix = np.zeros((10, 10), dtype=int)
    with torch.no_grad():
        for data, target in test_loader_32:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            for t, p in zip(target, pred.cpu()):
                conf_matrix[t, p] += 1
    return conf_matrix

def signal_handler(sig, frame):
    print('\nShutting down server...')
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    print("\nMNIST CNN Training Monitor")
    print("==========================")
    print("Press Ctrl+C to stop the server")
    
    # Start training in a separate thread
    training_thread = threading.Thread(target=train_models, daemon=True)
    training_thread.start()
    
    # Start Flask server
    print("\nStarting web server...")
    print("Open http://localhost:5000 in your browser to monitor training")
    try:
        app.run(debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print('\nShutting down server...')
        os._exit(0) 