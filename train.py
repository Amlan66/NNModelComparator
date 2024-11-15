import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Initialize model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize visualizer after Flask app
visualizer = TrainingVisualizer()

def train_model():
    num_epochs = 10
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 99:
                training_status['train_loss'] = running_loss / 100
                training_status['losses'].append(running_loss / 100)
                running_loss = 0.0
        
        # Validation
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
        
        training_status['epoch'] = epoch + 1
        training_status['val_acc'] = correct / total

    # Generate test results
    generate_test_results()
    training_status['training_complete'] = True

def generate_test_results():
    training_status['test_results'] = visualizer.create_test_results(
        model, device, test_loader
    )
    
    # Generate and save confusion matrix
    conf_matrix_buf = visualizer.create_confusion_matrix(
        model, device, test_loader
    )
    training_status['confusion_matrix'] = base64.b64encode(
        conf_matrix_buf.getvalue()
    ).decode()

@app.route('/')
def home():
    return render_template('monitor.html')

@app.route('/status')
def status():
    return jsonify(training_status)

@app.route('/plot')
def plot():
    buf = visualizer.create_loss_plot(training_status['losses'])
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    # Start training in a separate thread
    training_thread = threading.Thread(target=train_model)
    training_thread.start()
    
    # Start Flask server
    app.run(debug=False) 