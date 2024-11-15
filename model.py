import torch.nn as nn

class MNISTNet32(nn.Module):
    def __init__(self):
        super(MNISTNet32, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # First conv layer
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv layer
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv layer
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Fourth conv layer
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x 

class MNISTNet8(nn.Module):
    def __init__(self):
        super(MNISTNet8, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # First conv layer
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv layer
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv layer
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Fourth conv layer
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 10)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x 