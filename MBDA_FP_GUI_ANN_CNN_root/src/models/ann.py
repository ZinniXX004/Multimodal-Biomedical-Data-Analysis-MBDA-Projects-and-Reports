import torch
import torch.nn as nn
import torch.nn.functional as F

class MultilayerPerceptron(nn.Module):
    """
    4-layer ANN for classification EMNIST Letters (26 classes)
    Restored exactly to match Jupyter Notebook state_dict keys.
    """
    def __init__(self, in_sz=784, out_sz=26, layers=[512, 256, 128]):
        super().__init__()
        
        self.fc1 = nn.Linear(in_sz,      layers[0])
        self.bn1 = nn.BatchNorm1d(layers[0])
        self.do1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(layers[0],  layers[1])
        self.bn2 = nn.BatchNorm1d(layers[1])
        self.do2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(layers[1],  layers[2])
        self.bn3 = nn.BatchNorm1d(layers[2])
        self.do3 = nn.Dropout(0.2)

        self.out = nn.Linear(layers[2],  out_sz)

        # Weight initialization with He (Kaiming)
        for layer in [self.fc1, self.fc2, self.fc3, self.out]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    def forward(self, X):
        # Flatten image if it comes as 2D/3D matrix from DataLoader
        if X.dim() > 2:
            X = X.view(X.size(0), -1)
            
        X = self.do1(F.relu(self.bn1(self.fc1(X))))
        X = self.do2(F.relu(self.bn2(self.fc2(X))))
        X = self.do3(F.relu(self.bn3(self.fc3(X))))
        return self.out(X)