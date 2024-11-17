import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class NeuralNetRegressorWithDropout(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, dropout_rate):
        super(NeuralNetRegressorWithDropout, self).__init__()
        
        # Define layers with customizable hidden size and dropout
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_1)
        self.fc3 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc4 = nn.Linear(hidden_size_2, hidden_size_2)
        self.fc5 = nn.Linear(hidden_size_2, hidden_size_2 // 2)
        self.fc6 = nn.Linear(hidden_size_2 // 2, hidden_size_2 // 4)
        
        self.dropout_1 = nn.Dropout(p=dropout_rate)
        self.dropout_2 = nn.Dropout(p=dropout_rate)
        self.dropout_3 = nn.Dropout(p=dropout_rate)
        
        self.output = nn.Linear(hidden_size_2 // 4, 2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout_1(x)
        
        x = F.relu(self.fc2(x))
        
        x = F.relu(self.fc3(x))
        x = self.dropout_2(x)
        
        x = F.relu(self.fc4(x))
        
        x = F.relu(self.fc5(x))
        x = self.dropout_3(x)
        
        x = F.relu(self.fc6(x))
                
        x = F.softmax(self.output(x), dim=1)
        return x
    
    def fit(self, X, y, learning_rate):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)
        
        # Training loop
        self.train()
        for epoch in range(10):  # Reduce epochs for faster testing
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                predictions = self(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                
    def predict(self, X_test, y_test):
        criterion = nn.CrossEntropyLoss()
        
        self.eval()
        with torch.no_grad():
            val_predictions = self(X_test)
            val_loss = criterion(val_predictions, y_test).item()
            
        return val_loss
        
