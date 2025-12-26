"""
Model definitions for Two-Moon Classification Problem
All models implemented from scratch using PyTorch for fair comparison
"""

import torch
import torch.nn as nn
import pennylane as qml
from tqdm import tqdm
import numpy as np


class LogisticRegressionModel(nn.Module):
    """Logistic Regression implemented from scratch with PyTorch"""
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 2)
        
    def forward(self, x):
        return self.linear(x)


class SVMModel(nn.Module):
    """SVM with RBF kernel approximation using PyTorch"""
    def __init__(self, n_features=2, n_hidden=20):
        super(SVMModel, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DecisionTreeModel(nn.Module):
    """Decision Tree approximation using neural network"""
    def __init__(self, depth=5):
        super(DecisionTreeModel, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RandomForestModel(nn.Module):
    """Random Forest approximation using ensemble of neural networks"""
    def __init__(self, n_trees=10):
        super(RandomForestModel, self).__init__()
        self.trees = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, 8),
                nn.ReLU(),
                nn.Linear(8, 2)
            ) for _ in range(n_trees)
        ])
        
    def forward(self, x):
        outputs = torch.stack([tree(x) for tree in self.trees])
        return outputs.mean(dim=0)


class NaiveBayesModel(nn.Module):
    """Naive Bayes approximation using neural network"""
    def __init__(self):
        super(NaiveBayesModel, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class AdaBoostModel(nn.Module):
    """AdaBoost approximation using weighted ensemble"""
    def __init__(self, n_estimators=10):
        super(AdaBoostModel, self).__init__()
        self.estimators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, 4),
                nn.ReLU(),
                nn.Linear(4, 2)
            ) for _ in range(n_estimators)
        ])
        
    def forward(self, x):
        outputs = torch.stack([est(x) for est in self.estimators])
        return outputs.mean(dim=0)


def create_quantum_model(n_qubits=2, n_layers=6):
    """
    Create a hybrid quantum-classical neural network model
    
    Args:
        n_qubits: Number of qubits in the quantum circuit
        n_layers: Number of layers in the quantum circuit
        
    Returns:
        HybridModel instance
    """
    # Create quantum device
    backend = qml.device("default.qubit", wires=n_qubits)
    
    # Define quantum circuit
    @qml.qnode(backend)
    def qnode(inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
    
    # Create quantum layer
    weight_shapes = {"weights": (n_layers, n_qubits)}
    qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
    
    # Define hybrid model
    class HybridModel(nn.Module):
        def __init__(self):
            super(HybridModel, self).__init__()
            self.input_layer = nn.Linear(2, n_qubits)
            self.qlayer = qlayer
            self.output_layer = nn.Linear(n_qubits, 2)
            
        def forward(self, x):
            x = self.input_layer(x)
            x = self.qlayer(x)
            x = self.output_layer(x)
            return x
    
    return HybridModel()


class ClassicalNN(nn.Module):
    """Classical Neural Network implemented from scratch with PyTorch"""
    def __init__(self, hidden_sizes=[8, 4]):
        super(ClassicalNN, self).__init__()
        self.fc1 = nn.Linear(2, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_classical_models():
    """
    Get dictionary of classical ML models implemented from scratch
    All models are PyTorch neural networks for fair comparison
    
    Returns:
        Dictionary mapping model names to factory functions
    """
    return {
        'Logistic Regression': lambda: LogisticRegressionModel(),
        'SVM': lambda: SVMModel(n_hidden=20),
        'Decision Tree': lambda: DecisionTreeModel(depth=5),
        'Random Forest': lambda: RandomForestModel(n_trees=10),
        'Naive Bayes': lambda: NaiveBayesModel(),
        'AdaBoost': lambda: AdaBoostModel(n_estimators=10)
    }


def train_model(model, x_train, y_train, x_val, y_val,
                epochs=10, batch_size=5, lr=0.01, desc='Training Model'):
    """
    Universal training function for all PyTorch models
    
    Args:
        model: PyTorch model
        x_train, y_train: Training data (as torch tensors)
        x_val, y_val: Validation data (as torch tensors)
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        desc: Description for progress bar
        
    Returns:
        Tuple of (trained_model, history_dict)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    for epoch in tqdm(range(epochs), desc=desc, unit='epoch', leave=True):
        model.train()
        train_loss = 0
        train_correct = 0
        
        for i in range(0, len(x_train), batch_size):
            batch_x = x_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (outputs.argmax(dim=1) == batch_y).sum().item()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_correct = (val_outputs.argmax(dim=1) == y_val).sum().item()
        
        train_loss /= (len(x_train) / batch_size)
        train_acc = train_correct / len(x_train)
        val_acc = val_correct / len(x_val)
        
        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
    
    return model, history


def create_classical_nn():
    """Create a classical neural network from scratch"""
    return ClassicalNN(hidden_sizes=[8, 4])


def train_classical_nn(model, x_train, y_train, x_val, y_val,
                       epochs=10, batch_size=5, lr=0.01):
    """
    Train classical neural network model
    
    Args:
        model: ClassicalNN instance
        x_train, y_train: Training data (as torch tensors)
        x_val, y_val: Validation data (as torch tensors)
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        
    Returns:
        Tuple of (trained_model, history_dict)
    """
    return train_model(model, x_train, y_train, x_val, y_val,
                      epochs, batch_size, lr, desc='Training Classical NN')


def train_quantum_model(model, x_train, y_train, x_val, y_val, 
                       epochs=10, batch_size=5, lr=0.01):
    """
    Train quantum neural network model
    
    Args:
        model: HybridModel instance
        x_train, y_train: Training data
        x_val, y_val: Validation data
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        
    Returns:
        Tuple of (trained_model, history_dict, training_time)
    """
    import time
    start = time.time()
    model, history = train_model(model, x_train, y_train, x_val, y_val,
                                epochs, batch_size, lr, desc='Training Quantum NN')
    train_time = time.time() - start
    return model, history, train_time
