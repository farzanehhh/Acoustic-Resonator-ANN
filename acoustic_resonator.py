import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



def calculate_resonant_frequency(L_m, C_m):
    """
    Resonant frequency calculation from the BVD model.
    L_m: Mechanical inductance (in H)
    C_m: Mechanical capacitance (in F)
    """
    return 1 / (2 * np.pi * np.sqrt(L_m * C_m))


def calculate_anti_resonant_frequency(L_m, C_m, C_e):
    """
    Anti-resonant frequency calculation from the BVD model.
    L_m: Mechanical inductance (in H)
    C_m: Mechanical capacitance (in F)
    C_e: Electrical capacitance (in F)
    """
    return 1 / (2 * np.pi * np.sqrt(L_m * (C_m + C_e)))


def calculate_admittance(frequency, f_r, f_a, Q):
    """
    Calculate the admittance of the system based on resonant and anti-resonant frequencies.
    frequency: Operating frequency
    f_r: Resonant frequency
    f_a: Anti-resonant frequency
    Q: Quality factor
    """
    return (f_r + f_a) / (frequency * Q)


loo = LeaveOneOut()

def train_model_with_loo(input_data, target_data, epochs=100, learning_rate=0.0001, hidden_dim=64):
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(loo.split(input_data)):
        print(f"\nFold {fold + 1}")

        X_train, X_val = input_data[train_idx], input_data[val_idx]
        y_train, y_val = target_data[train_idx], target_data[val_idx]

        if np.isnan(X_train).any() or np.isnan(y_train).any():
            print("NaN detected in training data!")
            continue
        if np.isnan(X_val).any() or np.isnan(y_val).any():
            print("NaN detected in validation data!")
            continue

        X_train_new = []
        X_val_new = []

        for i in range(X_train.shape[0]):
            stiffness, mass, inductance, capacitance, electrical_capacitance = X_train[i]
            
           
            L_m = inductance  
            C_m = capacitance  
            
            
            f_r = calculate_resonant_frequency(L_m, C_m)
            f_a = calculate_anti_resonant_frequency(L_m, C_m, electrical_capacitance)
            admittance = calculate_admittance(1e6, f_r, f_a, Q=1000) 
            
        
            X_train_new.append([stiffness, mass, inductance, capacitance, electrical_capacitance, f_r, f_a, admittance])

        for i in range(X_val.shape[0]):
            stiffness, mass, inductance, capacitance, electrical_capacitance = X_val[i]
            
            L_m = inductance  
            C_m = capacitance 
            
            f_r = calculate_resonant_frequency(L_m, C_m)
            f_a = calculate_anti_resonant_frequency(L_m, C_m, electrical_capacitance)
            admittance = calculate_admittance(1e6, f_r, f_a, Q=1000) 
            
           
            X_val_new.append([stiffness, mass, inductance, capacitance, electrical_capacitance, f_r, f_a, admittance])
        
        X_train_new = torch.tensor(X_train_new, dtype=torch.float32)
        X_val_new = torch.tensor(X_val_new, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

        
        model = MLPModel(input_dim=8, hidden_dim=hidden_dim, output_dim=2) 
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_new)
            
            if torch.isnan(outputs).sum() > 0:
                print("NaN detected in model outputs during training!")
                break

            loss = criterion(outputs, y_train)
            
            if torch.isnan(loss).sum() > 0:
                print("NaN detected in loss function!")
                break

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0: 
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_new)
            val_loss = criterion(val_outputs, y_val)
            print(f"Validation Loss for fold {fold + 1}: {val_loss.item():.4f}")
            
          
            print(f"Predictions for fold {fold + 1}: {val_outputs.numpy()}")
            print(f"Actual target values for fold {fold + 1}: {y_val.numpy()}")

        fold_results.append(val_loss.item())

    avg_val_loss = np.mean(fold_results)
    print(f"\nAverage Validation Loss across all folds: {avg_val_loss:.4f}")

input_data = np.array([
    [0.5, 0.2, 1.5, 0.7, 0.8], 
    [0.3, 0.4, 1.2, 0.6, 0.7],
    [0.6, 0.3, 1.4, 0.5, 0.9]
])

target_data = np.array([
    [0.1, 0.8], 
    [0.2, 0.5],
    [0.3, 0.6]
])

train_model_with_loo(input_data, target_data, epochs=100, learning_rate=0.001, hidden_dim=64)
