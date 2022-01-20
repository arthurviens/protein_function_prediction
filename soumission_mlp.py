import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import warnings
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import WeightedRandomSampler
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from zipfile import ZipFile

from utils import *

print(f"CUDA Available ? {torch.cuda.is_available()}")

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        if y is not None:
            self.y = y
            self.labels = True
        else:
            self.labels = False
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        x = self.X[idx, :]
        tensor = torch.from_numpy(x).float()
        if self.labels:
            class_id = self.y[idx]
            class_id = torch.tensor([class_id]).float()
            return tensor, class_id
        else:
            return tensor
        
    
class Data_Loaders():
    def __init__(self, batch_size, scaler=None, weighting=True):
        self.weighting=weighting
        if scaler is not None:
            self.scaling=True
        
        X, y, X_test, X_valid = load_data("data", test=True, valid=True)
        print("Data Loaded")

        X, X_test, X_valid = remove_hard_corrs(X, X_test, X_valid)
        
        
        if self.scaling:
            self.scaler = scaler
            X_forscaled, _ = remove_outliers(X, y, zscore_th=10, outlier_dim_number=30)
            self.scaler.fit(X_forscaled)
            X_train = self.scaler.transform(X)
            X_test = self.scaler.transform(X_test)
            X_valid = self.scaler.transform(X_valid)
            print("Data scaled")
        
        self.train_set = CustomDataset(X_train, y)
        self.test_set = CustomDataset(X_test)
        self.val_set = CustomDataset(X_valid)
        
        
        # Weighting
        if weighting:
            target_list = y
            _, counts = np.unique(target_list, return_counts=True)
            class_weights = [1 - (x / sum(counts)) for x in counts]
            class_weights = torch.tensor(class_weights).float().to(device)
            class_weights_all = class_weights[target_list]

            weighted_sampler = WeightedRandomSampler(
                weights=class_weights_all,
                num_samples=len(class_weights_all),
                replacement=True)
            
            self.train_loader = DataLoader(self.train_set, batch_size=batch_size, sampler=weighted_sampler)
            print("Using weighted train set")
        else:
            self.train_loader = DataLoader(self.train_set, batch_size=batch_size)
            
        self.test_loader = DataLoader(self.test_set, batch_size=128)
        self.val_loader = DataLoader(self.val_set, batch_size=128)

class BinaryMLP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.hidden_layer_1 = nn.Linear(
            in_features=kwargs["input_shape"], out_features=256
        )
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(p=0.1)
        
        self.hidden_layer_2 = nn.Linear(
            in_features=256, out_features=128
        )
        self.dropout2 = nn.Dropout(p=0.1)
        
        self.batchnorm2 = nn.BatchNorm1d(128)
        
        self.out_layer = nn.Linear(
            in_features=128, out_features=1
        )
        

    def forward(self, features):
        x = self.hidden_layer_1(features)
        #x = self.batchnorm1(x)
        x = self.dropout1(x)
        x = torch.relu(x)
        x = self.hidden_layer_2(x)
        #x = self.batchnorm2(x)
        x = self.dropout2(x)
        x = torch.relu(x)
        out = torch.sigmoid(self.out_layer(x))
        return out

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)
    y_pred_tag = y_pred_tag.detach().cpu().numpy().flatten()
    y_test = y_test.cpu().numpy().flatten()
    acc = balanced_accuracy_score(y_pred_tag, y_test)
    acc = np.round(acc * 100, 2)
    return acc
    


if __name__ == "__main__":
    base_size = 941

    dataset = Data_Loaders(256, scaler=StandardScaler(), weighting=True)


    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = BinaryMLP(input_shape=base_size).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # mean-squared error loss
    criterion = nn.BCELoss()

    epochs=15
    
    warnings.filterwarnings("ignore", category=UserWarning)

    ### Training loop
    for epoch in range(epochs):
        loss = 0
        epoch_acc = 0
        for batch_features, batch_labels in dataset.train_loader:
            # reshape mini-batch data to [N, 941] matrix
            # load it to the active device
            batch_features = batch_features.view(-1, base_size).to(device)
            batch_labels = batch_labels.view(-1, 1).to(device)
            
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            
            # compute reconstructions
            outputs = model(batch_features)
            
            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_labels)
            acc = binary_acc(outputs, batch_labels)
            
            # compute accumulated gradients
            train_loss.backward()
            
            # perform parameter update based on current gradients
            optimizer.step()
            
            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
            epoch_acc += acc.item()
        
        # compute the epoch training loss
        epoch_loss = loss / len(dataset.train_loader)
        epoch_acc = epoch_acc / len(dataset.train_loader)
        print(f'Epoch {epoch+0:03}: | Train Loss: {epoch_loss:.5f} | Train Acc: {epoch_acc:.3f}')
    
    warnings.filterwarnings("default", category=UserWarning)


    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in dataset.test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())
    y_test = np.array([])
    for a in y_pred_list:
        y_test = np.append(y_test, a.squeeze())
        
    y_pred_list = []
    with torch.no_grad():
        for X_batch in dataset.val_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())
    y_valid = np.array([])
    for a in y_pred_list:
        y_valid = np.append(y_valid, a.squeeze())
    
    np.savetxt("protein_test.predict", y_test, fmt="%d")
    np.savetxt("protein_valid.predict", y_valid, fmt="%d")

    zip_obj = ZipFile('submission.zip', 'w')
    zip_obj.write("protein_test.predict")
    zip_obj.write("protein_valid.predict")

    zip_obj.close()

    