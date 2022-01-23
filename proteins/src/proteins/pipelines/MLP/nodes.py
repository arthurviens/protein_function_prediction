"""
This is a boilerplate pipeline 'MLP'
generated using Kedro 0.17.6
"""

import numpy as np
import warnings
import logging

from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import WeightedRandomSampler

from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    roc_auc_score,
)

log = logging.getLogger(__name__)
log.info(f"CUDA Available ? {torch.cuda.is_available()}")

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


class Data_Loaders:
    def __init__(self, batch_size, X, y_train=None, weighting=True):
        if y_train is not None:
            self.train_set = CustomDataset(X, y_train)

            # Weighting
            if weighting:
                target_list = y_train
                _, counts = np.unique(target_list, return_counts=True)
                class_weights = [1 - (x / sum(counts)) for x in counts]
                class_weights = torch.tensor(class_weights).float().to(device)
                class_weights_all = class_weights[target_list]

                weighted_sampler = WeightedRandomSampler(
                    weights=class_weights_all,
                    num_samples=len(class_weights_all),
                    replacement=True,
                )

                self.train_loader = DataLoader(
                    self.train_set, batch_size=batch_size, sampler=weighted_sampler
                )
                print("Using weighted train set")
            else:
                self.train_loader = DataLoader(self.train_set, batch_size=batch_size)
        else:
            self.test_set = CustomDataset(X)
            self.test_loader = DataLoader(self.test_set, batch_size=batch_size)


class BinaryMLP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=kwargs["input_shape"], out_features=256)
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(p=0.1)

        self.hidden_layer_2 = nn.Linear(in_features=256, out_features=128)
        self.dropout2 = nn.Dropout(p=0.1)

        self.batchnorm2 = nn.BatchNorm1d(128)

        self.out_layer = nn.Linear(in_features=128, out_features=1)

    def forward(self, features):
        x = self.hidden_layer_1(features)
        x = self.batchnorm1(x)
        x = self.dropout1(x)
        x = torch.relu(x)
        x = self.hidden_layer_2(x)
        x = self.batchnorm2(x)
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


def tensors_to_numpy(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)
    y_pred_tag = y_pred_tag.detach().cpu().numpy().flatten()
    y_test = y_test.cpu().numpy().flatten()
    return y_pred, y_test


def train_model(X_train: np.array, y_train: np.array, parameters: Dict[str, Any]) -> BinaryMLP:
    """
    Node for training a SVC model given data provided to this function as the time of execution.
    """
    base_size = X_train.shape[1]

    dataset = Data_Loaders(int(parameters["batch_size"]), X_train, y_train, weighting=True)

    # load model to the specified device, either gpu or cpu
    model = BinaryMLP(input_shape=base_size).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(parameters["lr"]),
        weight_decay=float(parameters["weight_decay"]),
    )

    # mean-squared error loss
    criterion = nn.BCELoss()

    epochs = parameters["epochs"]
    log = logging.getLogger(__name__)
    warnings.filterwarnings("ignore", category=UserWarning)
    ### Training loop
    for epoch in range(epochs):
        loss = 0
        epoch_acc = 0
        for batch_features, batch_labels in dataset.train_loader:

            # reshape mini-batch data to [N, base_size] matrix
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
        log.info(
            f"Epoch {epoch+0:03}: | Train Loss: {epoch_loss:.5f} | Train Acc: {epoch_acc:.3f}"
        )

    warnings.filterwarnings("default", category=UserWarning)
    return model


def predict(model: BinaryMLP, X_test: np.array, parameters: Dict[str, Any]) -> np.array:
    """
    Node for making predictions given a pre-trained model and a test data set.
    """
    dataset = Data_Loaders(parameters["batch_size"], X_test, weighting=False)
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
    return y_test


def report_scores(y_true: np.array, y_pred: np.array) -> None:
    """
    Node for reporting the scores of the predictions performed by previous node.
    """
    target = np.where(y_pred > 0.5, 1, 0)
    log = logging.getLogger(__name__)
    log.info("Model 'MLP' AUC : {}".format(roc_auc_score(y_true, y_pred)))
    log.info("Model 'MLP' accuracy : {}".format(accuracy_score(y_true, target)))
    log.info("Model 'MLP' precision : {}".format(precision_score(y_true, target)))
    log.info(
        "Model 'MLP' balanced accuracy : {}".format(balanced_accuracy_score(y_true, target))
    )
