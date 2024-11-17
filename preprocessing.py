import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from main import N_COMPONENTS, SAMPLE_FRAC

scaler = StandardScaler()
pca = PCA(n_components=N_COMPONENTS)

def sampling(train):
    return train.sample(frac=SAMPLE_FRAC)

def scale_train(train):
    return scaler.fit_transform(train)

def scale_test(test):
    return scaler.fit(test)

def pca_train(train):
    return pca.fit_transform(train)

def pca_test(test):
    return pca.fit(test)

def add_features_train(train):
    return train

def add_features_test(test):
    return test

def load_train():
    x_train = pd.read_csv('x_train.csv', index_col='ID')
    y_train = pd.read_csv('y_train.csv', index_col='ID')
    train = pd.concat([x_train, y_train], axis=1)
    train = sampling(train)
    
    return train

def load_test():
    x_test = pd.read_csv('x_test.csv', index_col='ID')
    y_test = pd.read_csv('x_test.csv', index_col='ID')
    test = pd.concat([x_test, y_test], axis=1) 
    
    return test

def preprocessing_train(train):
    train = add_features_train(train)
    train = scale_train(train)
    train = pca_train(train)
    
    return train

def preprocessing_test(test):
    test = add_features_test(test)
    test = scale_test(test)
    test = pca_test(test)
    
    return test

