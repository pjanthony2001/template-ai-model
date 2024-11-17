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

import logging

N_COMPONENTS = 123
SAMPLE_FRAC = 0.32
X_TRAIN_PATH = "x_train.csv"
Y_TRAIN_PATH = "y_train.csv"
X_TEST_PATH = "x_test.csv"
Y_TEST_PATH = "y_test.csv"

logger = logging.getLogger(__name__)


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

def load_train(x_path, y_path):
    logger.info("--- START Loading Training Data ---")
    x_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)
    train = pd.concat([x_train, y_train], axis=1)
    train = sampling(train)
    
    x_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
    logger.info("END Loading Training Data")
    return x_train, y_train 

def load_test(x_path, y_path):
    logger.info("--- START Loading Test Data ---")
    x_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)
    test = pd.concat([x_test, y_test], axis=1)
    
    x_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]
    logger.info("--- END Loading Test Data ---")
    return x_test, y_test

def preprocessing_train(x_train, y_train):
    logger.info("--- START Preprocessing Training Data ---")
    x_train = add_features_train(x_train)
    x_train = scale_train(x_train)
    x_train = pca_train(x_train)
    
    logger.info("--- END Preprocessing Training Data ---")
    return x_train, y_train

def preprocessing_test(x_test, y_test):
    logger.info("--- START Preprocessing Test Data ---")
    x_test = add_features_test(x_test)
    x_test = scale_test(x_test)
    x_test = pca_test(x_test)
    logger.info("--- END Preprocessing Test Data ---")   
    return x_test, y_test

def get_data():
    x_train, y_train = load_train()
    x_train, y_train = preprocessing_train(x_train, y_train)
    
    x_test, y_test = load_test()
    x_test, y_test = preprocessing_test(x_test, y_test)
    
    logger.info("--- END Preprocessing Test Data ---")   
    return x_train, y_train, x_test, y_test