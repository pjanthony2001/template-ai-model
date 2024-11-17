import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna

from preprocessing import preprocessing_train
from model import NeuralNetRegressorWithDropout
from sklearn.model_selection import train_test_split

import logging

logger = logging.getLogger(__name__)
NUM_EPOCHS_TRIALS = 10
N_TRIALS = 50
RANDOM_STATE = 230
TEST_SIZE = 0.20
class HyperParameterTesting:
    def __init__(self, x_train, y_train):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x_train, y_train, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        
    def run_trials(self):
        
        def objective(trial):
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            params = {
                "hidden_size_1" : trial.suggest_int('hidden_size_1', 32, 128, step=32),
                "hidden_size_2" : trial.suggest_int('hidden_size_2', 16, 64, step=8),
                "dropout_rate" : trial.suggest_float('dropout_rate', 0.2, 0.4, step=0.1),
                "input_size" : self.X_train.shape[1],
            }

            model = NeuralNetRegressorWithDropout(**params)
            model.fit(self.X_train, self.y_train, learning_rate, NUM_EPOCHS_TRIALS)
            loss = model.evaluate(self.X_test, self.y_test)
            return loss

        logger.info("--- START HyperParameter Testing ---")   
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=N_TRIALS)
        logger.info("--- END HyperParameter Testing ---")
        
        return dict(study.best_params)
    
