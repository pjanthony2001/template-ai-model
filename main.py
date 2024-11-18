from hyperparam import HyperParameterTesting
from preprocessing import get_data
from model import NeuralNetRegressorWithDropout
import torch
import logging

NUM_EPOCHS_BEST = 30
logger = logging.getLogger(__name__)


logging.basicConfig(filename='model.log')


x_train, y_train, x_test, y_test = get_data()
hyperparameter = HyperParameterTesting(x_train, y_train)
best_params = hyperparameter.run_trials()

logger.info("--- START Training Best Model ---")
learning_rate = best_params.pop("learning_rate")
model = NeuralNetRegressorWithDropout(input_size=x_train.shape[1], **best_params)
model.fit(x_train, y_train)
logger.info("--- END Training Best Model ---")


logger.info("--- START Saving Best Model ---")
torch.save(model, "bestmodel.mdl")
logger.info("--- END Saving Best Model ---")

loss = model.evaluate(x_test)
logging.info(f"Best loss achieved: {loss}")
print(loss)


