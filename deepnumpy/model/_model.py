from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt # noqa
import logging


logger = logging.getLogger(__name__)


class Model:
    """Model abstract class"""
    def __init__(self):
        self.layers = []
        self.train_loss_idx = []
        self.test_loss_idx = []
        self.train_loss = []
        self.test_loss = []

    def __len__(self):
        pass

    def __str__(self):
        out_string = ""
        for i, _ in list(enumerate(self.layers)):
            if self.layers[i].contains_weights:
                out_string += f"Layer : {i} {self.layers[i]}"
                out_string += f" Weights, {self.layers[i].weights.shape}"
                out_string += f" Bias {self.layers[i].bias.shape}"
            else:
                out_string += f"Layer : {i} {self.layers[i]}"
            out_string += "\n"
        return out_string

    def split_train_test(self, X, Y, train_split_ratio=0.8, random_seed=42):
        train_end_index = int(train_split_ratio * X.shape[0])
        logging.info(f"Data splitting into {train_end_index} Train and {X.shape[0]-train_end_index} test data points") # noqa
        idx = np.random.RandomState(seed=random_seed).permutation(X.shape[0])
        train_idx, test_idx = idx[:train_end_index], idx[train_end_index:]
        return X[train_idx, :], Y[train_idx], X[test_idx, :], Y[test_idx]

    def get_loss_function(self, loss_function):
        if loss_function == 'BinaryCrossEntropy':
            from deepnumpy.loss.classification_loss import BinaryCrossEntropy  # noqa
        elif loss_function == 'MeanSquaredError':
            from deepnumpy.loss.regression_loss import MeanSquaredError  # noqa
        else:
            logger.error("Selected Loss function is not implemented."
                         "Taking default loss function 'BinaryCrossEntropy'")
            loss_function == 'BinaryCrossEntropy'

        return eval(loss_function)

    def train(self, X, Y, learning_rate=0.01,
              epochs=100, loss_function='BinaryCrossEntropy',
              metrics="ClassificationMetrics",
              train_split_ratio=0.8, random_seed=42,
              early_stopping={},
              output_dir='./output', verbose=False):

        logger.info(f"Training Specs: \n"
                    f"Learning Rate: {learning_rate} \n"
                    f"Maximum Epochs Limit: {epochs} \n"
                    f"Loss Function: {loss_function} \n"
                    f"Metrics Function: {metrics} \n"
                    f"Train Test Split Ratio Function: {train_split_ratio} \n"
                    f"Early Stopping: {early_stopping} \n")

        if early_stopping:
            self.initialize_monitoring(early_stopping)

        x_train, y_train, x_test, y_test = self.split_train_test(X, Y, train_split_ratio, random_seed) # noqa
        self.loss_func_instance = self.get_loss_function(loss_function)
        for this_epoch in tqdm(range(epochs)):
            this_loss = self.optimize(x_train, y_train, learning_rate, self.loss_func_instance) # noqa
            self.train_loss.append(this_loss)
            self.train_loss_idx.append(this_epoch)
            if verbose:
                if this_epoch % 10 == 0:
                    logger.info(f"Epoch: {this_epoch}. Training Loss: {this_loss}") # noqa
                    this_loss = self.evaluate(x_test, y_test, metrics=metrics, output_dir=output_dir) # noqa
                    self.test_loss.append(this_loss)
                    self.test_loss_idx.append(this_epoch)
                    self.plot_loss_curve(output_dir)
                    if early_stopping:
                        if self.monitor_for_early_stopping({'val_loss': this_loss}): # noqa
                            logging.info(f"Training stopped due to satisfied early stopping condition at {this_epoch} epoch") # noqa
                            return None

    def predict(self, input_x):
        '''
        Input --> Layer1 --> Act1 --> Layer2 --> Act2 --> Output
        '''
        forward_value = input_x.copy()
        for i, _ in enumerate(self.layers):
            forward_value = self.layers[i].forward(forward_value)
        return forward_value

    def initialize_monitoring(self, early_stopping):
        self.early_stopping_dict = early_stopping
        self.prev_value_dict = {'val_loss': np.inf, 'val_acc': 0}
        self.monitoring_dict = {}
        for this_key in early_stopping.keys():
            self.monitoring_dict[this_key] = 0

    def monitor_for_early_stopping(self, current_values_dict: dict):
        stop_flag = False
        for this_key in self.early_stopping_dict.keys():
            if this_key not in current_values_dict.keys():
                raise "Monitoring Value not calculated"
            if (((self.early_stopping_dict.get(this_key)[0] == 'inc') and (current_values_dict[this_key] >= self.prev_value_dict.get(this_key)))  # noqa
                or ((self.early_stopping_dict.get(this_key)[0] == 'dec') and (current_values_dict[this_key] <= self.prev_value_dict.get(this_key)))):  # noqa
                self.monitoring_dict[this_key] += 1
                logging.info("Identified Early Stopping condition. Counter incremented")  # noqa
                if self.monitoring_dict[this_key] > self.early_stopping_dict.get(this_key)[1]:  # noqa
                    return True
            else:
                self.monitoring_dict[this_key] = 0
        self.prev_value_dict = current_values_dict
        return stop_flag

    def optimize(self):
        pass

    def get_metric_function(self, metrics):
        if metrics == "ClassificationMetrics":
            from deepnumpy.metrics.classification_metrics import ClassificationMetrics  # noqa
        else:
            logger.error("Selected Loss function is not implemented."
                         "Taking default loss function 'ClassificationMetrics'")  # noqa
            metrics == 'ClassificationMetrics'
            from deepnumpy.metrics.classification_metrics import ClassificationMetrics  # noqa
        return eval(metrics)

    def evaluate(self, x_test, y_actual,
                 metrics="ClassificationMetrics", output_dir='./output'):
        y_pred = self.predict(x_test)
        test_loss = self.loss_func_instance(y_pred, y_actual)
        test_loss_value = test_loss.forward()
        logger.info(f"Validation Loss: {test_loss_value}")
        metric_instance = self.get_metric_function(metrics)(output_dir)
        metric_instance.run(y_pred, y_actual)
        return test_loss_value

    def plot_loss_curve(self, output_path):
        plt.figure(figsize=(17, 10))
        plt.title('Loss Curve : Loss vs Epochs', fontsize=20)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Loss Value', fontsize=16)
        plt.plot(self.train_loss_idx, self.train_loss, "-g", label="Train Loss")  # noqa
        plt.plot(self.test_loss_idx, self.test_loss, "-r", label="Test Loss")
        plt.legend(loc="upper right")
        plt.grid()
        plt.savefig(os.path.join(output_path, 'TrainTestLossCurve.png'))
        plt.close()
