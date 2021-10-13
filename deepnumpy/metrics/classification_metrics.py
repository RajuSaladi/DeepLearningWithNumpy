import os
import logging
from deepnumpy.metrics._metrics import Metrics
import matplotlib.pyplot as plt
import numpy as np


logger = logging.getLogger(__name__)


class ClassificationMetrics(Metrics):

    def __init__(self, output_dir):
        super().__init__(output_dir)
        self.type = 'Classification Metrics'

    def run(self, predicted_value, actual_value):
        conf_threshold = self._get_optimal_threshold(predicted_value, actual_value) # noqa
        predicted_value = (predicted_value > conf_threshold).astype(int)
        self.true_positives = ((predicted_value == 1) & (actual_value == 1)).sum() # noqa
        self.false_positives = ((predicted_value == 1) & (actual_value == 0)).sum() # noqa
        self.true_negetives = ((predicted_value == 0) & (actual_value == 1)).sum() # noqa
        self.false_negetives = ((predicted_value == 0) & (actual_value == 0)).sum() # noqa
        self.total_sum = self.true_positives + self.true_negetives + self.false_positives + self.false_negetives # noqa
        self.accuracy = self._get_accuracy()
        self.precision = self._get_precision()
        self.recall = self._get_recall()
        self.f1_score = self._get_f1_score()
        self._get_classification_report()

    def _get_accuracy(self):
        return (self.true_positives + self.true_negetives)/self.total_sum
        # return (actual_value == predicted_value).mean()

    def _get_precision(self):
        return self.true_positives / (self.true_positives + self.false_positives) # noqa

    def _get_recall(self):
        return self.true_positives / (self.true_positives + self.false_negetives) # noqa

    def _get_f1_score(self):
        return (2 * self.precision * self.recall)/(self.precision + self.recall) # noqa

    def _confusion_matrix(self):
        out = "\n \t ******** \t Confusion Matrix \t ******** \n"
        out += "Actual '\' Prediction \t Positive \t Negetive \n"
        out += f"Positive \t {self.true_positives} \t {self.false_negetives} \n" # noqa
        out += f"Negative \t {self.false_positives} \t {self.true_negetives} \n" # noqa
        logging.info(out)

    def _get_classification_report(self):
        self._confusion_matrix()
        logging.info(f"Precison: {self.precision}")
        logging.info(f"Recall: {self.recall}")
        logging.info(f"F1 Score: {self.f1_score}")

    def _true_false_positive(self, threshold_vector, actual_value):
        true_positive = np.equal(threshold_vector, 1) & np.equal(actual_value, 1) # noqa
        true_negative = np.equal(threshold_vector, 0) & np.equal(actual_value, 0) # noqa
        false_positive = np.equal(threshold_vector, 1) & np.equal(actual_value, 0) # noqa
        false_negative = np.equal(threshold_vector, 0) & np.equal(actual_value, 1) # noqa

        tpr = true_positive.sum() / (true_positive.sum() + false_negative.sum()) # noqa
        fpr = false_positive.sum() / (false_positive.sum() + true_negative.sum()) # noqa
        return tpr, fpr

    def _calculate_roc_auc_score(self, probabilities, actual_value, num_partitions=100): # noqa
        roc_values = np.array([])
        for i in range(num_partitions + 1):
            this_threshold = i / num_partitions
            threshold_vector = np.greater_equal(probabilities, this_threshold).astype(int) # noqa
            this_tpr, this_fpr = self._true_false_positive(threshold_vector, actual_value) # noqa
            if len(roc_values):
                roc_values = np.vstack((roc_values, [this_fpr, this_tpr, this_threshold])) # noqa
            else:
                roc_values = np.array([this_fpr, this_tpr, this_threshold])
        self._plot_roc_auc_curve(roc_values)
        fpr, tpr, thresholds = roc_values[:, 0], roc_values[:, 1], roc_values[:, 2] # noqa
        total_roc_value = 0
        for k in range(num_partitions):
            total_roc_value = total_roc_value + (fpr[k] - fpr[k + 1]) * tpr[k]
        logging.info(f"ROC AUC Score: {total_roc_value}")
        return fpr, tpr, thresholds

    def _get_optimal_threshold(self, y_pred, y_actual):
        '''
        Reference:
            https://towardsdatascience.com/roc-curve-and-auc-from-scratch-in-numpy-visualized-2612bb9459ab
        '''
        num_partitions = 150
        fpr, tpr, thresholds = self._calculate_roc_auc_score(y_pred, y_actual, num_partitions=num_partitions) # noqa
        optimal_threshold = thresholds[np.argmax(tpr - fpr)]
        logging.info(f"Optimal Threshold: {optimal_threshold}")
        return optimal_threshold

    def _plot_roc_auc_curve(self, roc_value):
        plt.scatter(roc_value[:, 0], roc_value[:, 1], color='#0F9D58', s=100)
        plt.title('ROC Curve', fontsize=20)
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.grid()
        plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'))
        plt.close()
