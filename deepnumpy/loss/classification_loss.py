from deepnumpy.loss._loss import Loss
import numpy as np
import logging


logger = logging.getLogger(__name__)


class BinaryCrossEntropy(Loss):

    def __init__(self, predicted_value, actual_value):
        super().__init__()
        self.actual_value = actual_value
        self.predicted_value = predicted_value
        self.type = 'Binary Cross-Entropy'
        logger.debug('Loss Function', self.type)
        logger.debug('predicted_value shape ', self.predicted_value.shape)
        logger.debug('actual_value shape ', self.actual_value.shape)

    def forward(self):
        '''
        Loss = 1/N * (-Ya * log(Yp) - (1-Ya)*log(1-Yp)))
        '''
        loss = (1/len(self.actual_value)) * np.nansum(- np.dot(self.actual_value, np.log(self.predicted_value)) - np.dot((1 - self.actual_value), np.log(1 - self.predicted_value))) # noqa
        logger.debug('loss shape ', np.squeeze(loss).shape)
        return np.squeeze(loss)

    def backward(self):
        '''
        dLoss = -1/N * (-Ya/Yp + (1-Ya)*(1/1-Yp))
        '''
        dLoss = (1/len(self.actual_value)) * (-(self.actual_value / self.predicted_value) + ((1 - self.actual_value) / (1 - self.predicted_value))) # noqa
        logger.debug('dLoss shape ', dLoss.mean(axis=1).shape)
        return dLoss.mean(axis=1)
