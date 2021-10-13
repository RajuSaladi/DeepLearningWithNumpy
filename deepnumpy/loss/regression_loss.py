from deepnumpy.loss._loss import Loss
import numpy as np


class MeanSquaredError(Loss):

    def __init__(self, predicted_value, actual_value):
        super().__init__()
        self.predicted_value = predicted_value
        self.actual_value = actual_value
        self.type = 'Mean Squared Error'

    def forward(self):
        '''
        Loss = mean((Yp - Ya)^2)
        '''
        return np.power(self.predicted_value - self.actual_value, 2).mean()

    def backward(self):
        '''
        dLoss = 2*(Yp - Ya)
        '''
        return 2 * (self.predicted_value - self.actual_value).mean()
