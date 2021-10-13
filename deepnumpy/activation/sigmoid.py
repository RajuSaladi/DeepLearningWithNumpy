import numpy as np
from deepnumpy.activation._activation import Activation
import logging


logger = logging.getLogger(__name__)


class Sigmoid(Activation):

    def __init__(self, output_dim):
        super().__init__()
        self.units = output_dim
        self.type = 'Sigmoid'

    def forward(self, input_x):
        '''
        Sigmoid =  1/(1+exp(-x))
        '''
        logger.debug('This layer input', input_x.shape)
        self.prev_activation_output = 1 / (1 + np.exp(-1 * input_x))
        logger.debug('This layer Output', self.prev_activation_output.shape)
        return self.prev_activation_output

    def backward(self, dJ):
        '''
        Sigmoid =  1/(1+exp(-x))
        derivative of Sigmoid = exp(-x)/(1+exp(-1))^2 = Sig(x) * (1-Sig(x))
        '''
        logger.debug('dJ shape ', dJ.shape)
        sig = self.prev_activation_output
        dSigmoid = dJ * sig * (1 - sig)
        logger.debug('dSigmoid shape ', dSigmoid.mean(axis=1).shape)
        return dSigmoid.mean(axis=1)
