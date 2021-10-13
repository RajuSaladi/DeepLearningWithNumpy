import numpy as np
from deepnumpy.activation._activation import Activation
import logging


logger = logging.getLogger(__name__)


class ReLU(Activation):

    def __init__(self, output_dim):
        super().__init__()
        self.units = output_dim
        self.type = 'ReLU'

    def forward(self, input_x):
        self.prev_activation_output = np.maximum(0, input_x)
        return self.prev_activation_output

    def backward(self, dJ):
        logger.debug('dJ shape ', dJ.shape)
        logger.debug('prev_activation_output shape', self.prev_activation_output.shape) # noqa

        out_grad = (np.heaviside(self.prev_activation_output, 0) * dJ.mean(axis=1)).T # noqa
        logger.debug('Output grad shape', out_grad.shape)
        return out_grad
