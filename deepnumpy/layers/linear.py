from deepnumpy.layers._layer import Layer
import numpy as np
import logging


logger = logging.getLogger(__name__)


class Linear(Layer):

    '''
    Loss = Sum((Ya - Yp)^2)
    Yp = Sigmoid(y)
    y = wx + b
    dL/dW = 2*(Ya-Yp) * dYp/dW
    dYp/dW = dS/dy * dy/dw
    dy/dw = d(w* prev_layer_activation_output + b)/dw = prev_layer_activation_output # noqa
    dy/db = 1
    '''

    def __init__(self, n_inp, n_out):
        super().__init__()
        self.type = 'Linear'
        self.weights = np.random.rand(n_inp, n_out)
        self.bias = np.random.rand(n_out, 1)

    def forward(self, input_x):
        '''
        y = wx + b
        '''
        self.prev_layer_activation_output = input_x
        logger.debug('This layer input', input_x.shape)
        logger.debug('This layer Weights', self.weights.shape)
        logger.debug('This layer Bias', self.bias.shape)
        return np.add(np.matmul(input_x, self.weights), self.bias.reshape(-1))

    def backward(self, d_activation):
        '''
        dYp/dw = dS/dy * dy/dw = d_activation * prev_layer_activation_output
        dYp/db = d_activation
        '''
        logger.debug('d_activation Shape', d_activation.shape)
        logger.debug('prev_layer_activation_output Shape', self.prev_layer_activation_output.shape) # noqa
        if len(d_activation.shape) <= 1:
            d_w = np.dot(self.prev_layer_activation_output.T,
                         d_activation.reshape(-1, 1))
        else:
            d_w = np.dot(self.prev_layer_activation_output.T, d_activation.T)
        logger.debug('This layer Weights Shape', self.weights.shape)
        logger.debug('d_w shape', d_w.shape)

        if len(d_activation.shape) <= 1:
            d_b = d_activation.mean(axis=0, keepdims=True)
        else:
            d_b = d_activation.mean(axis=1, keepdims=True)
        logger.debug('This layer Bias Shape', self.bias.shape)
        logger.debug('d_b shape', d_b.shape)
        delta = np.dot(self.weights, d_activation.mean())
        return delta, d_w, d_b

    def optimize(self, d_w, d_b, learning_rate):
        logger.debug('This layer Weights before optimizing', self.weights.shape) # noqa
        logger.debug('This layer Bias before optimizing', self.bias.shape)
        self.weights = self.weights - learning_rate * d_w
        self.bias = self.bias - learning_rate * d_b
        logger.debug('This layer Weights after optimizing', self.weights.shape)
        logger.debug('This layer Bias after optimizing', self.bias.shape)
