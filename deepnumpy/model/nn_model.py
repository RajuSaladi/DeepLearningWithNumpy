from deepnumpy.model._model import Model
import logging


logger = logging.getLogger(__name__)


class NN(Model):

    def __init__(self):
        super().__init__()

    def add(self, layer):
        self.layers.append(layer)

    def optimize(self, input_x, actual_y, learning_rate, loss_function):
        # Forward pass
        logger.debug("*********Forward pass**************")
        forward_value = self.predict(input_x)

        # Compute loss and first gradient
        loss_instance = loss_function(forward_value, actual_y)
        loss_value = loss_instance.forward()
        gradient = loss_instance.backward()

        # Backpropagation
        logger.debug("*********Backward pass**************")
        for i, _ in reversed(list(enumerate(self.layers))):
            logger.debug('Layer :', i, self.layers[i])
            if self.layers[i].contains_weights:
                gradient, dW, dB = self.layers[i].backward(gradient)
                self.layers[i].optimize(dW, dB, learning_rate)
            else:
                gradient = self.layers[i].backward(gradient)
        return loss_value
