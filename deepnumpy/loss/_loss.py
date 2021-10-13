class Loss:
    """Loss abstract class"""
    def __init__(self):
        self.contains_weights = False
        pass

    def __len__(self):
        pass

    def __str__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass
