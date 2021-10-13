class Activation:
    """Activation abstract class"""
    def __init__(self):
        self.type = 'Activation'
        self.contains_weights = False
        pass

    def __len__(self):
        pass

    def __str__(self):
        return f"{self.type} Layer"

    def forward(self):
        pass

    def backward(self):
        pass
