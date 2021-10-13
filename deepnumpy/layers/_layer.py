class Layer:
    """Layer abstract class"""
    def __init__(self):
        self.contains_weights = True
        pass

    def __len__(self):
        pass

    def __str__(self):
        return f"{self.type} Layer"

    def forward(self):
        pass

    def backward(self):
        pass

    def optimize(self):
        pass
