class Metrics:
    """Metrics abstract class"""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.type = 'Metrics'

    def __len__(self):
        pass

    def __str__(self):
        pass
