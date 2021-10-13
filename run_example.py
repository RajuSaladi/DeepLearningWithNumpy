from deepnumpy.model.nn_model import NN as Model
from deepnumpy.layers.linear import Linear
from deepnumpy.activation.sigmoid import Sigmoid
from deepnumpy.activation.relu import ReLU
import os
import logging
import matplotlib
import pandas as pd
from sklearn.datasets import make_moons, make_circles


def create_and_assign_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


LEARNING_RATE = 0.01
MAX_EPOCHS = 1000
LOSS_FUNCTION = 'BinaryCrossEntropy'
TRAIN_TEST_SPLIT_RATIO = 0.8
RANDOM_SEED = 42
EARLY_STOPPING_DICT = {'val_loss': ('inc', 5)}

log_dir = create_and_assign_folder('./logs')
out_dir = create_and_assign_folder('./output')

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.basicConfig(filename=os.path.join(log_dir, 'run_example.log'),
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', # noqa
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def generate_data(samples, shape_type='circles', noise=0.05):
    # We import in the method for the shake of simplicity
    if shape_type == 'moons':
        X, Y = make_moons(n_samples=samples, noise=noise)
    elif shape_type == 'circles':
        X, Y = make_circles(n_samples=samples, noise=noise)
    else:
        raise ValueError(f"The introduced shape {shape_type} is not valid. Please use 'moons' or 'circles' ") # noqa
    data = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=Y))
    return data


def plot_generated_data(data):
    ax = data.plot.scatter(x='x', y='y', figsize=(16, 12), color=data['label'],
                           cmap=matplotlib.colors.ListedColormap(['skyblue', 'salmon']), # noqa
                           grid=True)
    return ax


data = generate_data(samples=5000, shape_type='circles', noise=0.04)
plot_generated_data(data)


X = data[['x', 'y']].values
Y = data['label'].T.values

logger.info(f"Output class Distribution: \n{data['label'].value_counts()}")
logger.info(f"Input shape: {X.shape}")
logger.info(f"output shape: {Y.shape}")

# Create model
model = Model()
model.add(Linear(2, 5))
model.add(ReLU(5))

model.add(Linear(5, 2))
model.add(ReLU(2))

model.add(Linear(2, 1))
model.add(Sigmoid(1))

logger.info(f'Model Summary: \n{model}')

logger.info('Model training started')
try:
    model.train(X, Y, learning_rate=LEARNING_RATE,
                loss_function=LOSS_FUNCTION, epochs=MAX_EPOCHS,
                train_split_ratio=TRAIN_TEST_SPLIT_RATIO,
                early_stopping=EARLY_STOPPING_DICT,
                random_seed=RANDOM_SEED,
                output_dir=out_dir, verbose=True)
    logger.info('Model trained successfully')
except KeyboardInterrupt:
    logger.warning("Model Training ended forcefully by the User")
except Exception as e:
    logger.error("Model Training ended forcefully due to ", e)
