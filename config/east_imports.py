import gin
from dataset.icdar.icdar_data import ICDARTFDataset
from models.east.east_model import EASTModel
from engines.experiments import Experiments

@gin.configurable
def get_experiment_root_directory(value):
    return value
