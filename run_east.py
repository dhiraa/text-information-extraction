import gin
import time
import argparse
from tqdm import tqdm

import tensorflow as tf
from dataset.icdar.icdar_data import ICDARTFDataset
from dataset.icdar.icdar_iterator import CIDARIterator
from models.east.east_model import EASTModel
from engines.experiments import Experiments

from absl import logging



def main(args):
    print(' -' * 35)
    print('Running Experiment:')
    print(' -' * 35)
    dataset = ICDARTFDataset()
    dataset.run()

    iterator = CIDARIterator()
    model = EASTModel()
    print(model)

    experiment = Experiments(dataset=dataset, iterator=iterator, model=model)
    experiment.run(None)

    print(' -' * 35)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)

    ap = argparse.ArgumentParser()
    ap.add_argument("-cf", "--config_file",
                    default="config/east_config.gin",
                    help="Google gin config file path")
    args = vars(ap.parse_args())
    gin.parse_config_file(args['config_file'])
    main(args)