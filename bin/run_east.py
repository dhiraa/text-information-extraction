#!/usr/bin/env python
# coding: utf-8
import sys
import gin
import argparse
sys.path.append(".")

from dataset.icdar.icdar_data import ICDARTFDataset
from models.east.east_model import EASTTFModel
from engines.experiments import Experiments
import tensorflow as tf
from absl import logging
logging.set_verbosity(logging.INFO)

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

def main(args):
    print(' -' * 35)
    print('Running Experiment:')
    print(' -' * 35)
    dataset = ICDARTFDataset()

    model = EASTTFModel()
    print(model)

    experiment = Experiments(name="EAST", dataset=dataset, model=model)
    experiment.run()

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