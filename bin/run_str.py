import sys
import gin
import time
import argparse
from tqdm import tqdm
sys.path.append(".")

import tensorflow as tf
from dataset.scene_text_recognition.str_dataset import SceneTextRecognitionDataset
from models.str.str_models import SceneTextRecognitionModel
from engines.experiments import Experiments

from absl import logging

logging.set_verbosity(logging.INFO)

def main(args):
    print(' -' * 35)
    print('Running Experiment:')
    print(' -' * 35)
    dataset = SceneTextRecognitionDataset()
    model = SceneTextRecognitionModel()

    experiment = Experiments(dataset=dataset, model=model)
    experiment.run()

    print(' -' * 35)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)

    ap = argparse.ArgumentParser()
    ap.add_argument("-cf", "--config_file",
                    default="config/str_config.gin",
                    help="Google gin config file path")
    args = vars(ap.parse_args())
    gin.parse_config_file(args['config_file'])
    main(args)