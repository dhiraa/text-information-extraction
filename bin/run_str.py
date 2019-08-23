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
    experiment.run(mode=args["mode"],
                   inference_file_or_path=args["in_image_files_path"],
                   out_files_path=args["out_files_path"])

    print(' -' * 35)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)

    ap = argparse.ArgumentParser()
    ap.add_argument("-cf", "--config_file",
                    default="config/str_config.gin",
                    help="Google gin config file path")
    ap.add_argument("-smp", "--stored_model_path",
                    default="",
                    help="Pre-trained model path")
    ap.add_argument("-m", "--mode",
                    default="train",
                    help="[train/retrain/serving]")
    ap.add_argument("-ifp", "--in_image_files_path",
                    default="",
                    help="Directory of files to run the inference on")

    ap.add_argument("-ofp", "--out_files_path",
                    default="",
                    help="Directory to store the inference results")

    args = vars(ap.parse_args())
    gin.parse_config_file(args['config_file'])
    main(args)
