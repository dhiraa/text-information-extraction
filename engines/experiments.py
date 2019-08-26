"""
Experiments class that allows easy plug n play of modules
"""
from absl import logging
import os
import shutil
import gin
import tensorflow as tf
import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from print_helper import *
from engines.tf_executor import TFExecutor
from engines.torch_executor import TorchExecutor


@gin.configurable
class Experiments(object):
    """
    Experiments uses dataset, model classes and training params to carry out Deep Learning experiments
    """

    def __init__(self,
                 name,
                 dataset,
                 model,
                 num_epochs=5,
                 num_max_steps=1000,
                 validation_interval_steps=50,
                 save_checkpoints_steps=50,
                 keep_checkpoint_max=5,
                 save_summary_steps=25,
                 log_step_count_steps=10,
                 clear_model_data=False,
                 plug_dataset=True,
                 max_steps_without_decrease=1000,
                 random_seed=42):

        """ Seed and GPU setting """
        # print("Random Seed: ", random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)

        self._name = name

        self._num_epochs = num_epochs
        self._num_max_steps = num_max_steps
        self._dataset = dataset
        self._model = model
        self.save_checkpoints_steps = save_checkpoints_steps
        self.keep_checkpoint_max = keep_checkpoint_max
        self.save_summary_steps = save_summary_steps
        self.log_step_count_steps = log_step_count_steps
        self.clear_model_data = clear_model_data
        self.plug_dataset = plug_dataset
        self.max_steps_without_decrease = max_steps_without_decrease
        self._validation_interval_steps = validation_interval_steps

    def _init_tf_config(self):
        run_config = tf.compat.v1.ConfigProto()
        run_config.gpu_options.allow_growth = True
        # run_config.gpu_options.per_process_gpu_memory_fraction = 0.50
        run_config.allow_soft_placement = True
        run_config.log_device_placement = False
        model_dir = self._model.model_dir

        if self.clear_model_data:
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)

        self._run_config = tf.estimator.RunConfig(session_config=run_config,
                                                  save_checkpoints_steps=self.save_checkpoints_steps,
                                                  keep_checkpoint_max=self.keep_checkpoint_max,
                                                  save_summary_steps=self.save_summary_steps,
                                                  model_dir=model_dir,
                                                  log_step_count_steps=self.log_step_count_steps)
        return run_config

    def test_iterator(self):
        i = 0
        for features, label in tqdm(self._dataset.train_set()):
            for key in features.keys():
                print("Batch {} =>  Shape of feature : {} is {}".format(i, key, features[key].shape))
                i = i + 1

    def run(self,
            mode="train",
            inference_file_or_path="",
            out_files_path=""):
        if mode == "test_iterator":
            self.test_iterator()

        if isinstance(self._model, nn.Module):
            executor = TorchExecutor(experiment_name=self._name,
                                     model=self._model,
                                     dataset=self._dataset,
                                     max_train_steps=self._num_max_steps,
                                     validation_interval_steps=self._validation_interval_steps)
            if mode in ["train", "retrain"]:
                executor.train(num_max_steps=self._num_max_steps)
            elif mode in ["serving"]:
                executor.predict_directory(in_path=inference_file_or_path, out_path=out_files_path)

        else:
            self._init_tf_config()
            executor = TFExecutor(experiment_name=self._name,
                                  model=self._model,
                                  dataset=self._dataset,
                                  config=self._run_config,
                                  max_steps_without_decrease=self.max_steps_without_decrease,
                                  max_train_steps=self._num_max_steps,
                                  validation_interval_steps=self._validation_interval_steps)

            if mode in ["train", "retrain"]:
                for epoch in tqdm(range(self._num_epochs), desc="epoch"):
                    executor.train_and_evaluate(max_train_steps=self._num_max_steps,
                                                eval_steps=None)
            elif mode in ["serving"]:
                executor.predict_directory(in_path=inference_file_or_path, out_path=out_files_path)

