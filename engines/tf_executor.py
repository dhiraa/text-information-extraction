# Copyright 2018 The Shabda Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A class that executes training, evaluation, prediction, export of estimators.
"""

import gin
import time
import tensorflow as tf
import os
from tqdm import tqdm
from absl import logging

# pylint: disable=too-many-instance-attributes, too-many-arguments
from engines.executor_base import ExecutorBase
from print_helper import print_error

__all__ = [
    "TFExecutor"
]


class TFExecutor(ExecutorBase):
    """Class that executes training, evaluation, prediction, export, and other
    actions of :tf_main:`tf.estimator.Estimator <estimator/Estimator>`.

    Args:
        model: An instance of a subclass of
            :class:`~shabda.models.model_base.ModelBase`.
        data_hparams: A `dict` or an instance of :class:`~shabda.hparams.HParams`
            containing the hyperparameters of data. It must contain `train`
            and/or `eval` fields for relevant processes. For example, for
            :meth:`train_and_evaluate`, both fields are required.
        config: An instance of
            :tf_main:`tf.estimator.RunConfig <estimator/RunConfig>`, used as
            the :attr:`config` argument of
            :tf_main:`Estimator <estimator/Estimator#__init__>`.
        model_hparams (optional): A `dict` or an instance of
            :class:`~shabda.hparams.HParams` containing the hyperparameters of
            the model. If `None`, uses :attr:`model.hparams`. Used as
            the :attr:`params` argument of
            :tf_main:`Estimator <estimator/Estimator#__init__>`.
        train_hooks (optional): Iterable of :tf_main:`tf.train.SessionRunHook <train/SessionRunHook>`
            objects to run during training.
        eval_hooks (optional): Iterable of :tf_main:`tf.train.SessionRunHook <train/SessionRunHook>`
            objects to run during evaluation.
        session_config (optional): An instance of
            :tf_main:`tf.ConfigProto <ConfigProto>`, used as the :attr:`config`
            argument of :tf_main:`tf session <Session>`.

    Example:

        .. code-block:: python

            TODO

    See `bin/train_old.py` for the usage in detail.
    """

    def __init__(self,
                 experiment_name,
                 model,
                 dataset,
                 config,
                 max_train_steps,
                 validation_interval_steps,
                 stored_model="",
                 max_steps_without_decrease=1000,
                 train_hooks=None,
                 eval_hooks=None,
                 session_config=None):

        ExecutorBase.__init__(self,
                              experiment_name=experiment_name,
                              model=model,
                              dataset=dataset,
                              max_train_steps=max_train_steps,
                              validation_interval_steps=validation_interval_steps,
                              stored_model=stored_model)

        self._experiment_name = experiment_name
        self._model = model
        self._config = config
        self.dataset = dataset
        self._train_hooks = train_hooks
        self._eval_hooks = eval_hooks
        self._session_config = session_config

        self._estimator = tf.estimator.Estimator(model_fn=self._model, config=config, params=None)

        hook = tf.estimator.experimental.stop_if_no_decrease_hook(self._estimator,
                                                                  "loss",
                                                                  max_steps_without_decrease=max_steps_without_decrease)

        if self._train_hooks is None:
            self._train_hooks = [hook]
        else:
            self._train_hooks.append(hook)

    @property
    def model(self):
        return self._model

    @property
    def estimator(self):
        return self._estimator

    @property
    def data_iterator(self):
        return self.dataset

    def _get_train_spec(self, max_steps=None):
        # Estimators expect an input_fn to take no arguments.
        # To work around this restriction, we use lambda to capture the arguments and provide the expected interface.
        return tf.estimator.TrainSpec(
            input_fn=lambda: self.dataset.train_set(),
            max_steps=max_steps,
            hooks=self._train_hooks)

    def _get_eval_spec(self, steps):
        return tf.estimator.EvalSpec(
            input_fn=lambda: self.dataset.validation_set(),
            steps=steps,
            hooks=self._eval_hooks)

    def train(self, max_steps=None):
        """
        Trains the model. See :tf_main:`tf.estimator.Estimator.train
        <estimator/Estimator#train>` for more details.

        Args:
            max_steps (int, optional): Total number of steps for which
                to train model. If `None`, train forever or until the train
                data generates the OutOfRange exception. If OutOfRange occurs
                in the middle, training stops before :attr:`max_steps` steps.
        """
        self.train_spec = self._get_train_spec(max_steps=max_steps)
        self._estimator.train(
            input_fn=self.train_spec.input_fn,
            hooks=self.train_spec.hooks,
            max_steps=self.train_spec.max_steps)

    def evaluate(self, steps=None, checkpoint_path=None):
        """
        Evaluates the model. See :tf_main:`tf.estimator.Estimator.evaluate
        <estimator/Estimator#evaluate>` for more details.

        Args:
            steps (int, optional): Number of steps for which to evaluate
                model. If `None`, evaluates until the eval data raises an
                OutOfRange exception.
            checkpoint_path (str, optional): Path of a specific checkpoint to
                evaluate. If `None`, the the latest checkpoint in
                :attr:`config.model_dir` is used. If there are no checkpoints
                in :attr:`model_dir`, evaluation is run with newly initialized
                variables instead of restored from checkpoint.
        """
        eval_spec = self._get_eval_spec(steps=steps)
        self._estimator.evaluate(
            input_fn=eval_spec.input_fn,
            steps=eval_spec.steps,
            hooks=eval_spec.hooks,
            checkpoint_path=checkpoint_path)

    def train_and_evaluate(self, max_train_steps=None, eval_steps=None):
        """
        Trains and evaluates the model. See
        :tf_main:`tf.estimator.train_and_evaluate
        <estimator/train_and_evaluate>` for more details.

        Args:
            max_train_steps (int, optional): Total number of steps for which
                to train model. If `None`, train forever or until the train
                data generates the OutOfRange exception. If OutOfRange occurs
                in the middle, training stops before :attr:`max_steps` steps.
            eval_steps (int, optional): Number of steps for which to evaluate
                model. If `None`, evaluates until the eval data raises an
                OutOfRange exception.
        """
        train_spec = self._get_train_spec(max_steps=max_train_steps)
        eval_spec = self._get_eval_spec(steps=eval_steps)
        tf.estimator.train_and_evaluate(self._estimator, train_spec, eval_spec)

    def export_model(self, model_export_path):
        logging.info("Saving model to =======> {}".format(model_export_path))
        if not os.path.exists(model_export_path):
            os.makedirs(model_export_path)
        self._estimator.export_saved_model(
            model_export_path,
            serving_input_receiver_fn=self.dataset.serving_input_receiver_fn)
