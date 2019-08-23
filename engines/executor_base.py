class ExecutorBase(object):
    def __init__(self,
                 experiment_name,
                 model,
                 dataset,
                 max_train_steps,
                 validation_interval_steps,
                 stored_model):
        self._experiment_name = experiment_name
        self._model = model
        self._dataset = dataset
        self._max_train_steps = max_train_steps
        self._validation_interval_steps = validation_interval_steps
        self._stored_model = stored_model

    def predict_directory(self, in_path, out_path):
        raise NotImplementedError

    def predict_file(self, in_path, out_path):
        raise NotImplementedError
