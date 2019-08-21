class ExecutorBase(object):
    def __init__(self,
                 experiment_name,
                 model,
                 dataset,
                 data_iterator,
                 max_train_steps,
                 validation_interval_steps):
        self._experiment_name = experiment_name
        self._model = model
        self._dataset = dataset
        self._data_iterator = data_iterator
        self._max_train_steps = max_train_steps
        self._validation_interval_steps = validation_interval_steps