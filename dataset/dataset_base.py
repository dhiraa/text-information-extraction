
class DatasetBase(object):
    def __init__(self,
                 data_dir,
                 batch_size,
                 num_cores=4):
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._num_cores = num_cores

    def train_set(self):
        """
        Returns an data set iterator function that can be used in train loop
        :return:
        """
        return self._get_train_dataset()

    def validation_set(self):
        """
        Returns an data set iterator function that can be used in validation loop
        :return:
        """
        return self._get_val_dataset()

    def test_set(self):
        """
        Returns an data set iterator function that can be used in test loop
        :return:
        """
        return self._get_test_dataset()

    def serving_set(self, file_or_path):
        return self._get_serving_dataset(file_or_path=file_or_path)

    def _get_train_dataset(self):
        raise NotImplementedError

    def _get_val_dataset(self):
        raise NotImplementedError

    def _get_test_dataset(self):
        raise NotImplementedError

    def _get_serving_dataset(self, file_or_path):
        raise NotImplementedError

class TensorFlowDataset(DatasetBase):
    def __init__(self,
                 data_dir,
                 batch_size,
                 num_cores=4):
        DatasetBase.__init__(self,
                             data_dir=data_dir,
                             batch_size=batch_size,
                             num_cores=num_cores)


class TorchDataset(DatasetBase):
    def __init__(self,
                 data_dir,
                 batch_size,
                 num_cores=4):
        DatasetBase.__init__(self,
                             data_dir=data_dir,
                             batch_size=batch_size,
                             num_cores=num_cores)

