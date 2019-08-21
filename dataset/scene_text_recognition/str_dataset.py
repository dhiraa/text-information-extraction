import torch
import gin
import string

from dataset.dataset_base import TorchDataset
from dataset.scene_text_recognition.helpers import hierarchical_dataset, AlignCollate, BatchBalancedDataset
from absl import logging
from print_helper import *

@gin.configurable
class SceneTextRecognitionDataset(TorchDataset):
    def __init__(self,
                 train_data=None,
                 valid_data=None,
                 select_data="MJ-ST",
                 batch_ratio="0.5-0.5",
                 batch_size=192,
                 img_height=100,
                 img_width=32,
                 is_pad=True,
                 data_filtering_off=True,
                 batch_max_length=25,
                 character="0123456789abcdefghijklmnopqrstuvwxyz",
                 is_rgb=True,
                 sensitive=True,
                 num_cores=4,
                 total_data_usage_ratio=1.0):

        TorchDataset.__init__(self,
                              data_dir=None,
                              batch_size=batch_size,
                              num_cores=num_cores)

        if sensitive:
            character = string.printable[:-6]  # same with ASTER setting (use 94 char).

        select_data = select_data.split('-')
        batch_ratio = batch_ratio.split('-')

        self._batch_size = batch_size

        self.train_dataset = BatchBalancedDataset(train_data=train_data,
                                                  select_data=select_data,
                                                  batch_ratio=batch_ratio,
                                                  batch_size=batch_size,
                                                  img_height=img_height,
                                                  img_width=img_width,
                                                  is_pad=is_pad,
                                                  data_filtering_off=data_filtering_off,
                                                  batch_max_length=batch_max_length,
                                                  character=character,
                                                  is_rgb=is_rgb,
                                                  sensitive=sensitive,
                                                  workers=num_cores,
                                                  total_data_usage_ratio=total_data_usage_ratio)

        align_collate_valid = AlignCollate(img_height=img_height,
                                           img_width=img_width,
                                           keep_ratio_with_pad=is_pad)

        valid_dataset = hierarchical_dataset(root=valid_data,
                                             data_filtering_off=data_filtering_off,
                                             batch_max_length=batch_max_length,
                                             character=character,
                                             is_rgb=is_rgb,
                                             img_height=img_height,
                                             img_width=img_width,
                                             sensitive=sensitive)
        # select_data=select_data)

        self.valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size,
            shuffle=True,  # 'True' to check training progress with validation function.
            num_workers=int(num_cores),
            collate_fn=align_collate_valid,
            pin_memory=True)
        print('-' * 80)

    def __len__(self):
        return len(self.train_dataset)

    def _get_train_dataset(self):
        return self.train_dataset

    def _get_val_dataset(self):
        return self.valid_loader

    def _get_test_dataset(self):
        raise NotImplementedError
