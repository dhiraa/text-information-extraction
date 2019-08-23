import torch
import gin
import string

from dataset.dataset_base import TorchDataset
from dataset.scene_text_recognition.helpers import hierarchical_dataset, AlignCollate, BatchBalancedDataset, RawDataset
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

        self._train_data = train_data
        self._valid_data = valid_data
        self._select_data = select_data
        self._batch_ratio = batch_ratio
        self._batch_size = batch_size
        self._img_height = img_height
        self._img_width = img_width
        self._is_pad = is_pad
        self._data_filtering_off = data_filtering_off
        self._batch_max_length = batch_max_length
        self._character = character
        self._is_rgb = is_rgb
        self._sensitive = sensitive
        self._num_cores = num_cores
        self._total_data_usage_ratio = total_data_usage_ratio

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

    def _get_serving_dataset(self, file_or_path):
        # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
        align_collate_demo = AlignCollate(img_height=self._img_height,
                                          img_width=self._img_width,
                                          keep_ratio_with_pad=self._is_pad)
        demo_data = RawDataset(file_or_path=file_or_path,
                               is_rgb=self._is_rgb,
                               img_height=self._img_height,
                               img_width=self._img_width)
        self._demo_loader = torch.utils.data.DataLoader(
            demo_data,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=int(self._num_cores),
            collate_fn=align_collate_demo,
            pin_memory=True)

        return self._demo_loader
