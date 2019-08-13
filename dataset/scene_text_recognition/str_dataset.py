import torch
import gin
import string
from dataset.scene_text_recognition.dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset


@gin.configurable
class SceneTextRecognitionDataset(object):
    def __init__(self,
                 train_data=gin.REQUIRED,
                 valid_data=gin.REQUIRED,
                 select_data=gin.REQUIRED,
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
                 workers=4,
                 total_data_usage_ratio=1.0):

        if sensitive:
            character = string.printable[:-6]  # same with ASTER setting (use 94 char).

        self.train_dataset = Batch_Balanced_Dataset(train_data=train_data,
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
                                                    workers=workers,
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
                                             sensitive=sensitive,
                                             select_data=select_data)

        self.valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size,
            shuffle=True,  # 'True' to check training progress with validation function.
            num_workers=int(workers),
            collate_fn=align_collate_valid,
            pin_memory=True)
        print('-' * 80)

    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):
        return self.valid_loader