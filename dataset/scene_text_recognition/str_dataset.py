import torch
import gin
from dataset.scene_text_recognition.dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset


@gin.configurable
class SceneTextRecognitionDataset(object):
    def __init__(self,
                 train_data,
                 valid_data,
                 select_data,
                 batch_ratio,
                 batch_size,
                 img_height,
                 img_width,
                 is_pad,
                 data_filtering_off,
                 num_samples,
                 batch_max_length,
                 character,
                 is_rgb,
                 sensitive,
                 workers):
        train_dataset = Batch_Balanced_Dataset(train_data=train_data,
                 select_data=select_data,
                 batch_ratio=batch_ratio,
                 batch_size=batch_size,
                 img_height=img_height,
                 img_width=img_width,
                 is_pad=is_pad,
                 data_filtering_off=data_filtering_off,
                 num_samples=num_samples,
                 batch_max_length=batch_max_length,
                 character=character,
                 is_rgb=is_rgb,
                 sensitive=sensitive)

        align_collate_valid = AlignCollate(img_height=img_height, img_width=img_width, keep_ratio_with_pad=opt.PAD)
        valid_dataset = hierarchical_dataset(root=valid_data,
                         data_filtering_off=data_filtering_off,
                         num_samples=num_samples,
                         batch_max_length=batch_max_length,
                         character=character,
                         is_rgb=is_rgb,
                         img_height=img_height,
                         img_width=img_width,
                         sensitive=sensitive,
                         select_data=select_data)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size,
            shuffle=True,  # 'True' to check training progress with validation function.
            num_workers=int(workers),
            collate_fn=align_collate_valid,
            pin_memory=True)
        print('-' * 80)