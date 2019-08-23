import os
import sys
import re
import six
import math
import lmdb
import torch

from natsort import natsorted
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms

from print_helper import *


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


class AlignCollate(object):

    def __init__(self, img_height=32, img_width=100, keep_ratio_with_pad=False):
        self.imgH = img_height
        self.imgW = img_width
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            transform = NormalizePAD((1, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


class LmdbDataset(Dataset):

    def __init__(self,
                 root,
                 data_filtering_off,
                 batch_max_length,
                 character,
                 is_rgb,
                 img_height,
                 img_width,
                 sensitive):

        self.root = root
        self.data_filtering_off = data_filtering_off
        self.batch_max_length = batch_max_length
        self.character = character
        self.is_rgb = is_rgb
        self.img_height = img_height
        self.img_width = img_width
        self.sensitive = sensitive

        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            num_samples = int(txn.get('num-samples'.encode()))
            self.num_samples = num_samples

            if self.data_filtering_off:
                # for fast check with no filtering
                self.filtered_index_list = [index + 1 for index in range(self.num_samples)]
            else:
                # Filtering
                self.filtered_index_list = []
                for index in range(self.num_samples):
                    index += 1  # lmdb starts with 1
                    label_key = 'label-%09d'.encode() % index
                    label = txn.get(label_key).decode('utf-8')

                    if len(label) > self.batch_max_length:
                        # print(f'The length of the label is longer than max_length: length
                        # {len(label)}, {label} in dataset {self.root}')
                        continue

                    # By default, images containing characters which are not in opt.character are filtered.
                    # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                    out_of_char = f'[^{self.character}]'
                    if re.search(out_of_char, label.lower()):
                        continue

                    self.filtered_index_list.append(index)

                self.num_samples = len(self.filtered_index_list)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            try:
                if self.is_rgb:
                    img = Image.open(buf).convert('RGB')  # for color image
                else:
                    print_debug("=====================")
                    img = Image.open(buf).convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.is_rgb:
                    img = Image.new('RGB', (self.img_width, self.img_height))
                else:
                    img = Image.new('L', (self.img_width, self.img_height))
                label = '[dummy_label]'

            if not self.sensitive:
                label = label.lower()

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            out_of_char = f'[^{self.character}]'
            label = re.sub(out_of_char, '', label)

        return (img, label)

def hierarchical_dataset(root,
                         data_filtering_off,
                         batch_max_length,
                         character,
                         is_rgb,
                         img_height,
                         img_width,
                         sensitive,
                         select_data='/'):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []
    print_info(f'dataset_root:    {root}\t dataset: {select_data[0]}')

    for dirpath, dirnames, filenames in os.walk(root+'/'):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                dataset = LmdbDataset(root=dirpath,
                                      data_filtering_off=data_filtering_off,
                                      batch_max_length=batch_max_length,
                                      character=character,
                                      is_rgb=is_rgb,
                                      img_height=img_height,
                                      img_width=img_width,
                                      sensitive=sensitive)
                print_info(f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}')
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)

    return concatenated_dataset


class BatchBalancedDataset(object):
    def __init__(self,
                 train_data,
                 select_data,
                 batch_ratio,
                 batch_size,
                 img_height,
                 img_width,
                 is_pad,
                 data_filtering_off,
                 batch_max_length,
                 character,
                 is_rgb,
                 sensitive,
                 workers,
                 total_data_usage_ratio):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
        """
        print_info('-' * 80)
        print_info(f'dataset_root: {train_data}\nselect_data: {select_data}\nbatch_ratio: {batch_ratio}')
        assert len(select_data) == len(batch_ratio)

        _AlignCollate = AlignCollate(img_height=img_height, img_width=img_width, keep_ratio_with_pad=is_pad)

        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        total_batch_size = 0

        for selected_d, batch_ratio_d in zip(select_data, batch_ratio):
            print_info(selected_d)
            _batch_size = max(round(batch_size * float(batch_ratio_d)), 1)
            print('-' * 80)
            _dataset = hierarchical_dataset(root=train_data,
                                            data_filtering_off=data_filtering_off,
                                            batch_max_length=batch_max_length,
                                            character=character,
                                            is_rgb=is_rgb,
                                            img_height=img_height,
                                            img_width=img_width,
                                            sensitive=sensitive,
                                            select_data=[selected_d])

            total_number_dataset = len(_dataset)

            """
            The total number of data can be modified with opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            See 4.2 section in our paper.
            """
            number_dataset = int(total_number_dataset * float(total_data_usage_ratio))
            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            print(f'num total samples of {selected_d}: {total_number_dataset} x {total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}')
            print(f'num samples of {selected_d} per batch: {batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}')

            batch_size_list.append(str(_batch_size))
            total_batch_size += _batch_size

            _data_loader = torch.utils.data.DataLoader(
                _dataset,
                batch_size=_batch_size,
                shuffle=True,
                num_workers=int(workers),
                collate_fn=_AlignCollate, pin_memory=True)
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

        print('-' * 80)
        print('total_batch_size: ', '+'.join(batch_size_list), '=', str(total_batch_size))
        batch_size = total_batch_size
        print('-' * 80)

    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text = data_loader_iter.next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = self.dataloader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        return balanced_batch_images, balanced_batch_texts

    def __len__(self):
        return 7224586 + 5522808 #self.num_samples


class RawDataset(Dataset):

    def __init__(self,
                 file_or_path,
                 is_rgb,
                 img_height,
                 img_width):
        self.is_rgb = is_rgb
        self.img_height, self.img_width = img_height, img_width
        self.image_path_list = []
        if os.path.isdir(file_or_path):
            for dirpath, dirnames, filenames in os.walk(file_or_path):
                for name in filenames:
                    _, ext = os.path.splitext(name)
                    ext = ext.lower()
                    if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                        self.image_path_list.append(os.path.join(dirpath, name))

            self.image_path_list = natsorted(self.image_path_list)
            self.nSamples = len(self.image_path_list)
        else:
            self.image_path_list.append(file_or_path)
            self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        try:
            if self.is_rgb:
                img = Image.open(self.image_path_list[index]).convert('RGB')  # for color image
            else:
                print_error("=============================")
                img = Image.open(self.image_path_list[index]).convert('L')

        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            if self.is_rgb:
                img = Image.new('RGB', (self.img_width, self.img_height))
            else:
                img = Image.new('L', (self.img_width, self.img_height))

        return (img, self.image_path_list[index])
