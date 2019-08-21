import os
import time

import gin
import string
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.metrics.distance import edit_distance

import numpy as np

from dataset.scene_text_recognition.str_dataset import SceneTextRecognitionDataset
from models.str.modules.transformation import TPS_SpatialTransformerNetwork
from models.str.modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from models.str.modules.sequence_modeling import BidirectionalLSTM
from models.str.modules.prediction import Attention
from dataset.scene_text_recognition.utils import AttnLabelConverter, Averager, CTCLabelConverter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@gin.configurable
class SceneTextRecognitionModel(nn.Module):

    def __init__(self,
                 transformation_stage="TPS",
                 feature_extraction_stage="ResNet",
                 sequence_modeling_stage="BiLSTM",
                 prediction_stage="Attn",
                 img_width=32,
                 img_height=100,
                 input_channel=1,
                 output_channel=512,
                 num_fiducial=20,
                 hidden_size=256,
                 is_adam=True,
                 character=None,
                 is_sensitive=True,
                 batch_size=192,
                 batch_max_length=25,
                 model_root_directory=None):
        super(SceneTextRecognitionModel, self).__init__()
        self.stages = {"transformation_stage": transformation_stage, 
                       "feature_extraction_stage": feature_extraction_stage,
                       "sequence_modeling_stage": sequence_modeling_stage, 
                       "prediction_stage": prediction_stage}

        self.is_adam = is_adam
        self.character = character
        self.batch_size = batch_size
        self.batch_max_length = batch_max_length
        self._model_root_directory = model_root_directory

        if is_sensitive:
            # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            self.character = string.printable[:-6]

        self.num_classes = len(self.character)

        """ Transformation """
        if transformation_stage == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=num_fiducial,
                I_size=(img_height, img_width),
                I_r_size=(img_height, img_width),
                I_channel_num=input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if feature_extraction_stage == 'VGG':
            self.feature_extraction_model = VGG_FeatureExtractor(input_channel, output_channel)
        elif feature_extraction_stage == 'RCNN':
            self.feature_extraction_model = RCNN_FeatureExtractor(input_channel, output_channel)
        elif feature_extraction_stage == 'ResNet':
            self.feature_extraction_model = ResNet_FeatureExtractor(input_channel, output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')

        self.FeatureExtraction_output = output_channel  # int(imgH/16-1) * 512

        self.adaptive_avg_pool_layer = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if sequence_modeling_stage == 'BiLSTM':
            self.sequence_model = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
                BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
            self.SequenceModeling_output = hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if prediction_stage == 'CTC':
            self.prediction_model = nn.Linear(self.SequenceModeling_output, self.num_classes)
        elif prediction_stage == 'Attn':
            self.prediction_model = Attention(self.SequenceModeling_output, hidden_size, self.num_classes)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages["transformation_stage"] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.feature_extraction_model(input)
        visual_feature = self.adaptive_avg_pool_layer(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages["sequence_modeling_stage"] == 'BiLSTM':
            contextual_feature = self.sequence_model(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages["prediction_stage"] == 'CTC':
            prediction = self.prediction_model(contextual_feature.contiguous())
        else:
            prediction = self.prediction_model(contextual_feature.contiguous(),
                                               text,
                                               is_train,
                                               batch_max_length=self.batch_max_length)

        return prediction

    @property
    def model_dir(self):
        """
        Returns model directory `model_root_directory`/SceneTextRecognitionModel
        :return:
        """
        return os.path.join(self._model_root_directory,
                            type(self).__name__)

    def get_loss_op(self):
        """ setup loss """
        if 'CTC' in self.stages["prediction_stage"]:
            criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
        else:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

        return criterion

    def get_cost(self, model, feature, label):
        assert (isinstance(model, SceneTextRecognitionModel))

        criterion = self.get_loss_op()

        image, text, length = feature["image"], label["text"], label["length"]
        if 'CTC' in self.stages["prediction_stage"]:
            preds = model(image, text).log_softmax(2)
            preds_size = torch.IntTensor([preds.size(1)] * self.batch_size).to(device)
            preds = preds.permute(1, 0, 2)  # to use CTCLoss format

            # To avoid ctc_loss issue, disabled cudnn for the computation of the ctc_loss
            # https://github.com/jpuigcerver/PyLaia/issues/16
            torch.backends.cudnn.enabled = False
            cost = criterion(preds, text, preds_size, length)
            torch.backends.cudnn.enabled = True

        else:
            preds = model(image, text[:, :-1])  # align with Attention.forward
            target = text[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        return cost

    def get_converter(self):
        """ model configuration """
        if 'CTC' in self.stages["prediction_stage"]:
            converter = CTCLabelConverter(self.character)
        else:
            converter = AttnLabelConverter(self.character)
        self.num_class = len(converter.character)

        return converter

    def get_optimizer(self, model):
        # assert (isinstance(model, SceneTextRecognitionModel))
        # filter that only require gradient decent
        filtered_parameters = []
        params_num = []
        for p in filter(lambda p: p.requires_grad, model.parameters()):
            filtered_parameters.append(p)
            params_num.append(np.prod(p.size()))
        print('Trainable params num : ', sum(params_num))
        # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

        # setup optimizer
        if self.is_adam:
            optimizer = optim.Adam(filtered_parameters, lr=1.0, betas=(0.9, 0.999))
        else:
            optimizer = optim.Adadelta(filtered_parameters, lr=1.0, rho=0.95, eps=1e-8)
        print("Optimizer:")
        print(optimizer)

        return optimizer


    def get_predictions(self, model, batch_size, image, labels):
        start_time = time.time()

        criterion = self.get_loss_op()
        batch_max_length = 25
        converter = self.get_converter()

        # For max length prediction
        length_for_pred = torch.IntTensor([batch_max_length] * self._dataset._batch_size).to(device)
        text_for_pred = torch.LongTensor(self._dataset._batch_size, batch_max_length + 1).fill_(0).to(device)

        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=batch_max_length)


        if 'CTC' in self.stages["prediction_stage"]:
            preds = model(image, text_for_pred).log_softmax(2)
            forward_time = time.time() - start_time

            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds = preds.permute(1, 0, 2)  # to use CTCloss format

            # To avoid ctc_loss issue, disabled cudnn for the computation of the ctc_loss
            # https://github.com/jpuigcerver/PyLaia/issues/16
            torch.backends.cudnn.enabled = False
            cost = criterion(preds, text_for_loss, preds_size, length_for_loss)
            torch.backends.cudnn.enabled = True

            # Select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            preds_str = converter.decode(preds_index.data, preds_size.data)

        else:
            preds = model(image, text_for_pred, is_train=False)
            forward_time = time.time() - start_time

            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        return forward_time, cost, preds_str

    def get_accuracy(self, preds_str, labels):
        n_correct = 0
        norm_ED = 0
        # calculate accuracy.
        for pred, gt in zip(preds_str, labels):
            if 'Attn' in self.stages["prediction_stage"]:
                pred = pred[:pred.find('[s]')]  # prune after "end of sentence" token ([s])
                gt = gt[:gt.find('[s]')]

            if pred == gt:
                n_correct += 1
            if len(gt) == 0:
                norm_ED += 1
            else:
                norm_ED += edit_distance(pred, gt) / len(gt)

        return n_correct, norm_ED