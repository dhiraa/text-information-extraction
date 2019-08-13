import gin
import torch.nn as nn

from models.str.modules.transformation import TPS_SpatialTransformerNetwork
from models.str.modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from models.str.modules.sequence_modeling import BidirectionalLSTM
from models.str.modules.prediction import Attention

@gin.configurable
class SceneTextRecognitionModel(nn.Module):

    def __init__(self,
                 transformation_stage,
                 feature_extraction_stage,
                 sequence_modeling_stage,
                 prediction_stage,
                 img_width,
                 img_height,
                 input_channel,
                 output_channel,
                 num_fiducial,
                 hidden_size,
                 num_class):
        super(SceneTextRecognitionModel, self).__init__()
        self.stages = {"transformation_stage": transformation_stage, 
                       "feature_extraction_stage": feature_extraction_stage,
                       "sequence_modeling_stage": sequence_modeling_stage, 
                       "prediction_stage": prediction_stage}

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
            self.prediction_model = nn.Linear(self.SequenceModeling_output, num_class)
        elif prediction_stage == 'Attn':
            self.prediction_model = Attention(self.SequenceModeling_output, hidden_size, num_class)
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
                                               batch_max_length=self.opt.batch_max_length)

        return prediction
