import os
import numpy as np
import tensorflow as tf
import gin
from models.model_base import TFModelBase

from absl import logging

from print_helper import print_info, print_warn, print_error

layers = tf.keras.layers
models = tf.keras.models
keras_utils = tf.keras.utils

# https://arxiv.org/pdf/1701.02362.pdf -> Resnet Visulization
# http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006
def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    print_warn(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> identity block")
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    print_info(input_tensor)
    # >>>>>>>>>>>>>>>>>
    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    print_info(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    print_info(x)
    x = layers.Activation('relu')(x)
    print_info(x)

    # >>>>>>>>>>>>>>>>>
    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    print_info(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    print_info(x)
    x = layers.Activation('relu')(x)
    print_info(x)

    # >>>>>>>>>>>>>>>>>
    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    print_info(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    print_info(x)
    x = layers.add([x, input_tensor])
    print_info(x)
    x = layers.Activation('relu')(x)
    print_info(x)

    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    print_warn(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> conv block")

    filters1, filters2, filters3 = filters

    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    print_info(input_tensor)
    # >>>>>>>>>>>>>>>>>
    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    print_info(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    print_info(x)
    x = layers.Activation('relu')(x)
    print_info(x)

    # >>>>>>>>>>>>>>>>>
    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    print_info(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    print_info(x)
    x = layers.Activation('relu')(x)
    print_info(x)

    # >>>>>>>>>>>>>>>>>
    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    print_info(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    print_info(x)

    # >>>>>>>>>>>>>>>>>
    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    print_info(shortcut)
    shortcut = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    print_info(shortcut)

    x = layers.add([x, shortcut])
    print_info(x)
    x = layers.Activation('relu')(x)
    print_info(x)

    return x

def unpool(inputs):
    return tf.image.resize(inputs, size=[tf.shape(inputs)[1]*2,  tf.shape(inputs)[2]*2])


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    """
    image normalization
    :param images:
    :param means:
    :return:
    """
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] = channels[i] - tf.convert_to_tensor(means[i])
    return tf.concat(axis=3, values=channels)


def model(images, text_scale=512, weight_decay=1e-5, is_training=True):
    """
    define the model, we use Keras implemention of resnet
    """
    images = mean_image_subtraction(images)

    bn_axis = 3

    end_points = dict()

    print_warn(">>>>>>>>>>>>>>> Model Definition Started: ")
    print_warn(images)
    # http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(images)
    print_warn(x)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    print_warn(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    print_warn(x)
    x = layers.Activation('relu')(x)
    print_warn(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    print_warn(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    print_warn(x)

    print_warn(">>>>>>>>>>>>>>> Resnet Definition Started: ")
    print_warn(">>>>> pool2")
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    print_warn(x)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    print_warn(x)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    print_warn(x)

    end_points["pool2"] = x

    print_warn(">>>>> pool3")
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    print_warn(x)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    print_warn(x)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    print_warn(x)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    print_warn(x)
    end_points["pool3"] = x

    print_warn(">>>>> pool4")

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    print_warn(x)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    print_warn(x)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    print_warn(x)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    print_warn(x)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    print_warn(x)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    print_info(x)
    end_points["pool4"] = x

    print_warn(">>>>> pool5")
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    print_warn(x)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    print_warn(x)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    print_warn(x)

    end_points["pool5"] = x

    f = [end_points['pool5'], end_points['pool4'],
         end_points['pool3'], end_points['pool2']]

    for i in range(4):
        logging.info('Shape of f_{} : {}'.format(i, f[i].shape))

    g = [None, None, None, None]
    h = [None, None, None, None]
    num_outputs = [None, 128, 64, 32]

    for i in range(4):
        if i == 0:
            h[i] = f[i]
        else:
            c1_1 = layers.Conv2D(filters=num_outputs[i], kernel_size=1)(tf.concat([g[i - 1], f[i]], axis=-1))
            # slim.conv2d(tf.concat([g[i-1], f[i]], axis=-1), num_outputs[i], 1)
            h[i] = layers.Conv2D(filters=num_outputs[i], kernel_size=3, padding="same")(c1_1) #TODO kernel size to 3
            # slim.conv2d(c1_1, num_outputs[i], 3)
        if i <= 2:
            g[i] = unpool(h[i])
        else:
            g[i] = layers.Conv2D(filters=num_outputs[i], kernel_size=3, padding="same")(h[i]) #TODO kernel size to 3
            # slim.conv2d(h[i], num_outputs[i], 3)
        logging.info('Shape of h_{} : {}, g_{} : {}'.format(i, h[i].shape, i, g[i].shape))

    # here we use a slightly different way for regression part,
    # we first use a sigmoid to limit the regression range, and also
    # this is do with the angle map
    F_score = layers.Conv2D(filters=1, kernel_size=1, activation=tf.nn.sigmoid)(g[3])
    # slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
    # 4 channel of axis aligned bbox and 1 channel rotation angle
    geo_map = layers.Conv2D(filters=4, kernel_size=1, activation=tf.nn.sigmoid)(g[3])
    # slim.conv2d(g[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * FLAGS.text_scale
    angle_map = layers.Conv2D(filters=1, kernel_size=1, activation=tf.nn.sigmoid)(g[3])
    # (slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2 # angle is between [-45, 45]
    F_geometry = tf.concat([geo_map, angle_map], axis=-1)

    return F_score, F_geometry


def dice_coefficient(y_true_cls, y_pred_cls):#, training_mask):
    """
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    """
    eps = 1e-5
    # intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    # union = tf.reduce_sum(y_true_cls * training_mask) + \
    #         tf.reduce_sum(y_pred_cls * training_mask) + eps
    # loss = 1. - (2 * intersection / union)

    intersection = tf.reduce_sum(y_true_cls * y_pred_cls)
    union = tf.reduce_sum(y_true_cls) + \
            tf.reduce_sum(y_pred_cls) + eps
    loss = 1. - (2 * intersection / union)

    tf.summary.scalar('classification_dice_loss', loss)
    return loss


def get_loss(y_true_cls,
             y_pred_cls,
             y_true_geo,
             y_pred_geo):
    # training_mask):
    """
    define the loss used for training, contraning two part,
    the first part we use dice loss instead of weighted logloss,
    the second part is the iou loss defined in the paper
    :param y_true_cls: ground truth of text
    :param y_pred_cls: prediction os text
    :param y_true_geo: ground truth of geometry
    :param y_pred_geo: prediction of geometry
    :param training_mask: mask used in training, to ignore some text annotated by ###
    :return:
    """

    """
    Section: EAST : 3.4.2 Loss for Geometries
      p0 : d1                  p1 : d2
       --------------------------
      |                          |
      |                          |
      |                          |
       --------------------------  
     p3 : d4                   p2 : d3
     
     where d1,d2,d3 and d4 represents the distance from a pixel to the top, right, bottom and 
     left boundary of its corresponding rectangle, respectively. 
    """

    classification_loss = dice_coefficient(y_true_cls, y_pred_cls)#, training_mask)

    # scale classification loss to match the iou loss part
    classification_loss *= 0.01

    # p0 -> top, p1->right, p2->bottom, p3->left
    p0_gt, p1_gt, p2_gt, p3_gt, theta_gt = tf.split(
        value=y_true_geo, num_or_size_splits=5, axis=3)
    p0_pred, p1_pred, p2_pred, p3_pred, theta_pred = tf.split(
        value=y_pred_geo, num_or_size_splits=5, axis=3)

    area_gt = (p0_gt + p2_gt) * (p1_gt + p3_gt)
    area_pred = (p0_pred + p2_pred) * (p1_pred + p3_pred)

    w_union = tf.minimum(p1_gt, p1_pred) + tf.minimum(p3_gt, p3_pred)
    h_union = tf.minimum(p0_gt, p0_pred) + tf.minimum(p2_gt, p2_pred)
    area_intersect = w_union * h_union

    area_union = area_gt + area_pred - area_intersect

    L_AABB = -tf.math.log((area_intersect + 1.0)/(area_union + 1.0))
    L_theta = 1 - tf.cos(theta_pred - theta_gt)
    L_g = L_AABB + 20 * L_theta

    tf.summary.scalar('geometry_AABB', tf.reduce_mean(L_AABB * y_true_cls))# * training_mask))
    tf.summary.scalar('geometry_theta', tf.reduce_mean(L_theta * y_true_cls))#* training_mask))

    # return tf.reduce_mean(L_g * y_true_cls * training_mask) + classification_loss
    return tf.reduce_mean(L_g * y_true_cls) + classification_loss


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads

# -----------------------------------------------------------------------------------------------------------

@gin.configurable
class EASTTFModel(TFModelBase):
    def __init__(self,
                 learning_rate=0.0001,
                 model_root_directory=gin.REQUIRED,
                 moving_average_decay=0.997):
        self._model_root_directory = model_root_directory
        self._learning_rate = learning_rate
        self._moving_average_decay = moving_average_decay

    def _get_optimizer(self, loss):
        tower_grads = []
        with tf.name_scope("optimizer") as scope:

            global_step=tf.compat.v1.train.get_global_step()
            learning_rate = tf.compat.v1.train.exponential_decay(self._learning_rate,
                                                                 global_step,
                                                                 decay_steps=100,
                                                                 decay_rate=0.94,
                                                                 staircase=True)
            # add summary
            tf.summary.scalar('learning_rate', learning_rate)

            optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8)

            batch_norm_updates_op = tf.group(*tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)) #TODO scope
            grads = optimizer.compute_gradients(loss)

            tower_grads.append(grads)
            grads = average_gradients(tower_grads)
            apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

            variable_averages = tf.train.ExponentialMovingAverage(self._moving_average_decay,
                                                                  global_step)
            variables_averages_op = variable_averages.apply(tf.compat.v1.trainable_variables())

            # batch norm updates
            with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
                train_op = tf.no_op(name='train_op')

        return train_op

    # def _get_optimizer(self, loss):
    #     with tf.name_scope("optimizer") as scope:
    #
    #         global_step = tf.compat.v1.train.get_global_step()
    #         learning_rate = tf.compat.v1.train.exponential_decay(self._learning_rate,
    #                                                    global_step,
    #                                                    decay_steps=100,
    #                                                    decay_rate=0.94,
    #                                                    staircase=True)
    #
    #         optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,
    #                                              beta_1=0.9,
    #                                              beta_2=0.999,
    #                                              epsilon=1e-7,
    #                                              amsgrad=False,
    #                                              name='Adam')
    #
    #         optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()
    #
    #         # Get both the unconditional updates (the None part)
    #         # and the input-conditional updates (the features part).
    #         # update_ops = model.get_updates_for(None) + model.get_updates_for(features)
    #         # Compute the minimize_op.
    #         minimize_op = optimizer.get_updates(
    #             loss,
    #             tf.compat.v1.trainable_variables())[0]
    #         train_op = tf.group(minimize_op)
    #         return train_op

    def __call__(self, features, labels, params, mode, config=None):
        """
        Used for the :tf_main:`model_fn <estimator/Estimator#__init__>`
        argument when constructing
        :tf_main:`tf.estimator.Estimator <estimator/Estimator>`.
        """
        return self._build(features, labels, params, mode, config=config)

    @property
    def model_dir(self):
        """
        Returns model directory `model_root_directory`/`experiment_name`/VanillaGAN
        :return:
        """
        return os.path.join(self._model_root_directory,
                            type(self).__name__)

    def _build(self, features, labels, params, mode, config=None):

        input_images = features['images']
        input_images = tf.convert_to_tensor(input_images)

        print_error(input_images)

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        # Build inference graph
        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=False):
            f_score, f_geometry = model(input_images, is_training=is_training)

        loss = None
        optimizer = None
        predictions = {"f_score" : f_score, "f_geometry" : f_geometry}

        if mode != tf.estimator.ModeKeys.PREDICT:
            input_score_maps = features['score_maps']
            input_geo_maps = features['geo_maps']
            # input_training_masks = features['training_masks']

            model_loss = get_loss(input_score_maps,
                                  f_score,
                                  input_geo_maps,
                                  f_geometry)#,
            # input_training_masks)
            loss = tf.add_n(
                [model_loss] + tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))

            # add summary
            logging.info(input_images)

            tf.compat.v1.summary.image('input', input_images)
            tf.compat.v1.summary.image('score_map', input_score_maps)
            tf.compat.v1.summary.image('score_map_pred', f_score * 255)
            tf.compat.v1.summary.image('geo_map_0', input_geo_maps[:, :, :, 0:1])
            tf.compat.v1.summary.image('geo_map_0_pred', f_geometry[:, :, :, 0:1])
            # tf.compat.v1.summary.image('training_masks', input_training_masks)
            tf.summary.scalar('model_loss', model_loss)
            tf.summary.scalar('total_loss', loss)

            optimizer = self._get_optimizer(loss=loss)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={'predict': tf.estimator.export.PredictOutput(predictions)},
            loss=loss,
            train_op=optimizer,
            eval_metric_ops=None)
