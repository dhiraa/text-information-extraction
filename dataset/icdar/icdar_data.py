#!/usr/bin/env python
# coding: utf-8
import glob
import multiprocessing

from dataset.dataset_base import TensorFlowDataset
from dataset.icdar.icdar_utils import *



########## Tensorflow Feature preparing routines ###########

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _mat_feature(mat):
    return tf.train.Feature(float_list=tf.train.FloatList(value=mat.flatten()))

########## Tensorflow Feature preparing routines ###########


def get_tf_records_count(files):
    total_records = 0
    for file in tqdm(files, desc="tfrecords size: "):
        total_records += sum(1 for _ in tf.data.TFRecordDataset(file))
    return total_records


@gin.configurable
class ICDARTFDataset(TensorFlowDataset):
    """
    Reads ICDAR 2019 dataset which is organized as train/val/test folder which contains image and
    text files with polygon co-ordinates
    References:
        https://www.geeksforgeeks.org/multiprocessing-python-set-1/
        https://www.geeksforgeeks.org/multiprocessing-python-set-2/
    """
    def __init__(self,
                 data_dir=gin.REQUIRED,
                 out_dir=gin.REQUIRED,
                 max_image_large_side=1280,
                 max_text_size=800,
                 min_text_size=5,
                 min_crop_side_ratio=0.1,
                 geometry="RBOX",
                 number_images_per_tfrecords=8,
                 num_cores=4,
                 batch_size=16,
                 prefetch_size=16):
        """

        :param data_dir:
        :param out_dir:
        :param max_image_large_side:
        :param max_text_size:
        :param min_text_size:
        :param min_crop_side_ratio:
        :param geometry:
        :param number_images_per_tfrecords:
        :param num_cores: Not used as of now
        :param batch_size:
        :param prefetch_size:
        """

        TensorFlowDataset.__init__(self,
                                   data_dir=data_dir,
                                   batch_size=batch_size,
                                   num_cores=num_cores)

        self._data_dir = data_dir

        self._train_out_dir = out_dir + "/train/"
        self._val_out_dir = out_dir + "/val/"
        self._test_out_dir = out_dir + "/test/"

        make_dirs(self._train_out_dir)
        make_dirs(self._val_out_dir)
        make_dirs(self._test_out_dir)

        self._geometry = geometry
        self._min_text_size = min_text_size
        self._max_image_large_side = max_image_large_side
        self._max_text_size = max_text_size
        self._min_crop_side_ratio = min_crop_side_ratio
        self._number_images_per_tfrecords = number_images_per_tfrecords

        self.preprocess()

        self._data_dir = data_dir
        self._num_cores = num_cores
        self._batch_size = batch_size
        self._prefetch_size = prefetch_size

        self._num_train_examples = 0

        # TODO find a right way to get this
        files = glob.glob(os.path.join(self._data_dir, "train/*.tfrecords"))
        self._num_train_examples = get_tf_records_count(files=files)

    def _get_features(self, image_mat, score_map_mat, geo_map_mat):#, training_masks_mat):
        """
        Given different features matrices, this routine wraps the matrices as TF features
        """
        return {
            "images": _mat_feature(image_mat),
            "score_maps": _mat_feature(score_map_mat),
            "geo_maps": _mat_feature(geo_map_mat),
            # "training_masks": _mat_feature(training_masks_mat)
        }

    def write_tf_records(self, images, file_path_name):
        """
        Uses sub routine to create TF records files from list of images and corresponding
        text files with text polygon regions (inffered from image file names)
        :param images: List of image files
        :param file_path_name: TF record file path
        :return:
        """
        num_of_files_skipped = 0

        if os.path.exists(file_path_name):
            num_records = get_tf_records_count([file_path_name])
            print("Found ", file_path_name, f"with {num_records} records already! Hence skipping")
            return

        with tf.io.TFRecordWriter(file_path_name) as writer:
            for image_file in tqdm(images, desc="pid : " + str(os.getpid())):
                ret = image_2_data(image_file_path=image_file,
                                   geometry=self._geometry,
                                   min_text_size=self._min_text_size,
                                   min_crop_side_ratio=self._min_crop_side_ratio)
                try:
                    # image_mat, score_map_mat, geo_map_mat, training_masks_mat = ret
                    image_mat, score_map_mat, geo_map_mat = ret
                except:
                    num_of_files_skipped += 1
                    print("Number of files skipped : ", num_of_files_skipped)
                    continue
                features = tf.train.Features(
                    feature=self._get_features(image_mat, score_map_mat, geo_map_mat))#, training_masks_mat))
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())

    def task(self, images_out_path_tuple):
        self.write_tf_records(images=images_out_path_tuple[0], file_path_name=images_out_path_tuple[1])

    def prepare_data(self, data_path, out_path):

        print("Serializing data found in ", data_path)

        images = get_images(data_path)

        index = 0
        multiprocess_list = [] #list of tuples: list of images and a TFRecord file name

        for i in tqdm(range(0, len(images), self._number_images_per_tfrecords), desc="prepare_data: "):
            multiprocess_list.append((images[i:i + self._number_images_per_tfrecords],
                                      out_path + "/" + str(index) + ".tfrecords"))
            index += 1

        # creating a pool object
        pool = multiprocessing.Pool()

        # map list to target function
        pool.map(self.task, multiprocess_list)

        pool.close()
        pool.join()

    def preprocess(self):
        self.prepare_data(data_path=self._data_dir + "/train/", out_path=self._train_out_dir)
        self.prepare_data(data_path=self._data_dir + "/val/", out_path=self._val_out_dir)
        self.prepare_data(data_path=self._data_dir + "/test/", out_path=self._test_out_dir)

    @property
    def num_train_examples(self):
        return self._num_train_examples

    def dataset_to_iterator(self, dataset):
        # Create an iterator
        iterator = dataset.make_one_shot_iterator()

        # Create your tf representation of the iterator
        image, label = iterator.get_next()

        # # Bring your picture back in shape
        # image = tf.reshape(image, [-1, 256, 256, 1])
        #
        # # Create a one hot array for your labels
        # label = tf.one_hot(label, NUM_CLASSES)

        return image, label

    def decode(self, serialized_example):
        # 1. define a parser
        features = tf.io.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'images': tf.io.FixedLenFeature([512 * 512 * 3], tf.float32),
                'score_maps': tf.io.FixedLenFeature([128 * 128 * 1], tf.float32),
                'geo_maps': tf.io.FixedLenFeature([128 * 128 * 5], tf.float32),
                # 'training_masks': tf.io.FixedLenFeature([128 * 128 * 1], tf.float32),
            })

        image = tf.reshape(
            tf.cast(features['images'], tf.float32), shape=[512, 512, 3])
        score_map = tf.reshape(
            tf.cast(features['score_maps'], tf.float32), shape=[128, 128, 1])
        geo_map = tf.reshape(
            tf.cast(features['geo_maps'], tf.float32), shape=[128, 128, 5])
        # training_masks = tf.reshape(
        #     tf.cast(features['training_masks'], tf.float32), shape=[128, 128, 1])
        #
        # return {"images": image, "score_maps": score_map, "geo_maps": geo_map,
        #         "training_masks": training_masks}, training_masks #dummy label/Y

        return {"images": image, "score_maps": score_map, "geo_maps": geo_map}, image #dummy label/Y

    def _get_train_dataset(self):
        """
        Reads TFRecords, decode and batches them
        :return: dataset
        """
        files = glob.glob(os.path.join(self._train_out_dir, "/*.tfrecords"))

        # TF dataset APIs
        dataset = tf.data.TFRecordDataset(files, num_parallel_reads=self._num_cores)
        # Map the generator output as features as a dict and labels
        dataset = dataset.map(self.decode)

        dataset = dataset.batch(batch_size=self._batch_size, drop_remainder=False)
        dataset = dataset.prefetch(self._prefetch_size)
        # dataset = dataset.cache(filename=os.path.join(self.iterator_dir, "train_data_cache"))
        print("Dataset output sizes are: ")
        print(dataset)

        return dataset

    def _get_val_dataset(self):
        """
        Reads TFRecords, decode and batches them
        :return: callable
        """
        files = glob.glob(os.path.join(self._val_out_dir, "/*.tfrecords"))
        # TF dataset APIs
        dataset = tf.data.TFRecordDataset(files, num_parallel_reads=self._num_cores)
        # Map the generator output as features as a dict and labels
        dataset = dataset.map(self.decode)

        dataset = dataset.batch(
            batch_size=self._batch_size, drop_remainder=False)
        dataset = dataset.prefetch(self._prefetch_size)
        print("Dataset output sizes are: ")
        print(dataset)

        return dataset

    def _get_test_dataset(self):
        """
        Reads TFRecords, decode and batches them
        :return: callable
        """
        files = glob.glob(os.path.join(self._test_out_dir, "/*.tfrecords"))
        # TF dataset APIs
        dataset = tf.data.TFRecordDataset(files, num_parallel_reads=self._num_cores)

        # Map the generator output as features as a dict and labels
        dataset = dataset.map(self.decode)

        dataset = dataset.batch(
            batch_size=self._hparams.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self._hparams.prefetch_size)
        print("Dataset output sizes are: ")
        print(dataset)

        return dataset

    def serving_input_receiver_fn(self):
        inputs = {
            # "images": tf.Variable(dtype=tf.float32, shape=[None, None, None, 3], validate_shape=False),
            "images": tf.compat.v1.placeholder(tf.float32, [None, None, None, 3]),
        }
        return tf.estimator.export.ServingInputReceiver(inputs, inputs)

