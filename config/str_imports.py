import gin

from dataset.scene_text_recognition.str_dataset import SceneTextRecognitionDataset
from models.str.str_models import SceneTextRecognitionModel

@gin.configurable
def get_experiment_root_directory(value):
    return value
