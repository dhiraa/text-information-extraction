# EAST: An Efficient and Accurate Scene Text Detector

**Refernce Git** : https://github.com/argman/EAST  
**Paper** : https://arxiv.org/abs/1704.03155v2   



## Configuration

Check this [file](../config/east_config.gin)!

##Compile lanms

```sh
cd lanms
make
```
If you get compilation issues with lanms, follow this [link](https://github.com/argman/EAST/issues/156#issuecomment-404166990)! to resolve it.


### PS

- As compared to original EAST repo, we have used Tensorflow high level APIs tf.data and tf.Estimators
- This comes in handy when we move to big dataset or if we wanted to experiment with different models/data
- TF Estimator also takes care of exporting the model for serving! [Reference](https://medium.com/@yuu.ishikawa/serving-pre-modeled-and-custom-tensorflow-estimator-with-tensorflow-serving-12833b4be421)

### Similar Models / Gits:
- https://github.com/Michael-Xiu/ICDAR-SROIE
- https://github.com/xieyufei1993/FOTS
- https://github.com/hwalsuklee/awesome-deep-text-detection-recognition
- https://github.com/tangzhenyu/Scene-Text-Understanding
- https://github.com/kurapan/EAST
- https://github.com/songdejia/EAST
- https://github.com/huoyijie/AdvancedEAST

### References
- https://berlinbuzzwords.de/18/session/scalable-ocr-pipelines-using-python-tensorflow-and-tesseract
- http://machinelearninguru.com/deep_learning/data_preparation/tfrecord/tfrecord.html
- https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618
- https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/
- https://databricks.com/blog/2018/07/10/how-to-use-mlflow-tensorflow-and-keras-with-pycharm.html
- https://stackoverflow.com/questions/51455863/whats-the-difference-between-a-tensorflow-keras-model-and-estimator