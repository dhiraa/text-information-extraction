# text-localization
Text Localization using Deep Learning 


## Environment Setup
```
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

sudo apt-get update && sudo apt-get install tensorflow-model-server

conda create --name tie
pip install -r requirements

conda activate tie

```

## Dataset  
**ICDAR [2019](http://rrc.cvc.uab.es/?ch=13)**   
- Use Google Drive Link: https://drive.google.com/drive/folders/1ShItNWXyiY1tFDM5W02bceHuJjyeeJl2 
and download the files

- All images are provided as JPEG or PNG files and the text files are UTF-8 files with CR/LF new line endings.
- The ground truth is given as separate text files (one per image) where each line specifies the coordinates of one 
word's bounding box and its transcription in a comma separated format 
- [2019](http://rrc.cvc.uab.es/?ch=13&com=tasks)

img_1.txt <-> img_01.txt

```sh
x1_1, y1_1,x2_1,y2_1,x3_1,y3_1,x4_1,y4_1, transcript_1

x1_2,y1_2,x2_2,y2_2,x3_2,y3_2,x4_2,y4_2, transcript_2

x1_3,y1_3,x2_3,y2_3,x3_3,y3_3,x4_3,y4_3, transcript_3
```

## Model

- [EAST](east)

### Commands

**EAST**

Most of the commands needs individual shell session.
 
 ```
cd /path/to/text-localization/

#Training
python run_east.py #to train

#Visualization
tensorboard --logdir=store/east/EASTModel/ #to view model metrics

#Serving
export MODEL_NAME=EAST
export MODEL_PATH=$PWD/store/east/EASTModel/exported/ #full path is needed!

tensorflow_model_server   \
--port=8500   \
--rest_api_port=8501   \
--model_name="$MODEL_NAME" \
--model_base_path="$MODEL_PATH"

python serving/east/grpc_predict.py \
--image data/icdar-2019-data/test/X00016469671.jpg \
--output_dir tmp/icdar/ \
--model EAST  \
--host "localhost" \
--signature_name serving_default


python grpc_predict.py \
--images_dir /opt/tmp/test/ \
--output_dir /opt/tmp/icdar/ \
--model EAST  \
--host "localhost" \
--signature_name serving_default 
```