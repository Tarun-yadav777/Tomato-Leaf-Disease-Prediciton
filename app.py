import os
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


app = Flask(__name__)
xception_model = load_model('model_xception.h5')
inceptionv3_model = load_model('model_inception.h5')
resnet50_model = load_model('model_resnet50.h5')


def xception_predict(img_path, xception_model):
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)
    prediction = xception_model.predict(x)
    prediction = np.argmax(prediction, axis=1)
    
    if prediction==0:
        prediction="The Disease is Tomato_Bacterial_spot"
    elif prediction==1:
        prediction="The Disease is Tomato_Early_blight"
    elif prediction==2:
        prediction="Te Disease is Tomato_healthy"
    elif prediction==3:
        prediction="The Disease is Tomato_Late_blight"
    elif prediction==4:
        prediction="The Disease is Tomato_Leaf_Mold"
    elif prediction==5:
        prediction="The Disease is Tomato_Septoria_Leaf_Spot"
    elif prediction==6:
        prediction="The Disease is Tomato_Spider_Mites_Two_Spotted_Spider_Mites"
    elif prediction==7:
        prediction="The Disease is Tomato_Target_Spots"
    elif prediction==8:
        prediction="The Disease is Tomato_Mosaic_Virus"
    elif prediction==9:
        prediction="The Disease is Tomato_Yellow_Leaf_Curl_Virus"

    
    return prediction

def inception_predict(img_path, inceptionv3_model):
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)
    prediction = xception_model.predict(x)
    prediction = np.argmax(prediction, axis=1)
    
    if prediction==0:
        prediction="The Disease is Tomato_Bacterial_spot"
    elif prediction==1:
        prediction="The Disease is Tomato_Early_blight"
    elif prediction==2:
        prediction="Te Disease is Tomato_healthy"
    elif prediction==3:
        prediction="The Disease is Tomato_Late_blight"
    elif prediction==4:
        prediction="The Disease is Tomato_Leaf_Mold"
    elif prediction==5:
        prediction="The Disease is Tomato_Septoria_Leaf_Spot"
    elif prediction==6:
        prediction="The Disease is Tomato_Spider_Mites_Two_Spotted_Spider_Mites"
    elif prediction==7:
        prediction="The Disease is Tomato_Target_Spots"
    elif prediction==8:
        prediction="The Disease is Tomato_Mosaic_Virus"
    elif prediction==9:
        prediction="The Disease is Tomato_Yellow_Leaf_Curl_Virus"

    
    return prediction


def resnet_predict(img_path, resnet50_model):
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)
    prediction = xception_model.predict(x)
    prediction = np.argmax(prediction, axis=1)
    
    if prediction==0:
        prediction="The Disease is Tomato_Bacterial_spot"
    elif prediction==1:
        prediction="The Disease is Tomato_Early_blight"
    elif prediction==2:
        prediction="Te Disease is Tomato_healthy"
    elif prediction==3:
        prediction="The Disease is Tomato_Late_blight"
    elif prediction==4:
        prediction="The Disease is Tomato_Leaf_Mold"
    elif prediction==5:
        prediction="The Disease is Tomato_Septoria_Leaf_Spot"
    elif prediction==6:
        prediction="The Disease is Tomato_Spider_Mites_Two_Spotted_Spider_Mites"
    elif prediction==7:
        prediction="The Disease is Tomato_Target_Spots"
    elif prediction==8:
        prediction="The Disease is Tomato_Mosaic_Virus"
    elif prediction==9:
        prediction="The Disease is Tomato_Yellow_Leaf_Curl_Virus"

    
    return prediction



@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/xception', methods=['GET'])
def xception():
    return render_template('xception.html')

@app.route('/inceptionv3', methods=['GET'])
def inception():
    return render_template('inceptionv3.html')

@app.route('/resnet50', methods=['GET'])
def resnet():
    return render_template('resnet50.html')

@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
       
        f = request.files['file']

        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        preds = xception_predict(file_path, xception_model)
        result=preds
        return result
    return None



if __name__ == '__main__':
    app.run()