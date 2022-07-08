from __future__ import division, print_function
from glob import glob
from posixpath import dirname
# coding=utf-8
import sys
import os

import numpy as np
# Keras
from colorama import Fore,Style
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
# 
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import tensorflow as tf
# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'maskdetection.h5'

# Loading model 
model = load_model(MODEL_PATH)

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = tf.keras.utils.load_img(img_path, target_size=(400, 400))
    x = tf.keras.utils.img_to_array(img)
    x = x/255
    proba = model.predict(x.reshape(1, 400, 400, 3))
    return proba


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)
        res = ["Elbes Mask ya ham :3", "Sa7itek ya sidi"]
        print(preds)
        b = res[np.argmax(preds)]
        #print(Fore.GREEN,b,Fore.WHITE)
        os.remove('./uploads/' + f.filename)
        #print(b)
        return b
    return None

global basepath

@app.route('/data')
def data(): 
    if request.method == 'POST ': 
        f = request.files['file']
        basepath =  os.path.dirname(__file__)



@app.route('/myteam')
def myteam():
    #Main page
    return render_template('myteam.html')


if __name__ == '__main__':
    app.run(debug=True,threaded=False)
