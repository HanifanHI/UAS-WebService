# ======================================================================================
# NAMA KELOMPOK :
# 1. HANIFAN HUSEIN ISNANTO - 19090006
# 2. MUHAMMAD FIKRI - 19090126
# KELAS : 6 C
# ======================================================================================

from tensorflow.keras.models import load_model
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, BatchNormalization, Flatten, Activation
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import datetime
from datetime import date
from keras.utils.vis_utils import plot_model
import pickle
from flask import Flask, jsonify, request, flash
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS
from flask_restful import Resource, Api
import pymongo
import re
from flask_ngrok import run_with_ngrok
import pyngrok
from PIL import Image
import random
import string

app = Flask(__name__)
run_with_ngrok(app)

app.secret_key = "planttect"

# ========== SETUP MONGODB =====================================================================
MONGO_ADDR = 'mongodb://localhost:27017'
MONGO_DB = "Plantect"

conn = pymongo.MongoClient(MONGO_ADDR)
db = conn[MONGO_DB]
api = Api(app)

# Model Path
MODEL_PATH = 'models/model_plant.h5'
model = load_model(MODEL_PATH, compile=False)

pickle_inn = open('num_class_plant.pkl', 'rb')
num_classes_plant = pickle.load(pickle_inn)

UPLOAD_FOLDER = 'foto_tanaman'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ========== PREDICT IMAGE ============================================================================
@app.route('/api/v1/plants/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        flash('No file part')
        return jsonify({
            "pesan": "Tidak Ada Form Image"
        })

    # Request File Image
    file = request.files['image']

    if file.filename == '':
        return jsonify({
            "Message": "Tidak ada file image yang dipilih"
        })
    if file and allowed_file(file.filename):
        path_del = r"foto_tanaman\\"
        for file_name in os.listdir(path_del):
            file_del = path_del + file_name
            if os.path.isfile(file_del):
                os.remove(file_del)

        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(5))

        filename = secure_filename(file.filename+result_str+".jpg")
        print(filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        path = ("foto_tanaman/"+filename)

        img = image.load_img(path, target_size=(224, 224))
        img1 = image.img_to_array(img)
        img1 = img1/255
        img1 = np.expand_dims(img1, [0])
        predict = model.predict(img1)
        classes = np.argmax(predict, axis=1)
        for key, values in num_classes_plant.items():
            if classes == values:
                accuracy = float(round(np.max(model.predict(img1))*100, 2))
                info = db['plants'].find_one({'nama': str(key)})
                print("Prediksi Tanaman : "+str(key) +
                      " dengan probability : "+str(accuracy)+"%")

                return jsonify({
                    "Nama_Tanaman": str(key),
                    "Accuracy": str(accuracy)+"%",
                    "Jenis_Tanaman": info['jenis'],
                    "Deskripsi": info['deskripsi'],
                    "Nutrisi": info['nutrisi'],
                    "Manfaat":  info['manfaat']
                })
    else:
        return jsonify({
            "Message": "Bukan File Image"
        })


if __name__ == '__main__':
    app.run()
    #app.run(debug = True, port=5000, host='0.0.0.0')
