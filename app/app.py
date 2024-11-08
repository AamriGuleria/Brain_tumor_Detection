from flask import Flask,request, render_template, redirect, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
# C:\Users\asus\Desktop\Brain_Tumor_Detection\app\app.py
# C:\Users\asus\Desktop\Brain_Tumor_Detection\models\Brain_Tumor_Detection_Kaggle_Dataset_Final.h5
# model = load_model('../models/Brain_Tumor_Detection_Kaggle_Dataset_Final.h5')
model_path = os.path.join(os.path.dirname(__file__), '../models/Brain_Tumor_Detection_Kaggle_Dataset_Final.h5')
model=load_model(model_path)
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/main')
def mainpage():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))
        
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        filename = file.filename
        filepath = os.path.join('./static', filename)
        file.save(filepath) # Saving the file in the static folder

        img = tf.keras.preprocessing.image.load_img(filepath, target_size=(100, 100))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array.reshape((1,) + img_array.shape)

        prediction = model.predict(img_array)
        print("Prediction Result ",prediction[0][0])
        result = "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"

        return render_template('Result.html', prediction=result, image_url=url_for('static', filename=filename))

if __name__=="__main__":
    app.run(debug=True)