import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.utils import load_img, img_to_array 
from flask import *
from werkzeug.utils import secure_filename


app = Flask(__name__)

def get_file_path_and_save(request):
    # Get the file from post request
    f = request.files['file']

    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)
    return file_path

def predict_probability(model, image_path):
    img = load_img(image_path, target_size=(100, 100))
    # Preprocessing the image
    x = img_to_array(img)
    img_pred = np.expand_dims(x/255, axis=0)
    result = model.predict(img_pred)
    return result

def get_model():
    model = load_model('./models/v2.h5', compile=False)
    model.summary()
    print("Model loaded!")
    return model

# Classes 
classes = {
            0: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)	',
            1: 'Corn_(maize)___Northern_Leaf_Blight	',
            2: 'Cherry_(including_sour)___Powdery_mildew',
            3: 'Soybean___healthy',
            4: 'Tomato___Late_blight',
            5: 'Apple___healthy',
            6: 'Tomato___Septoria_leaf_spot',
            7: 'Peach___healthy',
            8: 'Tomato___Tomato_mosaic_virus',
            9: 'Raspberry___healthy',
            10: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            11: 'Potato___Early_blight',
            12: 'Strawberry___healthy',
            13 : 'Strawberry___Leaf_scorch',
            14: 'Apple___Apple_scab',
            15: 'Pepper,_bell___healthy',
            16: 'Corn_(maize)___healthy',
            17: 'Orange___Haunglongbing_(Citrus_greening)',
            18: 'Tomato___Target_Spot',
            19: 'Tomato___healthy',
            20: 'Grape___healthy',
            21: 'Tomato___Leaf_Mold', 
            22: 'Tomato___Bacterial_spot',
            23: 'Apple___Cedar_apple_rust',
            24: 'Cherry_(including_sour)___healthy',
            25: 'Corn_(maize)___Common_rust_',
            26: 'Pepper,_bell___Bacterial_spot',
            27: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            28: 'Apple___Black_rot',
            29: 'Grape___Black_rot',
            30: 'Potato___healthy', 
            31: 'Peach___Bacterial_spot',
            32: 'Potato___Late_blight',
            33: 'Squash___Powdery_mildew',
            34: 'Tomato___Spider_mites Two-spotted_spider_mite',
            35: 'Blueberry___healthy',
            36: 'Grape___Esca_(Black_Measles)	',
            37: 'Tomato___Early_blight',
        }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            # Get the file from post request
            file_path = get_file_path_and_save(request)
            pred = predict_probability(get_model(), file_path)

            pred_class = pred.argmax(axis=-1)
            print(f"predicted class:", pred_class)

            pred_prob = pred.max(axis=-1)*100
            formated_pred_prob = format(pred_prob[0], '.2f')
            print(f"Predicted probability:",formated_pred_prob, '%' )

            result = classes[np.argmax(pred)]
            print(f"Predicted image:", result)

            os.remove(file_path)

            return {"data": result, "probability": formated_pred_prob, "status": 200}, 200
        
        except Exception as e:
            print(e)
            return {"data": "Warning!!!, Please upload valid images format only"}, 400
    return None, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
            