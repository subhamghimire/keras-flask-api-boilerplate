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
    img_pred = np.expand_dims(x, axis=0)
    result = model.predict(img_pred)
    print('result')
    print(result)
    return result


def get_model():
    model = load_model('./models/v1.h5')
    model.summary()
    model.freeze()
    print(" * Model loaded!")
    return model


# Classes 
classes = {0: 'A',
           1: 'B',
           2: 'C',
           3: 'D',
           4: 'E'}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            print('FJNDJNF')
            # Get the file from post request
            file_path = get_file_path_and_save(request)
            pred = predict_probability(get_model(), file_path)

            pred_class = pred.argmax(axis=-1)
            print(f"predicted class:", pred_class)

            pred_prob = pred.max(axis=-1)*100
            formated_pred_prob = "{:.2f}".format(pred_prob)
            print(f"Predicted probability:",formated_pred_prob)

            s = pred_class
            result = classes[s]
            print(f"Predicted image:", result)

            os.remove(file_path)

            return {"data": result, "probability": formated_pred_prob, "status": 200}, 200

        except Exception as e:
            print(e)
            return {"data": "Warning!!!, Please upload valid images format only"}, 400
    return None, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
            