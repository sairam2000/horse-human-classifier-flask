from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', title='Horse-Human-Classifier')


@app.route('/details')
def details():
    return render_template('details.html', title='Horse-Human-Classifier')


@app.route('/predict', methods=['POST'])
def predict():
    f = request.files['img']
    f.save('uploads/'+f.filename)
    model = load_model("horse-human-classifier.h5")
    image = load_img('uploads/'+f.filename, target_size=(300, 300))
    x = img_to_array(image)
    x = np.expand_dims(x, axis=0)
    c = model.predict(x)
    os.remove('uploads/'+f.filename)
    if c[0] > 0.5:
        return render_template('index.html', text='OUTPUT: Human', title='Horse-Human-Classifier')
    else:
        return render_template('index.html', text='OUTPUT: Horse', title='Horse-Human-Classifier')


if __name__ == "__main__":
    app.run(debug=True)
