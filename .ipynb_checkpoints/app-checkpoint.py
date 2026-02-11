import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the trained model
model = load_model("Vgg-16-nail-disease.h5")

# Class labels
index = ['Darier_s_disease', 'Muehrcke_s_lines', 'alopecia_areata',
         'beau_s_lines', 'bluish_nail', 'clubbing', 'eczema',
         'half_and_half_nails_(Lindsay_s_nails)', 'koilonychia',
         'leukonychia', 'onycholysis', 'pale_nail', 'red_lunula',
         'splinter_hemorrhage', 'terry_s_nail', 'white_nail',
         'yellow_nails']

@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/nailhome')
def nailhome():
    return render_template('nailhome.html')

@app.route('/nailpred')
def nailpred():
    return render_template('nailpred.html')

@app.route('/nailresult', methods=['GET', 'POST'])
def nailresult():
    if request.method == 'POST':
        f = request.files['image']
        filepath = os.path.join("static/uploads", f.filename)
        f.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        prediction = np.argmax(model.predict(x), axis=1)[0]
        result = index[prediction]

        return render_template('nailpred.html', prediction_text=result)

    return render_template('nailpred.html')

if __name__ == "__main__":
    app.run(debug=False, port=5000)
