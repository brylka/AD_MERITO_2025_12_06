import numpy as np
from flask import Flask, render_template, request
from PIL import Image
import joblib

model = joblib.load('model.joblib')
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    digit = ''
    if request.method == 'POST':
        file = request.files['image']
        img = Image.open(file).convert('L')
        img = img.resize((8,8))
        data = np.array(img)
        data = 16 - (data / 255.0 * 16)
        data = data.flatten().reshape(1,-1)
        digit = int(model.predict(data)[0])

    return render_template('d2.html', digit=digit)

if __name__ == '__main__':
    app.run(debug=True)