import numpy as np
from flask import Flask, render_template, request
from PIL import Image
import joblib

model = joblib.load('model_mnist.joblib')
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    digit = ''
    if request.method == 'POST':
        file = request.files['image']
        img = Image.open(file).convert('L')
        img = img.resize((28,28))
        data = np.array(img)
        data = (255 - data).reshape(1,-1) / 255
        digit = int(model.predict(data)[0])

    return render_template('d4.html', digit=digit)

if __name__ == '__main__':
    app.run(debug=True)