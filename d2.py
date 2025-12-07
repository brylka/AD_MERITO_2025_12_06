from flask import Flask, render_template

import joblib

model = joblib.load('model.joblib')
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('d2.html')

if __name__ == '__main__':
    app.run(debug=True)