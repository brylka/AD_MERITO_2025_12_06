from flask import Flask, render_template, request

import joblib

model = joblib.load('model.joblib')
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return "Odebrałem plik"
    return render_template('d2.html')

if __name__ == '__main__':
    app.run(debug=True)