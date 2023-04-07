from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

app=Flask(__name__)
CORS(app, support_credentials=True)

@app.route("/")
def home():
    return "Backend is running..."


@app.route('/predict', methods=['POST'])
def home():
    n = request.form['n']
    p = request.form['p']
    k = request.form['k']
    t = request.form['t']
    h = request.form['h']
    ph = request.form['ph']
    rf = request.form['rf']
    arr = np.array([[n, p, k, t,h, ph , rf]],dtype=float)
    pred = model.predict(arr)
    print(pred)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000", debug=True)
