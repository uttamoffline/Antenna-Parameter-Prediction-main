import numpy as np
import pandas as pd
from flask import Flask , request , jsonify , render_template
import pickle

#create flask app
app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
model1 = pickle.load(open("model1.pkl","rb"))
@app.route("/")

def Home():
    return render_template("index.html")
@app.route("/predict" , methods = ["POST"])

def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree = 5 , include_bias= False)
    features = poly.fit_transform(features)
    prediction = model.predict(features)
    prediction1 = model1.predict(features)
    return render_template("index.html" , prediction_text = "The required length of the patch antenna is {} and width of the patch antenna is {}".format(prediction,prediction1))

if __name__ == "__main__":
    app.run(debug = True)