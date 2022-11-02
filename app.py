#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pickle
from collections.abc import Mapping
from flask import Flask, request, render_template
app = Flask(__name__)
model = pickle.load(open('model.sav', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods = ['POST'])
def predict():
    features = [np.array([float(x) for x in request.form.values()])]
    prediction = model.predict(features)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text = 'House price prediction: ${}'.format(output))
if __name__ == '__main__':
    app.run(port = 5000, debug = True)


# In[ ]:




