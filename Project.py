# load all the required libraries
from flask import Flask,jsonify,request, render_template
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/')
def mainPage():
    return render_template("index.html")

@app.route('/train_model')
def train():
    data = pd.read_excel('False Alarm Cases.xlsx')
    x = data.iloc[:, 1:7]
    y = data['Spuriosity Index(0/1)']
    logm = LogisticRegression()
    logm.fit(x, y)
    joblib.dump(logm, open('model.pkl','wb'))
    return "Done"

#  load pickle file and test your model, pass test data via POSt method
#  First we need to load pickle file for it to get training data ref
@app.route('/test_model', methods=['POST'])
def test():

    pkl_file = joblib.load(open('model.pkl','rb'))
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
   # print(int_features)
    #print(final)
    y_pred = pkl_file.predict(final)

    if y_pred == 1:
        return render_template('forest.html',pred='False Alarm Danger')
    else:
        return render_template('forest.html', pred="True Alarm Danger ")


app.run(debug=True,port=7000)
