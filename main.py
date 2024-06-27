import flask
import numpy as np
import pickle
from flask import render_template
import sklearn
#from sklearn.linear_model import LinearRegression


app=flask.Flask(__name__,template_folder='templates')
@app.route('/',methods=['POST','GET'])
@app.route('/index',methods=['POST','GET'])

def main():
    if flask.request.method=='GET':
        return render_template('main.html')
    if flask.request.method=='POST':
        with open('model.pkl','rb') as f:
            loaded_model=pickle.load(f)
        iw =float(flask.request.form['iw'])
        if_=float(flask.request.form['if'])
        vw =float(flask.request.form['vw'])
        fp =float(flask.request.form['fp'])
        y_pred=loaded_model.predict([[iw,if_,vw,fp]])
        return render_template('main.html',result=[round(y_pred[0][0],2),round(y_pred[0][1],2)])

if __name__=='__main__':
    app.run()
