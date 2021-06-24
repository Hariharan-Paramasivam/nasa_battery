# -*- coding: utf-8 -*-


"""
Created on Tue May 25 09:51:47 2021

@author: Muthu Periyal
"""

from flask import Flask,render_template,request
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle as pkl
pol_reg=pkl.load(open('model/state_of_health_poly_reg.pkl','rb'))
mx1=pkl.load(open('model/state_of_health_minmax.pkl','rb'))
model_soh=pkl.load(open('model/state_of_health.pkl','rb'))
model_rul=pkl.load(open('model/remaining_useful_cycle.pkl','rb'))
mx2=pkl.load(open('model/remaining_useful_cycle_minmax.pkl','rb'))


#app=Flask(__name__)
app=Flask(__name__,static_url_path='/static')



from time import time
start=time()




@app.route('/')
def main():
    return render_template('Front.html')

@app.route('/front',methods=['POST','GET'])
def result():
    if request.method=='POST':
        a=request.form.get("Discharge")
        b=request.form.get("Charge")
        x1=[[a,b]]
        input1=pol_reg.transform(mx1.transform(x1))
        SOH= float(model_soh.predict(input1))
        x2=[[a,b,SOH]]
        input2=mx2.transform(x2)
        RUL= float(model_rul.predict(input2))
        return render_template('result.html',SOH='{:2.4}'.format(SOH*100),RUL=round(RUL))
    else:
        return render_template('Front.html')
    




if __name__=='__main__':
    app.run(debug=True)
    


#y_pred=model.predict([[6436.141,1427.625]]) #predicting model for dynamic data
#print(y_pred)

