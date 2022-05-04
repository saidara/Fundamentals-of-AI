import string
from flask import Flask, request,render_template,jsonify
import joblib
import os
import numpy as np
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")



@app.route('/', methods=['POST', 'GET'])
def result():
    Item_Weight = float(request.form['Item_Weight'])
    Item_Fat_Content = float(request.form['Item_Fat_Content'])
    Item_Visibility = float(request.form['Item_Visibility'])
    Item_Type = float(request.form['Item_Type'])
    Item_MRP = float(request.form['Item_MRP'])
    Outlet_Establishment_Year = float(request.form['Outlet_Establishment_Year'])
    Outlet_Size = float(request.form['Outlet_Size'])
    Outlet_Location_Type = float(request.form['Outlet_Location_Type'])
    Outlet_Type = float(request.form['Outlet_Type'])



    X = np.array([Item_Weight,Item_Fat_Content,Item_Visibility,Item_Type,Item_MRP,Outlet_Establishment_Year,Outlet_Size,Outlet_Location_Type,Outlet_Type]).reshape(1,-1)

    print(X)


    model_path = r'/Users/saidaraogonuguntla/Desktop/bigmart project/model.sav'

    model = joblib.load(model_path)

    Y = model.predict(X)
    pred = Y

    return render_template('index.html', prediction_text = '{}'.format(pred))



if __name__ =="__main__":
    app.run(debug=True, port=5623)
