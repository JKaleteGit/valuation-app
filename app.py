import flask
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from cbase import predict
from sklearn.model_selection import train_test_split

app = flask.Flask(__name__, template_folder='templates')
@app.route('/',  methods=['GET', 'POST'])



def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        address = flask.request.form['address']
        bedroom = flask.request.form['bedroom']
        sqf = flask.request.form['sqf']
        style = flask.request.form['style']
        age = flask.request.form['age']
        condition = flask.request.form['condition']
        garden = flask.request.form['garden']
        parking = flask.request.form['parking']
        bathroom = flask.request.form['bathroom']
        arr = [[address, bedroom, sqf, style, condition, age, garden, parking, bathroom]]                    

        model = RandomForestRegressor(n_estimators=150, max_features='sqrt',min_samples_split=5,min_samples_leaf=1,max_depth=35)                               
        p= predict(arr)
        df, x, y = p.transform_data(scale=True)
        xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.1,random_state=25)
        prediction = p.predict(xtrain,ytrain,df, model)
        return flask.render_template('main.html',
                                     original_input=str(address).title(),                                  
                                     result='Price: £{}'.format(round(float(prediction), 2)),
                                     )
