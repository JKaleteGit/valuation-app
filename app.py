import flask
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor

app = flask.Flask(__name__, template_folder='templates')
@app.route('/',  methods=['GET', 'POST'])



def main():
    if flask.request.method == 'GET':
        return(flask.render_template('home.html'))
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
        location = flask.request.form['location']
        arr = [[address, bedroom, sqf, style, condition, age, garden, parking, bathroom, location]]                    
                                       
        def get_data():
            url_main = 'https://raw.githubusercontent.com/JKaleteGit/valuation-app/master/val.csv'
            url_sub = 'https://raw.githubusercontent.com/JKaleteGit/valuation-app/master/val_r.csv'
            df2 = pd.read_csv(url_sub)
            df = pd.read_csv(url_main)
            df2 = df2.dropna()

            df = df.drop(25)
            label = df['Sold Price'].values
            rpat = r'\d+,\d+(,\d+)?'
            y = [re.search(rpat, i).group().replace(',','') for i in label]
            y = [float(x) for x in y]

            df = df.drop(columns=['Location ranking (1-10)', 'Column1', 'Sold Price'], axis=1)
            return df, df2, y

        def sort_data(arr):
            df, df2, y = get_data()
            rd = {}
            for i in range(len(df2)):
                rd[df2['Property Address'].iloc[i]] = df2['Ranking'].iloc[i]

            rdf =  {k.lower().strip(): v for k, v in rd.items()}
        
            pat = r'[a-zA-Z]+\s[a-zA-Z]+|[a-zA-Z]+'
            rnums = []
            for i in range(len(df)):
                g = re.search(pat, str(df['Property Address'].iloc[i]))
                if g:
                    rnums.append(g)
            rval = [x.group() for x in rnums if x]
            for i in range(len(rval)):
                if rval[i] == 'a':
                    rval[i] = 'Sundorne Road'
            rval = [x.lower() for x in rval]


            lr = []
            for i in rval:
                try:
                    lr.append(rdf[i])
                except:
                    print(i)
            df['Location ranking'] = lr
            df['Property Address'] = rval
            xd = np.array(arr)
            xd.reshape(1,10)
            xd = pd.DataFrame(xd, columns=df.columns)
            df.index = df.index + 1
            df = df.append(xd)
            df = df.sort_index()
            return df, y

        def get_street():
            df, y = sort_data(arr)
            streetdf = df['Property Address']
            return streetdf

        def transform_data():
            df, y = sort_data(arr)
            le = LabelEncoder()
            categ = df[['Property Address', 'Property Style (terraced, semi-detached, detached)', 'Property era (Victorian, Georgian, 1950s,1930s,1960s or newer']]
            categ = categ.apply(lambda col: le.fit_transform(col.astype(str)), axis=0, result_type='expand')
            df[['Property Address', 'Property Style (terraced, semi-detached, detached)', 'Property era (Victorian, Georgian, 1950s,1930s,1960s or newer']] = categ[['Property Address', 'Property Style (terraced, semi-detached, detached)', 'Property era (Victorian, Georgian, 1950s,1930s,1960s or newer']]

            for i in range(len(df)):
                if df['Parking (yes or no)'].iloc[i] == 'yes':
                    df['Parking (yes or no)'].iloc[i] = 1
                else:
                    df['Parking (yes or no)'].iloc[i] = 0


            #x = df.drop(columns='Sold Price').values

            x = df.iloc[1:,:].values
            return df, x, y 

        model = RandomForestRegressor(n_estimators=150, max_features='auto')

        def fit(x,y, model):
            model = model.fit(x,y)
            return model

        def predict(df, model):
            fit(x,y,model)
            score = model.score(x, y)
            street = get_street()
            p = np.array(df.iloc[0:1,:])
            pred = model.predict(p)
            return pred
        df, x, y = transform_data()
        prediction = predict(df, model)
        return flask.render_template('main.html',
                                     original_input=str(address).title(),                                  
                                     result='Price: £{}'.format(round(float(prediction), 2)),
                                     )
