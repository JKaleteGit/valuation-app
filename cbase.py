import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class predict:
    def __init__(self, arr):
        self.arr = arr

    def get_data(self):
        url_main = 'https://raw.githubusercontent.com/JKaleteGit/valuation-app/master/val2.csv'
        url_sub = 'https://raw.githubusercontent.com/JKaleteGit/valuation-app/master/val_r.csv'
        df2 = pd.read_csv(url_sub)
        df = pd.read_csv(url_main)
        df2 = df2.dropna()
        
        label = df['Sold Price'].values
        rpat = r'\d+,\d+(,\d+)?'
        y = [re.search(rpat, i).group().replace(',','') for i in label]
        y = [float(x) for x in y]

        df = df.drop(columns=['Location ranking (1-10)', 'Column11', 'Sold Price'], axis=1)
        #df = df.dropna()
        
        return df, df2, y

    def sort_data(self, arr):
        p = predict(self.arr)
        df, df2, y = p.get_data()

        xd = np.array(arr)
        xd.reshape(1,9)
        xd = pd.DataFrame(xd, columns=df.columns)
        df.index = df.index + 1
        df = df.append(xd)
        df = df.sort_index()

        rd = {}
        for i in range(len(df2)):
            rd[df2['Property Address'].iloc[i]] = df2['Ranking'].iloc[i]

        rdf =  {k.lower().strip(): v for k, v in rd.items()}

        #####
        pat = r'[a-zA-Z]+\s[a-zA-Z]+(\s[a-zA-Z]+)?|[a-zA-Z]+'
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

        #####
        lr = []
        for i in rval:
            try:
                lr.append(rdf[i])
            except:
                print(i)
        df['Location ranking'] = lr
        
        return df, y

    def transform_data(self, scale = False):
        p = predict(self.arr)
        df, y = p.sort_data(self.arr)
        if scale:
            numdf = ['Bedrooms', 'SQF', 'Condition ranking (1-5)', 'Garden Size (1-3)', 'Bathrooms', 'Location ranking']
            
            sc = MinMaxScaler()
            newdf = df[numdf].astype('float32')
            df = df.drop(newdf, axis=1)
            newdf = pd.DataFrame(sc.fit_transform(newdf.values))
            df = pd.concat([df, newdf],axis=1)
        
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

    def fit(self, x,y, model):
        model = model.fit(x,y)
        return model

    def predict(self,xtrain,ytrain, df, model):
        p = predict(self.arr)
        p.fit(xtrain,ytrain,model)
        pr = np.array(df.iloc[0:1,:])
        pred = model.predict(pr)

        return pred