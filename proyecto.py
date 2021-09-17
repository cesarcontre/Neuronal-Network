# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 08:30:02 2021

@author: ccrgo
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


df = pd.read_csv('water_potability.csv')
df.isna().sum()

df.isna().sum() #1434 na

df.hist(bins=20,figsize=(10,10)) # vemos distribución de los datos

#------------Limpiamos los NaN-----

df = pd.read_csv('water_potability.csv')

df.shape
df.info()
#columns with nans 
df.isnull().sum() # sulfate, Trihalomethanes
#separeta nulls
tdata = df[df["ph"].isnull()]
t2data = df[df["Sulfate"].isnull()]
t3data = df[df["Trihalomethanes"].isnull()]
df.dropna(inplace=True)


#drop nan from df
df.shape

# train x and y from df without null

x_train = df.drop("ph",axis=1)
y_train = df["ph"]

x_train2 = df.drop("Sulfate",axis=1)
y_train2 = df["Sulfate"]

x_train3 = df.drop("Trihalomethanes",axis=1)
y_train3 = df["Trihalomethanes"]

from sklearn.linear_model import LinearRegression

lr  =LinearRegression()
lr2 =LinearRegression()
lr3 =LinearRegression()
# There are 3 columns with nans, so, we need 3 models
lr.fit(x_train,y_train)
lr2.fit(x_train2,y_train2)
lr3.fit(x_train3,y_train3)

x_test = tdata.drop("ph",axis=1).fillna(0)
x_test2 = t2data.drop("Sulfate",axis=1).fillna(0)
x_test3 = t3data.drop("Trihalomethanes",axis=1).fillna(0)

y_pred = lr.predict(x_test)
y_pred2 = lr2.predict(x_test2)
y_pred3 = lr2.predict(x_test3)

#Replace de missing values

tdata.loc[tdata.ph.isnull(),'ph'] = y_pred
ndf = pd.read_csv('water_potability.csv')
ndf.loc[ndf.ph.isnull(),'ph'] = y_pred
ndf.loc[ndf.Sulfate.isnull(),'Sulfate'] = y_pred2
ndf.loc[ndf.Trihalomethanes.isnull(),'Trihalomethanes'] = y_pred3

#ndf.to_csv(r'C:\Users\ccrgo\Documents\maestría\verano\waterclear.csv',index=False)

#Correlation bewteen varaibles
data = ndf

corr = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()
# The mayority is near 0 to each correlation 

#--------NeuronaL network-------------
import pandas as pd
import numpy as np
data = pd.read_csv('waterclear.csv')
#data = pd.read_csv('water_potability.csv').fillna(0)
# We give the df clear for learning

def learning(x, Wh, W0, a = 0.47, alp = 0.8001, E =0.000001, L = 9):
    
    Y = np.random.random((len(x),1))
    while True:
        
        for i in range(L):
            
            neth = Wh@x[i]
            
            yh = 1/(1+np.exp(-a*neth))
            
            neth0 = W0 @ yh
            
            Y[i] = 1/(1+np.exp(-a*neth0))
            
            delta0 = (D[i] - Y[i]) * Y[i] * (1 - Y[i])
            
            deltah = yh* (1-yh) * (np.transpose(W0)@delta0)
            
            W0 += np.transpose(np.atleast_2d(alp * delta0)) @ np.atleast_2d(yh)
            
            Wh += np.transpose(np.atleast_2d(alp*deltah)) @ np.atleast_2d(x[i])
        
        #print(np.linalg.norm(delta0))
        
        if np.linalg.norm(delta0) <= E:
            return Wh, W0
        
def funct(x, Wh, W0, a = 0.47, L = 9):
    
    Y = np.random.random((len(x),1))
    for i in range(len(x)):
        
        neth = Wh@x[i]
            
        yh = 1/(1+np.exp(-a*neth))
            
        neth0 = W0 @ yh
            
        Y[i] = 1/(1+np.exp(-a*neth0))
        
    return Y
#--normalization of data base for drecreasse CPU processing
         
def minmax_norm(df_input):
    return (data - data.min()) / ( data.max() - data.min())
data = minmax_norm(data)

# -------------Sample to train
index_list = np.random.choice(data.index, 2294, replace=False)

data_train = data.iloc[index_list, :]

# -------------Proof test
data_test = data.drop(index_list)
x = data_train.iloc[:,:9 ].to_numpy()
D = data_train['Potability'].to_numpy()

# ----------------Size 
Wh = np.random.random((15, 9))
W0 = np.random.random((1, 15))

# Learning
Wh, W0 = learning(x, Wh, W0)

#---------------------------- Test 
x = data_test.iloc[:, :9].to_numpy()

# --------------Add the learning
data_test['Y'] = funct(x, Wh, W0)

#-----------------Check accuracy
data_test.loc[(data_test['Potability'] == 1) & (data_test['Y'] >= 0.5), 'Resultado'] = True 
data_test.loc[(data_test['Potability'] == 0) & (data_test['Y'] <= 0.5), 'Resultado'] = True 
data_test.loc[(data_test['Potability'] == 1) & (data_test['Y'] < 0.5), 'Resultado'] = False 
data_test.loc[(data_test['Potability'] == 0) & (data_test['Y'] > 0.5), 'Resultado'] = False

resultado = data_test['Resultado'].value_counts(normalize=True) * 100

data.hist(bins=20,figsize=(10,10))



