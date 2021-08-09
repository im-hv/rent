import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd


rent_Data = pd.read_csv("./creel.csv",encoding='CP949')

from sklearn.model_selection import train_test_split
x=rent_Data[['holiday','holiday_before','holiday_after','ave_temper','low_temper','high_temper','ave_rain','rain_exist','ave_wind']]
y=rent_Data[['rent']]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

i_Want=[[1,1,0,2.8,-1.5,7.8,0,0,0.6]]
rent_Fee=lr.predict(i_Want)
r=np.round(rent_Fee,2)
print(r)


y_predict = lr.predict(x_test)
print(lr.score(x_train,y_train))
