#[참조 1] 소스코드 분석

import numpy as np
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

list=[]
list_1=[[]]
f = open("./predict.txt",'r')
lines=f.readlines()
for line in lines:
	line=line.strip()
	l=line.split(',')
	list.append(l)
f.close()

for i in list:
	for k in i:
		o=float(k)
		list_1[0].append(o)
	rent_Fee=lr.predict(list_1)
	r=np.round(rent_Fee,2)
	print(r)
	list_1[0].clear()

# text 파일 불러와서 줄별로 읽어들이기

import statsmodels.formula.api as sm
result=sm.ols(formula='rent ~ holiday + holiday_before + holiday_after + ave_temper + low_temper + high_temper + ave_rain + rain_exist + ave_wind',data=rent_Data).fit()
print(result.summary())

