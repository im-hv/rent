import pandas as pd
import statsmodels.formula.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

rent_Data = pd.read_csv("./creel.csv",encoding='CP949')
print (rent_Data.head())

result=sm.ols(formula='rent ~ holiday + holiday_before + holiday_after + ave_temper + low_temper + high_temper + ave_rain + rain_exist + ave_wind',data=rent_Data).fit()
print(result.summary())