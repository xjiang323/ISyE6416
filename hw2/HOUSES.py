import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

data = pd.read_csv('RealEstate.csv')

data1 = data.loc[data['Status'] == 'Short Sale',:]
data1 = data1.loc[:, data1.columns != 'Status']
data_1 = data1.select_dtypes(include=[object])
le = preprocessing.LabelEncoder()
data_1 = data_1.apply(le.fit_transform)
data1['Location'] = data_1['Location']

data2 = data.loc[data['Status'] == 'Foreclosure',:]
data2 = data2.loc[:, data2.columns != 'Status']
data_2 = data2.select_dtypes(include=[object])
data_2 = data_2.apply(le.fit_transform)
data2['Location'] = data_2['Location']

data3 = data.loc[data['Status'] == 'Regular',:]
data3 = data3.loc[:,data3.columns != 'Status']
data_3 = data3.select_dtypes(include=[object])
data_3 = data_3.apply(le.fit_transform)
data3['Location'] = data_3['Location']

x_data1 = data1.loc[:, data1.columns != 'Price' ]
y_data1 = data1.loc[:, data1.columns == 'Price']
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data1, y_data1, test_size = 0.2, random_state = 100, shuffle = True)

x_data2 = data2.loc[:, data2.columns != 'Price' ]
y_data2 = data2.loc[:, data2.columns == 'Price']
x_train2, x_test2, y_train2, y_test2 = train_test_split(x_data2, y_data2, test_size = 0.2, random_state = 100, shuffle = True)

x_data3 = data3.loc[:, data3.columns != 'Price' ]
y_data3 = data3.loc[:, data3.columns == 'Price']
x_train3, x_test3, y_train3, y_test3 = train_test_split(x_data3, y_data3, test_size = 0.2, random_state = 100, shuffle = True)

reg = LinearRegression()
reg.fit(x_train1, y_train1)
y_test_predict1 = reg.predict(x_test1).round()
plt.scatter(y_test1, y_test_predict1, c = 'green')
print("Short Sale:")
print("Coefficients: " , reg.coef_)
print("Score: ", reg.score(x_train1, y_train1))
print("Intercept: ", reg.intercept_)
print("Mean Square Error: ", mean_squared_error(y_test1, y_test_predict1))

reg.fit(x_train2, y_train2)
y_test_predict2 = reg.predict(x_test2).round()
plt.scatter(y_test2, y_test_predict2, c = 'red')
print("Foreclosure:")
print("Coefficients: " , reg.coef_)
print("Score: ", reg.score(x_train2, y_train2))
print("Intercept: ", reg.intercept_)
print("Mean Square Error: ", mean_squared_error(y_test2, y_test_predict2))

reg.fit(x_train3, y_train3)
y_test_predict3 = reg.predict(x_test3).round()
plt.scatter(y_test3, y_test_predict3, c = 'blue')
plt.show()
print("Regular:")
print("Coefficients: " , reg.coef_)
print("Score: ", reg.score(x_train3, y_train3))
print("Intercept: ", reg.intercept_)
print("Mean Square Error: ", mean_squared_error(y_test3, y_test_predict3))
