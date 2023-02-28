
##Bai tap 1
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Change to data path on your computer
data = pd.read_csv("/home/quan/Documents/MachineLearning_Thuhanh/Buoi2/SAT_GPA.csv")
# Show the description of data
data.describe()
# Set to training data (x, y)
y = np.array(data['GPA']).reshape(-1, 1)
x = np.array(data['SAT']).reshape(-1, 1)

regr = linear_model.LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=60, shuffle=False)
model = regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
yhat = regr.intercept_ + x * regr.coef_
# Remind that we need to put component x_0 = 1 to x
plt.scatter(x, y)
plt.scatter(X_test, y_test, color='b')
plt.plot(X_test, y_pred, color='k')
fig = plt.plot(x, yhat, lw=4, c='orange', label='regression line')
plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize=20)
plt.show()





##Baitap2
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

data = pd.read_csv('/home/quan/Documents/MachineLearning_Thuhanh/Buoi2/vidu4_lin_reg.txt', sep=" ", header=0)
print(data) 
regr = linear_model.LinearRegression()
y_data = data.iloc[:, -1]
x_data = data.iloc[:, 1:6]

print(x_data)
print(y_data)

regr.fit(x_data, y_data)
res = list(zip(x_data.columns.tolist(), regr.coef_))
for o in res:
    print("{: >20}: {: >10}".format(*o))


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, shuffle=False)
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)
print(mean_absolute_error(y_pred, y_test))
print(mean_squared_error(y_pred, y_test))





##bai tap 3
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_squared_error

data = pd.read_csv("/home/quan/Documents/MachineLearning_Thuhanh/Buoi2/real_estate.csv")
data

y_data = data.iloc[:, -1]
x_data = data.iloc[:, 1:7]
#lấy phần nguyên của năm giao dịch
x_data['X1 transaction date'] = data['X1 transaction date'].apply(int)
lregr = linear_model.LinearRegression()
#chia dữ liệu thành bộ training và validation
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=350, shuffle=False)
lregr.fit(x_train, y_train)
#dự đoán
y_pred = lregr.predict(x_test)
sse = ((y_test - y_pred) ** 2).sum()
print('Tổng bình phương sai số của dự đoán:', sse)