import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Change to data path on your computer
data =pd.read_csv("/home/quan/Documents/MachineLearning/SAT_GPA.csv")
# Show the description of data
data.describe()
# Set to training data (x, y)
y = data['GPA']
x = data['SAT']
# Remind that we need to put component x_0 = 1 to x
plt.scatter(x,y)
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)

plt.scatter(x1,y)
yhat = t_1*x1 + t_0
fig = plt.plot(x1,yhat, lw=4, c='orange', label = 'regression line')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()