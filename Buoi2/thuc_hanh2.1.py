import math
import matplotlib.pyplot as plt
import numpy as np
with open('fuel.txt') as f:lines = f.readlines()
x_data = []
y_data = []
lines.pop(0)
for line in lines:splitted = line.replace('\n', '').split(',')
splitted.pop(0)
splitted = list(map(float, splitted))
fuel = 1000 * splitted[1] / splitted[5]
dlic = 1000 * splitted[0] / splitted[5]
logMiles = math.log2(splitted[3])
y_data.append([fuel])
x_data.append([splitted[-1], dlic, splitted[2], logMiles])
x_data = np.asarray(x_data)
y_data = np.asarray(y_data)