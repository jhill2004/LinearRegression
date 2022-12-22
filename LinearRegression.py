import matplotlib.pyplot as plt
import numpy as np


# Using residual sum of squares as cost function

def RSS(x,y,slope,intercept):
    return (((y)-(intercept + slope*x))**2)

def dinterceptRSS(x,y,slope,intercept):
    return (-2 * ((y)-(intercept + slope*x)))

def dslopeRSS(x,y,slope,intercept):
    return (-2 * x * ((y)-(intercept + slope*x)))

def linreg(x,slope,intercept):
    return (slope * x + intercept)

# Getting coords from user and then putting it into numpy array

print("Enter x and y coordinate points, separated by spaces. Enter 'q' to quit.")

x = []
y = []
while True:
    line = input()
    if line == 'q':
        break


    coords = line.split()
    x.append(float(coords[0]))
    y.append(float(coords[1]))


x = np.array(x)
y = np.array(y)

# change these to affect accuracy/time taken to calculate
learning_rate = 0.01
min_step = 0.0001
max_loops = 1000

# variable declarations
old_intercept = 0
old_slope = 1
new_slope = 1
new_intercept = 0

# gradients
dintRSSsum = 0
dslopeRSSsum = 0

# calculating intercept (theta0) for linear regression line

for n in range(max_loops):
    old_intercept = new_intercept
    old_slope = new_slope
    for i in range(len(x)):
        dintRSSsum = dintRSSsum + dinterceptRSS(x[i], y[i], old_slope, old_intercept)
        dslopeRSSsum = dslopeRSSsum + dslopeRSS(x[i], y[i], old_slope, old_intercept)
    if (abs(dintRSSsum * learning_rate) < min_step and abs(dslopeRSSsum * learning_rate) < min_step):
        break

    new_intercept = old_intercept - (learning_rate * dintRSSsum)
    new_slope = old_slope - (learning_rate * dslopeRSSsum)
    dintRSSsum = 0
    dslopeRSSsum = 0









# plotting data with linear regression line to visualize data

plt.scatter(x,y,s=100, color='red', alpha=0.8)
plt.plot(x, linreg(x, new_slope, new_intercept), linewidth = 3, color='blue', alpha=0.6)
plt.style.use('fivethirtyeight')
plt.title('Linear Regression of Data')
plt.xlabel('X')
plt.ylabel('y')


plt.annotate(('Theta0 (intercept)', new_intercept), xy=(0.05, 0.95), xycoords='axes fraction')
plt.annotate(('Theta1 (slope)', new_slope), xy=(0.05, 0.85), xycoords='axes fraction')

plt.show()




