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

def RSS_3d(slope, intercept, x_list, y_list, cost_func):
    ret = 0
    for i in range(len(x_list)):
        ret = ret + cost_func(x_list[i], y_list[i], slope, intercept)
    return ret

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
slope_list = [new_slope]
intercept_list = [new_intercept]

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
    slope_list.append(new_slope)
    intercept_list.append(new_intercept)
    dintRSSsum = 0
    dslopeRSSsum = 0









#plotting data with linear regression line to visualize data

#plt.subplot(1,3,1)
plt.scatter(x,y,s=100, color='red', alpha=0.8)
plt.plot(x, linreg(x, new_slope, new_intercept), linewidth = 3, color='blue', alpha=0.6)
plt.style.use('fivethirtyeight')
plt.title('Linear Regression of Data')
plt.xlabel('X')
plt.ylabel('y')


plt.annotate(('Theta0 (intercept)', new_intercept), xy=(0.05, 0.95), xycoords='axes fraction')
plt.annotate(('Theta1 (slope)', new_slope), xy=(0.05, 0.85), xycoords='axes fraction')


# plt.subplot(1,3,2)
# plt.plot(x,RSS())

plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# x_axis = np.array(np.linspace(-50,50,500))
# z_axis = np.array(np.linspace(-50,50,500))
# y_axis = lambda x_axis, z_axis:RSS(x[0],y[0],x_axis,z_axis)

# x_axis,z_axis = np.meshgrid(x_axis, z_axis)


# ax.plot_surface(x_axis,RSS_3d(x_axis, z_axis, x, y, RSS),z_axis)
# ax.set_aspect('equal')
# plt.show()

# print(RSS_3d(10, 10, x, y, RSS))





