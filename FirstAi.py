import math, copy
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0])   #features
y_train = np.array([300.0, 500.0])  

def computeCost(x,y,m,b):
    cost = 0  
    xLenth = x.shape[0]
    for i in range(xLenth):
        funXY = m * x[i] + b
        cost += (y[i] - funXY)**2
    
    totalCost = 1/(2 * m) * cost if m != 0 else cost
    return totalCost

def computeGradient(x,y,m,b):
    xLenth = x.shape[0]
    tempX = 0
    tempY = 0
    for i in range(xLenth):
        funXY = m*x[i] + b
        tempX += (funXY-y[i]) * x[i]
        tempY += (funXY-y[i]) 
    d_x = (1/xLenth) * tempX
    d_y = (1/xLenth) * tempY
    return(d_x, d_y)

def gradentDesent(x,y,m,b,alpha,numIterations):
    totalM = m 
    totalB = b
    for i in range(numIterations):
        d_x, d_y = computeGradient(x,y,totalM,totalB)
        totalM = totalM - alpha * d_x
        totalB = totalB - alpha * d_y
    return totalM, totalB

m = 0
b = 0   
iterations = 100000
alpha = .001
print("starting cost: " +  str(computeCost(x_train,y_train,m,b)))
m,b = gradentDesent(x_train,y_train,m,b,alpha,iterations)
print(f"m = {m}, b = {b}")
print("final cost: " +  str(computeCost(x_train,y_train,m,b)))
