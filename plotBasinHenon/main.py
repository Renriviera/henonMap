import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from mpmath import *
import scipy.optimize as opti

############### Attempts to find basin of attractor for henon map with values:
###############  b =-0.52, a = 0.790264237374806
############### where the supersink of the 5-limit cycle is x= 1.7941852550596613, y = -0.04490211648122564,
############### 5-limit cycle refers to it being period 5 point
############### supersink refers to it being point where the jacobian of the n-th iterate of henon map has trace = 0, giving us best possible conditions on size of eigenvalues
aVal26 = 0.7323896410329971
bVal26 = -0.5438438438438438
aVal27 = 0.7471395937083065
bVal27 = -0.5378378378378378
xValM526 = 1.968915312362911 ##26th
yValM526 = -0.03531409684940832
xValM527 = 1.9408440477225315
yValM527 = -0.03421570487548523
xValM426 = 1.8751909380767302
yValM426 = -0.05479183380857264
xValM427 = 1.851288219708184
yValM427 = -0.05235398631318543
def hMap(a, b, x, y):
    return 1 - a * x ** 2 + b * y

def nHMap(a,b,x,y,n):
    x0 = x
    y0 = y
    for i in range(n):
        x1 = x0
        y1 = y0
        xN = hMap(a,b,x1,y1)
        yN = x1
        x0 = xN
        y0 = yN
    return xN,yN
def backwards_hMap(a, b, x, y):
    return -(1 / b) * (1 - a * y ** 2 - x)


def getPeriodic(a, b):
    p1 = (-1 * (1 - b) + np.sqrt((1 - b) ** 2 + 4 * a)) / (2 * a)
    p2 = (-1 * (1 - b) - np.sqrt((1 - b) ** 2 + 4 * a)) / (2 * a)
    return p1, p2


def getUnstableLin(a, b, x):
    evecUn = [-a * x + np.sqrt(b + a ** 2 * x ** 2), 1]
    return evecUn


def getStableLin(a, b, x):
    evecSt = [-a * x - np.sqrt(b + a ** 2 * x ** 2), 1]
    return evecSt


def getjacob(a, b, x):
    A = np.array([[-2 * a * x, b], [1, 0]])
    results = la.eig(A)
    return results[0]


def eigvalue(a, b, x):
    A = np.array([[-2 * a * x, b], [1, 0]])
    results = la.eig(A)
    return results[0]

# px1 = 1.7864808718060363            #########real period 5 point that is preserved
# py1 = -0.024620126046225987
# px1 = 1.7862194654530033            #########real period 5 point that is preserved
# py1 = -0.023437317382338785
px2 = -1.6113670517890801
py2 = 1.7862194654530033
px3 = -2.0274235117005617
py3 = -1.6113670517890801
px4 = -1.5739723582587517
py4 = -2.0274235117005617
px5 = -0.023437317382353884
py5 = -1.5739723582587517



x = 1.7329245955716046
y = -0.04325860269609838
x1 = -1.4078037868495432
y1 = 1.7329245955716046
x2 = -1.4694653088060274
y2 = -1.4078037868495432
x3 = -0.04325860269609538
y3 = -1.4694653088060274

# bVal5 = -0.49979979979979977#######works for both of the above
# aVal5 = 0.8221329776354395######## works for both of the above
fixedX = 0.51906516
fixedY = 0.51906516

bVal = -0.52 ########try b= -.38
aVal = 0.790264237374806
xi =  1.7941852550596613
yi = -0.04490211648122564

# fig2, ax2 = plt.subplots()
# fig3, ax3 = plt.subplots()
# fig4, ax4 = plt.subplots()
# fig5, ax5 = plt.subplots()
# fig6, ax6 = plt.subplots()
# fig7, ax7 = plt.subplots()
for k in range(4):
                xO = xi
                yO = yi
                xN = hMap(aVal, bVal, xO, yO)
                yN = xO
                xi = xN
                yi = yN
                print(xN,yN)
                print(k, 'stop')

n=5
aVal5 = 1.737699525748658568
bVal5 = -0.10903010033444815
px1 = 1.0875115790712062
py1 = -0.007247340278146497
# x_list = np.linspace(px1 - 0.000001, px1 + 0.000001, 101)
# y_list = np.linspace(py1 - 0.000001, py1 + 0.000001, 101)
xi = 1.4992214924481013
yi = -0.02205466462428818
aVal = 1.01235834715269949
bVal = -0.4057057057057057
x_list = np.linspace(px1 - 0.0005, px1 + 0.0005, 2000)
y_list = np.linspace(py1 - 0.0005, py1 + 0.0005, 2000)
arrayX, arrayY = np.meshgrid(x_list,y_list)
XY = nHMap(aVal5, bVal5, arrayX, arrayY,50)
graph1x, graph1y = nHMap(aVal5, bVal5, arrayX, arrayY,4)
graph2x, graph2y = nHMap(aVal5, bVal5, graph1x, graph1y,4)
graph3x, graph3y = nHMap(aVal5, bVal5, graph2x, graph2y,4)
graph4x, graph4y = nHMap(aVal5, bVal5, graph3x, graph3y,4)

# ax2.plot(arrayX,arrayY,color = 'red',label='Box', alpha= 0.5)
# ax2.set(xlim=(x - 1, x + 1), ylim=(y - 1, y + 1))
# ax2.set(xlim=(x1 - 1, x1 + 1), ylim=(y1 - 1, y1 + 1)) ### period 5 supersink

# ax2.plot(graph1x,graph1y,color = 'blue',label='First iterate', alpha= 0.5)
# ax2.plot(graph2x,graph2y,color = 'yellow',label='Second iterate', alpha= 0.5)
# ax2.plot(graph3x,graph3y,color = 'green',label='Third iterate', alpha= 0.5)
# ax2.plot(graph4x,graph4y,color = 'pink', label='Fourth iterate', alpha= 0.5)
# ax2.legend([arrayY, graph1y, graph2y, graph3y, graph4y], ['Box','First iterate', 'Second iterate', 'Third iterate','Fourth iterate'])
# ax2.set_title('4x Iterates of point cloud for x=1,2,3,4')

# ax3.set(xlim=(x - 0.000002, x + 0.000002), ylim=(y - 0.000002, y + 0.000002))
# ax7.plot(arrayX,arrayY, color = 'red')
# ax7.set_title('initial box')
#
# ax3.plot(graph1x,graph1y,color = 'red')
# ax3.set_title('4x Iterates of point cloud for x=1')
# ax4.plot(graph2x,graph2y,color = 'red')
# ax4.set_title('4x Iterates of point cloud for x=2')
# ax5.plot(graph3x,graph3y,color = 'red')
# ax5.set_title('4x Iterates of point cloud for x=3')
# ax6.plot(graph4x,graph4y,color = 'red')
# ax6.set_title('4x Iterates of point cloud for x=4')








print(XY)
# print('here')
# vals = np.unique(XY[0],return_counts=True)
# print('first',vals[0])
# print(vals[0][0])
# print(vals[0][1])
# print(vals[0][2])
# print('second', vals[1])
# total = 0
# for i in range(len(vals[1])):
#     total = vals[1][i] + total
#
# for i in range(len(vals[1])):
#     print('Percentage of trajectories going to:', vals[0][i], ' = ', vals[1][i] / total)
# for i in vals[0]:
    # print(vals[1])
# print(vals[2])
# print(vals[0] == vals[1])
# print(vals)
# print('ycoords')
# print(np.unique(XY[1], return_counts=True))
# print('stop')
fig = plt.figure(figsize = (11,11))
# ax1 = fig.add_subplot(1,1,1)
# ax1.set(xlim=(-6, 6), ylim=(-6, 6))
plt.pcolormesh(x_list,y_list,XY[0],cmap='viridis')
# plt.plot(x,y, 'x', color = 'red')
# plt.plot(x1,y1, 'x', color = 'red')
# plt.plot(x2,y2, 'x', color = 'red')
# plt.plot(x3,y3, 'x', color = 'red')
# plt.plot(fixedX,fixedY,'+', color = 'cyan')
plt.plot(px1,py1, 'o', color = 'black')
# plt.plot(px2,py2, 'o', color = 'black')
# plt.plot(px3,py3, 'o', color = 'black')
# plt.plot(px4,py4, 'o', color = 'black')
# plt.plot(px5,py5, 'o', color = 'black')
#
# plt.plot(xi,yi, 'o', color = 'black')

# for i in range(1000):
#     xO = xi
#     yO = yi
#     xN = hMap(aVal, bVal, xO, yO)
#     yN = xO
#     xi = xN
#     yi = yN
#     print(i,xN,yN)
####testvals:
#### x= 1.7941852550596613, y = -0.04490211648122564, b =-0.52, a = 0.790264237374806
def findBasin(attractorX, attractorY,a,b):
    x_list = np.linspace(attractorX - 0.5, attractorX + 0.5,2000)
    y_list = np.linspace(attractorY - 0.5, attractorY + 0.5,2000)
    finalXList = []
    finalYList = []
    # array = np.meshgrid(x_list,y_list)
    for i in x_list:
        print(i,'here')
        xi = i
        for j in y_list:
            yi = j
            for k in range(1000):
                xO = xi
                yO = yi
                xN = hMap(a, b, xO, yO)
                yN = xO
                xi = xN
                yi = yN
                if(abs(xN) > 10 or abs(yN) > 10):
                    break
                elif(xN == yN): ### converse not true must fix
                    break
                elif k == 999:
                    if( xN not in finalXList or yN not in finalYList):
                        finalXList.append(xN)
                        finalYList.append(yN)
    return finalXList, finalYList
aVal26 = 0.7323896410329971
bVal26 = -0.5438438438438438
aVal27 = 0.7471395937083065
bVal27 = -0.5378378378378378
xValM526 = 1.968915312362911 ##26th
yValM526 = -0.03531409684940832
xValM527 = 1.9408440477225315
yValM527 = -0.03421570487548523
xValM426 = 1.8751909380767302
yValM426 = -0.05479183380857264
xValM427 = 1.851288219708184
yValM427 = -0.05235398631318543


# diffpointX, diffpointY = findBasin(xValM526,yValM526,aVal26,bVal26)
# print(diffpointX, diffpointY)
# print('X',len(basinX))
# print(basinX)
# print('Y',len(basinY))
# print(basinY)
# plt.plot(x,y,'X', color= 'red')
# plt.plot(basinX,basinY,'.', color= 'green', markersize=0.5)
plt.show()