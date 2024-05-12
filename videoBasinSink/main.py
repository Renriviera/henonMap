import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import scipy.linalg as la
import mpmath as mp

bvalues = []
avalues = []
xvalues = []
yvalues = []

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

# −b−1(1 − ayn2 − bxn)
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


def eigvalue(a, b, x):
    A = np.array([[-2 * a * x, b], [1, 0]])
    results = la.eig(A)
    return results[0]


def findorbit(x, y, a, b, M):
    xi = x
    yi = y
    for j in range(M):
        xO = xi
        yO = yi
        xN = hMap(a, b, xO, yO)
        yN = xO
        xi = xN
        yi = yN
    return xN, yN


def findorbit1(x, y, a, b, M):
    xi = x
    yi = y
    for j in range(M):
        xO = xi
        yO = yi
        xN = yO
        yN = backwards_hMap(a, b, xO, yO)
        xi = xN
        yi = yN
    return xN, yN
period6A = 1.250326450672757
period6B = -0.27692307692307694
pxFinal = -0.02132989902
pyFinal = -0.7566328923
bPoint1 = -0.27
aPoint1 = 1.216294986
bPoint2 = -0.27005
aPoint2 = 1.2162201
midbPoint = (bPoint1+bPoint2)/2
midaPoint = (aPoint1+aPoint2)/2
slopeAB = (aPoint1-aPoint2)/(bPoint1-bPoint2)
intercept = aPoint1-slopeAB*bPoint1
inverseSlopeFinal = -1/slopeAB
interceptFinal = midaPoint - inverseSlopeFinal*midbPoint
bLine = np.linspace(bPoint1,bPoint2,100)
bLineBig = np.linspace(-0.2555,-0.29,400)
print(np.nonzero(bLineBig==-0.27002632))
# print(bLineBig)
aLine = slopeAB*bLine +intercept
aLineFinal = inverseSlopeFinal*bLineBig+interceptFinal

# fig1, ax1 = plt.subplots()
# ax1.plot(bLine,aLine, color = 'red')
# ax1.plot(bLine,aLineFinal, color = 'blue')
# ax1.set(xlim=(bPoint1-0.0001, bPoint1+0.0001), ylim=(aPoint1-0.0001, aPoint1+0.0001))
# with open('period 3 data - Sheet1.tsv') as f:
#     lines = f.readlines()
#     for i in range(1,len(lines)):
#         temp = lines[i].split(',')
#         bvalues.append(float(temp[0].strip()))
#         avalues.append(float(temp[1].strip()))
#         xvalues.append(float(temp[2].strip()))
#         yvalues.append(float(temp[3].strip()))


k=0
aVal = aLineFinal[k]
bVal = bLineBig[k]
# px1 = xvalues[k]
# py1 = yvalues[k]
# px1 = 1.20534725
# py1 = -0.022882752903382264
# px2 = -0.81021502
# py2 = 1.20534725
# px3 = -0.15456325
# py3 = -0.81021502
# px4 = 1.19449719
# py4 = -0.15456325
# px5 = -0.74119308
# py5 = 1.19449719
# px6 = -0.01767216
# py6 = -0.74119308
# aVal = period6A
# bVal = period6B
x_list = np.linspace(pxFinal - 2, pxFinal + 2, 2000)
y_list = np.linspace(pyFinal - 2, pyFinal + 2, 2000)
arrayX, arrayY = np.meshgrid(x_list,y_list)
XY = nHMap(aVal, bVal, arrayX, arrayY,103)
# print(XY[0])
fig = plt.figure(figsize = (11,11))
image = plt.pcolormesh(x_list,y_list,XY[0],cmap='viridis')
# plt.plot(px1,py1, 'o', color = 'black', markersize = 0.5 )
# plt.plot(px2,py2, 'o', color = 'black', markersize = 0.5 )
# plt.plot(px3,py3, 'o', color = 'black', markersize = 0.5 )
# plt.plot(px4,py4, 'o', color = 'black', markersize = 0.5 )
# plt.plot(px5,py5, 'o', color = 'black', markersize = 0.5 )
# plt.plot(px6,py6, 'o', color = 'black', markersize = 0.5 )

#
def animate (i):
    aVal = aLineFinal[i]
    bVal = bLineBig[i]
    px1 = pxFinal
    py1 = pyFinal
    # px2 = hMap(aVal,bVal,px1,py1)
    # py2 = px1
    # px3 = hMap(aVal,bVal,px2,py2)
    # py3 = px2
    x_list = np.linspace(px1 - 2, px1 + 2, 2000)
    y_list = np.linspace(py1 - 2, py1 + 2, 2000)
    arrayX, arrayY = np.meshgrid(x_list, y_list)
    XY = nHMap(aVal, bVal, arrayX, arrayY, 103)
    image.set_array(XY[0][:-1,:-1])
    plt.title('Basin corresponding to index '+ str(i)+ '')
    plt.plot(px1, py1, 'o', color='black')
    # plt.plot(px2, py2, 'o', color = 'black')
    # plt.plot(px3, py3, 'o', color = 'black')
    print(i)


anim = animation.FuncAnimation(fig,animate, frames = list(range(k,400)), interval=25)
anim.save('basin3-neighborhood2.gif')
plt.show()