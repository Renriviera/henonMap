import numpy as np
import matplotlib.pyplot as plt
import bisect
import scipy.linalg as la
import scipy.optimize as opti
import sklearn as skl
from skopt import gp_minimize
from mpmath import *

mp.dps = 30

def hMap(a, b, x, y):
    return 1 - a * x ** 2 + b * y


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

def testpoly(x):
    return x**2 +2*x+5
def paraboloid(x,y):
    return x**2+y**2+6
def testline(x):
    return 4

def pointsFit(a, b, N):
    density = 10000
    density1 = 10
    rho = 0.2
    p1, p2 = getPeriodic(a, b)
    p = p2
    ############ for stable  #############
    eigstable = eigvalue(a, b, p)[0].real
    left_cut_b = 0
    if b > -0.32:
        right_cut_b = 5 * 10 ** -(N - 1)
    if -0.45 < b <= -0.32:
        right_cut_b = 5 * 10 ** -(N - 3)
    if b <= -0.45:
        right_cut_b = 9 * 10 ** -(N - 4)
    evecSt = getStableLin(a, b, p)  # eigenvector for stable manifold

    dx_b = evecSt[0] * rho
    dy_b = evecSt[1] * rho

    testxline = mp.linspace(p + (left_cut_b * dx_b), p + (dx_b * right_cut_b), density)
    testyline = mp.linspace(p + (left_cut_b * dy_b), p + (dy_b * right_cut_b), density)
    xi_b = testxline
    yi_b = testyline

    if (True):  # to conveniently hide backward if needed
        for i in range(N):
            xOld_b = xi_b
            yOld_b = yi_b
            yNew_b = [backwards_hMap(a, b, xOld_b[j], yOld_b[j]) for j in range(density)]
            xNew_b = yOld_b
            xi_b = xNew_b
            yi_b = yNew_b
    lyNew_b = list(yNew_b)
    lxNew_b = list(xNew_b)
    ais = set()
    for i in range(density):
        if lxNew_b[i] > 1:
            ais.add(i)
    bis = set()
    for i in range(density):
        if lyNew_b[i] > -0.5 and lyNew_b[i] < 0.5:
            bis.add(i)

    iss = ais.intersection(bis)

    left_ind = min(iss)
    right_ind = max(iss)

    intelen = right_cut_b - left_cut_b

    left_cut_b1 = left_cut_b + intelen * left_ind / density
    right_cut_b1 = left_cut_b + intelen * right_ind / density

    xi_b1 = mp.linspace(p + (left_cut_b1 * dx_b), p + (dx_b * right_cut_b1), density1)
    yi_b1 = mp.linspace(p + (left_cut_b1 * dy_b), p + (dy_b * right_cut_b1), density1)

    xNew_b1, yNew_b1 = xi_b1, yi_b1

    if (True):  # to conveniently hide backward if needed
        for i in range(N):
            xOld_b1 = xi_b1
            yOld_b1 = yi_b1
            # yNew_b1 = backwards_hMap(a,b,xOld_b1,yOld_b1)
            yNew_b1 = [backwards_hMap(a, b, xOld_b1[j], yOld_b1[j]) for j in range(density1)]
            xNew_b1 = yOld_b1
            xi_b1 = xNew_b1
            yi_b1 = yNew_b1

    lxNewb = [float(xNew_b1[k]) for k in range(density1)]#####
    lyNewb = [float(yNew_b1[k]) for k in range(density1)]#####

    # model1 = np.poly1d(np.polyfit(lxNewb, lyNewb, 1))
    # lmodel1 = list(model1.c)
    # coef1 = ["al", "bl"]
    # result1 = dict(zip(coef1, lmodel1))
    ############################# for unstable ##################################
    eigunstable = eigvalue(a, b, p)[1].real
    rightcut17 = 2.5 * 10 ** -(7)
    left_cut = 0
    right_cut = rightcut17 * (eigunstable * 2) ** (N - 17)  # 0.7 too large 0.07 too small
    evecUn = getUnstableLin(a, b, p)
    dx = evecUn[0] * rho
    dy = evecUn[1] * rho
    testxline = np.linspace(p + (left_cut * dx), p + (dx * right_cut), density)
    testyline = np.linspace(p + (left_cut * dy), p + (dy * right_cut), density)
    xi = testxline
    yi = testyline
    # xNew, yNew = xi, yi
    for i in range(N):
        xOld = xi
        yOld = yi
        xNew = hMap(a, b, xOld, yOld)
        yNew = xOld
        xi = xNew
        yi = yNew
    lxNew = list(xNew)
    lyNew = list(yNew)
    inx = lxNew.index(max(lxNew))
    intelen = right_cut - left_cut
    sis = set()
    tis = set()
    for i in range(density):
        if yNew[i] > -0.5 and yNew[i] < 0.5:
            sis.add(i)

    for i in range(density):
        if xNew[i] > 1:
            tis.add(i)
    iss = sis.intersection(tis)
    leftind = min(iss)

    left_cut1 = left_cut + intelen * leftind / density
    right_cut1 = left_cut + intelen * (2 * inx - leftind) / density

    xline = np.linspace(p + (left_cut1 * dx), p + (dx * right_cut1), density1)
    yline = np.linspace(p + (left_cut1 * dy), p + (dy * right_cut1), density1)
    xi1 = xline
    yi1 = yline
    # xNew, yNew = xi, yi
    for i in range(N):
        xOld1 = xi1
        yOld1 = yi1
        xNew1 = hMap(a, b, xOld1, yOld1)
        yNew1 = xOld1
        xi1 = xNew1
        yi1 = yNew1
    quadxNew = list(xNew1)
    quadyNew = list(yNew1)
    return lxNewb, lyNewb, quadxNew, quadyNew
    # model = np.poly1d(np.polyfit(quadyNew, quadxNew, 2))
    # lmodel = list(model.c)
    # coef = ["aq", "bq", "cq"]
    # result = dict(zip(coef, lmodel))
    # # result["aq"] is my coefficient
    # y1 = ((1 / result1["al"]) - result["bq"]) / (2 * result["aq"])
    # x1 = result["aq"] * y1 ** 2 + result["bq"] * y1 + result["cq"]
    # x2 = (1 / result1["al"]) * y1 - result1["bl"] / result1["al"]
    # return (x1 - x2)

#######Test Lagrangian works##########
toler = 0.01
noisePara = 0.05*np.random.normal(0,1,5)
noiseLine = 0.05*np.random.normal(0,1,5)
# print('noisePara',noisePara)
parabolicPointsX = [-2,1,0,1,2]
linearPointsX = [-4,-2,1,3,4]
parabolicPointsY =[testpoly(parabolicPointsX[i]+noisePara[i]) for i in range(len(parabolicPointsX))]
linearPointsY = [testline(linearPointsX[i]+noiseLine[i]) for i in range(len(linearPointsX))]
# print(linearPointsY)
# print(parabolicPointsY)
#####################################
def findpar(xspace,yspace,deltaa):### give yellow part
    a = 1.07905729
    nxi = xspace
    nyi = yspace
    M=4
    for i in range(M):
        nxOld = nxi
        nyOld = nyi
        nxNew = hMap(a-deltaa,-0.4,nxOld,nyOld)
        nyNew = nxOld
        nxi = nxNew
        nyi = nyNew
    xpoint=[]
    ypoint=[]
    for i in range(5000):
        if nyNew[i] <= 0.25 and nyNew[i] >= -0.3:
            ypoint.append(nyNew[i])
            xpoint.append(nxNew[i])
    return xpoint,ypoint

def findpar2(xspace,yspace,deltaa,M):### give yellow part
    a = 1.07905729
    b = -0.4
    nxi = xspace
    nyi = yspace
    for i in range(M):
        nxOld = nxi
        nyOld = nyi
        nxNew = hMap(a-deltaa,b,nxOld,nyOld)
        nyNew = nxOld
        nxi = nxNew
        nyi = nyNew
    xpoint=[]
    ypoint=[]
    for i in range(5000):
        if nyNew[i] <= 0.25 and nyNew[i] >= -0.3 and nxNew[i] >1:
            ypoint.append(nyNew[i])
            xpoint.append(nxNew[i])
    length = len(xpoint)
    resultx = [xpoint[0],xpoint[round(length/10)],xpoint[round(2*length/10)],xpoint[round(3*length/10)],xpoint[round(4*length/10)],xpoint[round(5*length/10)],xpoint[round(6*length/10)],xpoint[round(7*length/10)],xpoint[round(8*length/10)],xpoint[round(9*length/10)],xpoint[-1]]
    resulty = [ypoint[0],ypoint[round(length/10)],ypoint[round(2*length/10)],ypoint[round(3*length/10)],ypoint[round(4*length/10)],ypoint[round(5*length/10)],ypoint[round(6*length/10)],ypoint[round(7*length/10)],ypoint[round(8*length/10)],ypoint[round(9*length/10)],ypoint[-1]]
    return resultx,resulty

xStable, yStable , xUnstable, yUnstable = pointsFit(1.07905729,-0.4,15)
print(xStable,yStable, 'stable manifold')
print(xUnstable,yUnstable,'unstable manifold')


def lagrangian(params):
    a,b,c,d,e = params
    linearFit = 0
    parabolicFit = 0
    tangencyCondition = ((b-e-((d-a)**2/(4*c)))**2)/((b/a)**2+1)
    for i in range(len(xStable)):
        linearFit = linearFit + (xStable[i] - a*yStable[i]-b)**2
    for i in range(len(xUnstable)):
        parabolicFit = parabolicFit+(xUnstable[i] - c*yUnstable[i]**2-d*yUnstable[i]-e)**2
    return linearFit+parabolicFit+tangencyCondition
x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
res = opti.minimize(lagrangian,x0)
# print(res)
# final = opti.minimize(testpoly,[1])
tangencyY = (res.x[0]-res.x[3])/(2*res.x[2])
tangencyX = res.x[2]*tangencyY**2+res.x[3]*tangencyY+res.x[4]
def initialPosition(params):
    points = 10000
    a = params
    perpSlope = -1/res.x[0]
    perpInter = tangencyX - (perpSlope*tangencyY)
    yVals = np.linspace(tangencyY-2, tangencyY+2,points)
    xVals = np.zeros(points)
    distancesSquare = np.zeros(points)
    for i in range(points):
        xVals[i] = perpSlope * yVals[i] + perpInter
    xi = xVals
    yi = yVals
    for i in range(4):
        xOld = xi
        yOld = yi
        yNew = xOld
        xNew = [hMap(a, -0.4, xOld[j], yOld[j]) for j in range(len(xi))]  ### iterate perpendicular through henon
        xi = xNew
        yi = yNew
    for i in range(points):
        distancesSquare[i] = np.sqrt((xVals[i]-xi[i])**2+(yVals[i]-yi[i])**2)

    return np.min(distancesSquare)
# initialBounds = ((1.07905729-0.2,1.07905729+0.2))
# initialRes = opti.minimize(initialPosition,[1.07905729],bounds= initialBounds, tol=1e-6)
# points = 10000
# perpSlope = -1/res.x[0]
# perpInter = tangencyX - (perpSlope*tangencyY)
# yVals = np.linspace(tangencyY-2, tangencyY+2,points)
# xVals = np.zeros(points)
# distancesSquare = np.zeros(points)
# for i in range(points):
#     xVals[i] = perpSlope * yVals[i] + perpInter
# xi = xVals
# yi = yVals
# for i in range(4):
#     xOld = xi
#     yOld = yi
#     yNew = xOld
#     xNew = [hMap(initialRes.x[0], -0.4, xOld[j], yOld[j]) for j in range(len(xi))]  ### iterate perpendicular through henon
#     xi = xNew
#     yi = yNew
# for i in range(points):
#     distancesSquare[i] = (xVals[i]-xi[i])**2+(yVals[i]-yi[i])**2
# temporary = np.min(distancesSquare)
# indexPoint = np.where(distancesSquare == temporary)[0][0]
# initialPointFinal = [xVals[indexPoint],yVals[indexPoint]]
# print('RUIPENG',initialPosition(1.07905729-0.1))
# print('initial points', initialRes)
# print('answer', initialPointFinal)
def firstPosition(alpha):
    perpSlope = -1 / res.x[0]
    perpInter = tangencyX - (perpSlope * tangencyY)
    yVals = np.linspace(tangencyY - 0.5, tangencyY + 0.5, 5000)
    perplineX = np.zeros(5000)
    for i in range(5000):
        perplineX[i] = perpSlope * yVals[i] + perpInter
    perpFinalX, perpFinalY = findpar(perplineX, yVals, alpha)
    xi = perpFinalX
    yi = perpFinalY
    for i in range(4):
        xOld = xi
        yOld = yi
        yNew = xOld
        xNew = [hMap(1.07905729-alpha, -0.4, xOld[j], yOld[j]) for j in range(len(xi))] ### iterate perpendicular through henon
        xi = xNew
        yi = yNew
    modelPar = np.poly1d(np.polyfit(yNew, xNew, 2))
    lmodelPar = list(modelPar.c)
    coef1 = ["c", "d", "e"]
    result1 = dict(zip(coef1, lmodelPar))
    y1= ((perpSlope-result1["d"])+sqrt((result1["d"]-perpSlope)**2-(4*result1["c"]*(result1["e"]-perpInter))))/(2*result1["c"])
    y2= ((perpSlope-result1["d"])-sqrt((result1["d"]-perpSlope)**2-(4*result1["c"]*(result1["e"]-perpInter))))/(2*result1["c"])
    x1 = perpSlope*y1+perpInter
    x2 = perpSlope*y2+perpInter
    if x1 > 0.5:
        intx, inty = x1, y1
    elif x1 <= 0.5:
        intx, inty = x2, y2
    return [intx,inty]
print('HERE WE GO', firstPosition(0.01))
def lagrangianUnfold(params):
    alpha,a,b,pX,pY = params
    perpSlope = -1/a
    perpInter = pX - (perpSlope*pY)
    yperpVals = np.linspace(pY - 1, pY + 1, 5000)
    perplineX = np.zeros(5000)
    for i in range(5000):
        perplineX[i] = perpSlope * yperpVals[i] + perpInter

    perpFinalX, perpFinalY = findpar2(perplineX,yperpVals,alpha,4)
    xi = perpFinalX
    yi = perpFinalY

    for i in range(4):
        xOld = xi
        yOld = yi
        yNew = xOld
        xNew = [hMap(1.07905729-alpha, -0.4, xOld[j], yOld[j]) for j in range(len(xi))] ### iterate perpendicular through henon
        xi = xNew
        yi = yNew
    model1 = np.poly1d(np.polyfit(yNew, xNew, 2))
    lmodel1 = list(model1.c)
    coef1 = ["c", "d", "e"]
    result1 = dict(zip(coef1, lmodel1))
    yDer = (a-result1["d"])/2*result1["c"]
    xFinal1 = a*yDer+b                          ## x-coord tangency on line
    xFinal2 = result1["c"]*yDer**2+result1["d"]*yDer+result1["e"] ## x-coord tangency on parabola

    midTangent = (xFinal2 + xFinal1)/2
    xj = midTangent
    yj = yDer
    pXj = pX
    pYj = pY
    henArray =[]
    for i in range(4):
        henArray.append(np.array([[-2 * (1.07905729 - alpha) * pXj, -0.4], [1, 0]]))
        pXOld = pXj
        pYOld = pYj
        xPeriodOld = xj
        yPeriodOld = yj
        yPeriodNew = xj
        xPeriodNew = hMap(1.07905729-alpha, -0.4, xPeriodOld, yPeriodOld)
        pXNew = hMap(1.07905729-alpha, -0.4, pXOld, pYOld)
        pYNew = pXj
        pXj = pXNew
        pYj = pYNew
        xj = xPeriodNew
        yj = yPeriodNew
    finalMatrix =np.array([[1,0],[0,1]])
    for i in range(4):
        finalMatrix = np.matmul(finalMatrix,henArray[3-i])
     ## jacobian for henon map at pX

    return ((xFinal1-xFinal2)**2 + (xj - (midTangent)/2)**2 + (yj - yDer)**2 + np.trace(finalMatrix))**2##+(pX-midTangent)**2+(pY-yDer)**2)

# foldBounds = ((-0.015,0.015),(res.x[0]-0.05,res.x[0]+0.05),(res.x[1]-0.05, res.x[1]+0.05))
# unfoldRes = opti.differential_evolution(lagrangianUnfold, bounds=foldBounds)
foldBounds = ((-0.1,0.1),(None,None),(None, None),(firstPosition(0.01)[0]-0.5,firstPosition(0.01)[0]+0.5),(firstPosition(0.01)[1]-0.5,firstPosition(0.01)[1]+0.5))
unfoldRes = opti.minimize(lagrangianUnfold,[0.01,res.x[0]-0.05, res.x[1]-0.05, firstPosition(0.01)[0], firstPosition(0.01)[1]], tol= 0.00000001, bounds=foldBounds) ###1.4750747449285, -0.0197361664572
print(unfoldRes)
print(res)
print(res.x)
print('tangency is:',tangencyX,tangencyY)
perpSlope = -1/unfoldRes.x[1]
perpInter = unfoldRes.x[3] - (perpSlope*unfoldRes.x[4])
print(perpSlope)
print(perpInter)
yVals = np.linspace(unfoldRes.x[4]-1, unfoldRes.x[4]+1,100)
yperpVals = np.linspace(unfoldRes.x[4]-0.5, unfoldRes.x[4]+0.5,100)
quadX = np.zeros(100)
lineX = np.zeros(100)
lineXr = np.zeros(100)
perplineX = np.zeros(100)
for i in range(100):
    quadX[i] = res.x[2]*yVals[i]**2+res.x[3]*yVals[i] + res.x[4]
    lineX[i] = res.x[0]*yVals[i]+res.x[1]
    perplineX[i] = perpSlope*yperpVals[i] + perpInter
    lineXr[i] =  unfoldRes.x[1]* yVals[i] + unfoldRes.x[2]
    # lineX[i] = -0.1380877 * yVals[i] + 1.62805374
    # perplineX[i] = 7.241774459667905 * yperpVals[i] + 1.7798521830140956

xi = perplineX
yi = yperpVals
for i in range(4):
    xOld = xi
    yOld = yi
    yNew = xOld
    xNew = [hMap(1.07905729-unfoldRes.x[0], -0.4, xOld[j], yOld[j]) for j in range(100)]
    xi = xNew
    yi = yNew
print('henon mapped perp',xNew,yNew)
model1 = np.poly1d(np.polyfit(yNew, xNew, 2))
lmodel1 = list(model1.c)
coef1 = ["c", "d", "e"]
result1 = dict(zip(coef1, lmodel1))
yDer = (unfoldRes.x[1]-result1["d"])/2*result1["c"]
xFinal1 = unfoldRes.x[1]*yDer+unfoldRes.x[2]
xFinal2 = result1["c"]*yDer**2+result1["d"]*yDer+result1["e"]
    # vertexY = -1*result1["d"]/2*result1["c"]
    # vertexX = result1["c"]*vertexY**2+result1["d"]*vertexY+result1["e"]
    # originalY = (vertexX - b)/a
xj = xFinal2
yj = yDer
pXj = unfoldRes.x[3]
pYj = unfoldRes.x[4]
henArray =[]
for i in range(4):
    henArray.append(np.array([[-2 * (1.07905729 - unfoldRes.x[0]) * pXj, -0.4], [1, 0]]))
    xPeriodOld = xj
    yPeriodOld = yj
    yPeriodNew = xj
    xPeriodNew = hMap(1.07905729-unfoldRes.x[0], -0.4, xPeriodOld, yPeriodOld)
    pXOld = pXj
    pYOld = pYj
    pXNew = hMap(1.07905729 - unfoldRes.x[0], -0.4, pXOld, pYOld)
    pYNew = pXj
    pXj = pXNew
    pYj = pYNew
    xj = xPeriodNew
    yj = yPeriodNew

finalMatrix =np.array([[1,0],[0,1]])
for i in range(4):
    finalMatrix = np.matmul(finalMatrix,henArray[3-i])
henMatrix = np.array([[-2*(1.07905729-unfoldRes.x[0])*unfoldRes.x[3],-0.4],[1,0]])
print('tangency distance', xFinal1-xFinal2)
print('periodic distance', sqrt((xFinal2-xj)**2+(yDer-yj)**2) )
print('Trace of Jacobian', np.trace(finalMatrix))
print('Distance between (Px,Py) and tangency',sqrt((unfoldRes.x[3]-(xFinal1+xFinal2)/2)**2+(unfoldRes.x[4]-yDer)**2))

plt.xlim(-1.5,2.5)
plt.ylim(-1.5,2.5)
plt.plot(unfoldRes.x[3],unfoldRes.x[4],'X', color = 'yellow')
plt.plot(quadX,yVals,'-',color='black') ###unstable
plt.plot(lineX,yVals,'-',color='red') ###stable
plt.plot(lineXr,yVals,'-',color='purple') ###line varied over
plt.plot(perplineX,yperpVals,'-',color='blue') ### perp
plt.plot(xNew,yNew,'-',color='green') ### henon mapped points

plt.show()

