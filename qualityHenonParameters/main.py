import numpy as np
import matplotlib.pyplot as plt

# Henon map (x,y) -> (1 - a*x**2 + b*y,x)
def hMap(a, b, x, y):
    return 1 - a * x ** 2 + b * y


# Inverse of Henon map
def backwards_hMap(a, b, x, y):
    return -(1 / b) * (1 - a * y ** 2 - x)


# get both fixed points of the system
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

def getFixed_np(a, b):
    # p1 = (-1*(1 - b) + sym.sqrt((1-b)**2 + 4*a))/(2*a)
    p2 = (-1 * (1 - b) - np.sqrt((1 - b) ** 2 + 4 * a)) / (2 * a)
    return p2


def getQualityUnst(b, a, x, y):
    x_vals = np.zeros(100)
    y_vals = np.zeros(100)
    x_vals[0], y_vals[0] = x, y

    fp = getFixed_np(a, b)

    curr_diff = np.sqrt((x - fp) ** 2 + (y - fp) ** 2)
    toler = 0.001

    for i in range(100 - 1):
        yNew = backwards_hMap(a, b, x_vals[i], y_vals[i])
        xNew = y_vals[i]

        new_diff = np.sqrt((xNew - fp) ** 2 + (yNew - fp) ** 2)

        if i > 4 and new_diff > curr_diff and new_diff > toler:
            return i
        curr_diff = new_diff
        x_vals[i + 1], y_vals[i + 1] = xNew, yNew
    return 0


def getQualitySt(b,a,x, y):
    x_vals = np.zeros(100)
    y_vals = np.zeros(100)
    x_vals[0],y_vals[0] = x,y

    fp = getFixed_np(a,b)

    curr_diff = np.sqrt((x-fp)**2 + (y-fp)**2)
    toler = 0.001

    for i in range(100-1):
        xNew = hMap(a,b,x_vals[i],y_vals[i])
        yNew = x_vals[i]

        new_diff = np.sqrt((xNew-fp)**2 + (yNew-fp)**2)

        if i > 4 and new_diff > curr_diff and new_diff > toler:
            return i
        curr_diff=new_diff
        x_vals[i+1],y_vals[i+1] = xNew, yNew
    return 0

def tanpoint(a, b, N):
    density = 50000
    density1 = 5
    rho = 0.2
    p1, p2 = getPeriodic(a, b)
    p = p2
    ############ for stable  #############
    # eigstable = eigvalue(a,b,p)[0].real
    left_cut_b = 0
    if b > -0.32:
        right_cut_b = 5 * 10 ** -(N - 2)
    if -0.45 < b <= -0.32:
        right_cut_b = 5 * 10 ** -(N - 3)
    if -0.65 < b <= -0.45:
        right_cut_b = 9 * 10 ** -(N - 4)
    if b <= -0.65:
        right_cut_b = 1 * 10 ** -(N - 6)
    evecSt = getStableLin(a, b, p)  # eigenvector for stable manifold

    dx_b = evecSt[0] * rho
    dy_b = evecSt[1] * rho

    testxline = np.linspace(p + (left_cut_b * dx_b), p + (dx_b * right_cut_b), density)
    testyline = np.linspace(p + (left_cut_b * dy_b), p + (dy_b * right_cut_b), density)
    xi_b = testxline
    yi_b = testyline

    if (True):  # to conveniently hide backward if needed
        for i in range(N):
            xOld_b = xi_b
            yOld_b = yi_b
            yNew_b = backwards_hMap(a, b, xOld_b, yOld_b)
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
        if lyNew_b[i] > -0.5 + (b + 0.3) and lyNew_b[i] < 0.5 - (b + 0.3):
            bis.add(i)

    iss = ais.intersection(bis)
    # print(ais)
    # print(bis)
    left_ind = min(iss)

    right_ind = max(iss)

    intelen = right_cut_b - left_cut_b

    left_cut_b1 = left_cut_b + intelen * left_ind / density
    right_cut_b1 = left_cut_b + intelen * right_ind / density

    xi_b1 = np.linspace(p + (left_cut_b1 * dx_b), p + (dx_b * right_cut_b1), density1)
    yi_b1 = np.linspace(p + (left_cut_b1 * dy_b), p + (dy_b * right_cut_b1), density1)

    xNew_b1, yNew_b1 = xi_b1, yi_b1

    if (True):  # to conveniently hide backward if needed
        for i in range(N):
            xOld_b1 = xi_b1
            yOld_b1 = yi_b1
            yNew_b1 = backwards_hMap(a, b, xOld_b1, yOld_b1)
            xNew_b1 = yOld_b1
            xi_b1 = xNew_b1
            yi_b1 = yNew_b1

    lxNew_b1 = list(xNew_b1)
    lyNew_b1 = list(yNew_b1)
    model1 = np.poly1d(np.polyfit(lxNew_b1, lyNew_b1, 1))
    lmodel1 = list(model1.c)
    coef1 = ["al", "bl"]
    result1 = dict(zip(coef1, lmodel1))
    ############################# for unstable ##################################
    # eigunstable = eigvalue(a,b,p)[1].real
    left_cut = 0
    right_cut = 1 * 10 ** -(N - 8)  # 0.7 too large 0.07 too small
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
        if yNew[i] > -0.5 + (b + 0.3) and yNew[i] < 0.5 - (b + 0.3):
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
    model = np.poly1d(np.polyfit(quadyNew, quadxNew, 2))
    lmodel = list(model.c)
    coef = ["aq", "bq", "cq"]
    result = dict(zip(coef, lmodel))
    # result["aq"] is my coefficient
    y1 = ((1 / result1["al"]) - result["bq"]) / (2 * result["aq"])
    x1 = result["aq"] * y1 ** 2 + result["bq"] * y1 + result["cq"]
    x2 = (1 / result1["al"]) * y1 - result1["bl"] / result1["al"]
    return (x1 + x2) / 2, y1
######Patryk's values############
# filename = "avalue of b -0.7-0.3.txt"
# filenameTanX ='x.txt'
# filenameTanY ='y.txt'
# N10para_ba = np.loadtxt("N10para_ba.txt").reshape(100, 2)
# N10para_xy = np.loadtxt("N10para_xy.txt").reshape(100, 2)
# qualSt = np.zeros(100)
# qualUnst =np.zeros(100)
# for i in range(100):
#     qualSt[i] = getQualitySt(N10para_ba[i][0], N10para_ba[i][1], N10para_xy[i][0], N10para_xy[i][1])
#     qualUnst[i] = getQualityUnst(N10para_ba[i][0], N10para_ba[i][1], N10para_xy[i][0], N10para_xy[i][1])
# fig = plt.figure(figsize = (4,4))
# ax = fig.add_subplot(1,1,1)
# plt.xlabel('b')
# plt.ylabel('Quality')
# b_p = np.zeros(100)
# for i in range(100):
#     b_p[i] = N10para_ba[i][0]
# plt.plot(b_p, qualSt, '-',color='red')
# plt.plot(b_p, qualUnst,'-',color='blue')
# print(N10para_ba)
# print(N10para_ba[2][1])
# blinePatryk = np.zeros(100)
# alinePatryk = np.zeros(100)
# tangentPointsPatrykX = np.zeros(100)
# tangentPointsPatrykY = np.zeros(100)
# qualityStablePatryk = np.zeros(100)
# qualityUnstablePatryk = np.zeros(100)
#
# for i in range(100):
#     blinePatryk[i] = N10para_ba[i][0]
#     alinePatryk[i] = N10para_ba[i][1]
#     print(blinePatryk[i],alinePatryk[i])
#     print('next')
#     tangentPointsPatrykX[i] = N10para_xy[i][0]
#     tangentPointsPatrykY[i] = N10para_xy[i][1]
#     print(tangentPointsPatrykX[i],tangentPointsPatrykY[i])
#     # tangentPointsPatrykX[i],tangentPointsPatrykY[i] = tanpoint(alinePatryk[i],blinePatryk[i],10)
#     qualityUnstablePatryk[i] = getQualityUnst(blinePatryk[i],alinePatryk[i],tangentPointsPatrykX[i],tangentPointsPatrykY[i])
#     qualityStablePatryk[i] = getQualitySt(blinePatryk[i],alinePatryk[i],tangentPointsPatrykX[i],tangentPointsPatrykY[i])

# bLine = np.linspace(-0.7,-0.3,4000)
# aLine = np.zeros(4000)
# qualityStable = np.zeros(4000)
# qualityUnstable = np.zeros(4000)
# tangentPointsX = np.zeros(4000)
# tangentPointsY = np.zeros(4000)
l = 0
j = 0
k = 0
i = 0
s = 0
t = 0
##########Our old values##############
# with open(filenameTanX, 'r') as filehandle: #4000
#     for line in filehandle:
#         tangentPointsX[l] = float(line)
#         l=l+1
#
# with open(filenameTanY, 'r') as filehandle: #4000
#     for line in filehandle:
#         tangentPointsY[j] = float(line)
#         j=j+1
# # print(tangentPointsX)
# # print(tangentPointsY)
# with open(filename, 'r') as filehandle: #4000
#     # for line in filehandle:
#     #     dummy = tanpoint(aLine[j],float(line),10)
#     #     tangentPointsX[j] = dummy[0]
#     #     tangentPointsY[j] = dummy[1]
#     #     j=j+1
#     #     print(float(line))
#     for line in filehandle:
#         aLine[i]=float(line)
#         qualityStable[i] = getQualitySt(float(bLine[i]),float(aLine[i]),tangentPointsX[i],tangentPointsY[i])
#         qualityUnstable[i] = getQualityUnst(float(bLine[i]),float(aLine[i]),tangentPointsX[i],tangentPointsY[i])
#         i=i+1
##########New values############ 7/4/21
# fileIndepA = 'aValues7421.txt'
# fileIndepB = 'bValues7421.txt'
# fileIndepX = 'xValues7421.txt'
# fileIndepY = 'yValues7421.txt'
# tangentPointsXIndep = np.zeros(40)
# tangentPointsYIndep = np.zeros(40)
# bLineIndep = np.linspace(-0.5,-0.3,40)
# aLineIndep = np.zeros(40)
# qualityStableIndep = np.zeros(40)
# qualityUnstableIndep = np.zeros(40)
# with open(fileIndepX, 'r') as filehandle: #4000
#     for line in filehandle:
#         tangentPointsXIndep[l] = float(line)
#         l=l+1
#
# with open(fileIndepY, 'r') as filehandle: #4000
#     for line in filehandle:
#         tangentPointsYIndep[j] = float(line)
#         j=j+1
# # print(tangentPointsX)
# # print(tangentPointsY)
# with open(fileIndepA, 'r') as filehandle: #4000
#     # for line in filehandle:
#     #     dummy = tanpoint(aLine[j],float(line),10)
#     #     tangentPointsX[j] = dummy[0]
#     #     tangentPointsY[j] = dummy[1]
#     #     j=j+1
#     #     print(float(line))
#     for line in filehandle:
#         aLineIndep[i]=float(line)
#         qualityStableIndep[i] = getQualitySt(float(bLineIndep[i]),float(aLineIndep[i]),tangentPointsXIndep[i],tangentPointsYIndep[i])
#         qualityUnstableIndep[i] = getQualityUnst(float(bLineIndep[i]),float(aLineIndep[i]),tangentPointsXIndep[i],tangentPointsYIndep[i])
#         i=i+1
#####100 points Ruipeng#######
fileRuiX = 'xcoorstalbe 100.txt'
fileRuiY = 'ycoorstable 100.txt'
fileRuiA = 'a values 100 points.txt'
fileRuiB = 'b values 100 points.txt'
xRui = np.zeros(100)
yRui = np.zeros(100)
bRui = np.zeros(100)
aRui = np.zeros(100)
tangentPointsXRui = np.zeros(100)
tangentPointsYRui = np.zeros(100)
qualRuiSt = np.zeros(100)
qualRuiUst = np.zeros(100)

with open(fileRuiA, 'r') as filehandle: #4000
    for line in filehandle:
        aRui[l] = float(line)
        l=l+1

with open(fileRuiB, 'r') as filehandle: #4000
    for line in filehandle:
        bRui[j] = float(line)
        j=j+1
with open(fileRuiX, 'r') as filehandle: #4000
    for line in filehandle:
        xRui[t] = float(line)
        t=t+1
with open(fileRuiY, 'r') as filehandle: #4000
    for line in filehandle:
        yRui[s] = float(line)
        s=s+1
for r in range(100):
    qualRuiSt[r] = getQualitySt(bRui[r],aRui[r],xRui[r],yRui[r])
    qualRuiUst[r] = getQualityUnst(bRui[r],aRui[r],xRui[r],yRui[r])
plt.title('Ruipeng 100 points')
plt.plot(bRui, qualRuiUst, '-',color='red')
plt.plot(bRui, qualRuiSt,'-',color='blue')
plt.legend(['Unstable','Stable'])
# print('b=',float(bLine[200]))
# print('a=',float(aLine[200]))
# print('tangentx=', tangentPointsX[200])
# print('tangenty=', tangentPointsY[200])
# print(getQualitySt(float(bLine[200]),float(aLine[200]),tangentPointsX[200],tangentPointsY[200]))

# print(aLine)
# print(bLine)
# print(qualityStable)
# print(qualityUnstable)
# fig = plt.figure(figsize = (11,11))
# ax = fig.add_subplot(1, 1, 1)
# plt.xlabel('b')
# plt.ylabel('Quality')

# plt.plot(blinePatryk,alinePatryk, '-', color = 'green')
# plt.plot(bLineIndep, qualityUnstableIndep, '-',color='red')
# plt.plot(bLineIndep, qualityStableIndep,'-',color='blue')
# plt.legend(['Unstable','Stable'])

# print(getQualitySt(-.45,0.954639,1.6270259396510076,-0.021103424887675432))
# print(getQualityUnst(-.45,0.954639,1.6270259396510076,-0.021103424887675432))
plt.show()