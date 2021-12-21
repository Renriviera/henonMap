# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 17:55:48 2021

@author: rzou
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from mpmath import *
import scipy.linalg as la

mp.dps = 20
M=15

# %% part for all def

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


def getPoints(a, b, lCut, rCut, p, number, dx, dy, iterations, mapVersion, precision):
    if (precision == 'arbitrary'):
        if (mapVersion == 'forwards'):

            xPointsLine = mp.linspace(p + (lCut * dx), p + (dx * rCut), number)
            yPointsLine = mp.linspace(p + (lCut * dy), p + (dy * rCut), number)
            xi = xPointsLine
            yi = yPointsLine
            for i in range(iterations):
                xOld = xi
                yOld = yi
                yNew = xOld
                xNew = [hMap(a, b, xOld[j], yOld[j]) for j in range(number)]
                xi = xNew
                yi = yNew
            floatYNew = [float(yNew[k]) for k in range(number)]  ##ends use of multiple precision
            floatXNew = [float(xNew[k]) for k in range(number)]
        elif (mapVersion == 'backwards'):

            xPointsLine = mp.linspace(p + (lCut * dx), p + (dx * rCut), number)
            yPointsLine = mp.linspace(p + (lCut * dy), p + (dy * rCut), number)
            xi_b1 = xPointsLine
            yi_b1 = yPointsLine
            for i in range(iterations):
                xOld = xi_b1
                yOld = yi_b1
                yNew = [backwards_hMap(a, b, xOld[j], yOld[j]) for j in range(number)]
                xNew = yOld
                xi_b1 = xNew
                yi_b1 = yNew
            floatYNew = [float(yNew[k]) for k in range(number)]  ##ends use of multiple precision
            floatXNew = [float(xNew[k]) for k in range(number)]
    elif (precision == 'standard'):
        if (mapVersion == 'forwards'):

            xPointsLine = np.linspace(p + (lCut * dx), p + (dx * rCut), number)
            yPointsLine = np.linspace(p + (lCut * dy), p + (dy * rCut), number)
            xi1 = xPointsLine
            yi1 = yPointsLine
            for i in range(iterations):
                xOld = xi1
                yOld = yi1
                xNew = hMap(a, b, xOld, yOld)
                yNew = xOld
                xi1 = xNew
                yi1 = yNew
            floatYNew = yNew
            floatXNew = xNew
        elif (mapVersion == 'backwards'):

            xline = np.linspace(p + (lCut * dx), p + (dx * rCut), number)
            yline = np.linspace(p + (lCut * dy), p + (dy * rCut), number)
            xi_b = xline
            yi_b = yline
            for i in range(iterations):
                xOld_b = xi_b
                yOld_b = yi_b
                yNew_b = backwards_hMap(a, b, xOld_b, yOld_b)
                xNew_b = yOld_b
                xi_b = xNew_b
                yi_b = yNew_b
            floatYNew = yNew_b
            floatXNew = xNew_b
    return floatXNew, floatYNew


def getindexstable(xcoor, ycoor, b, density):
    ais = set()
    for i in range(density):
        if xcoor[i] >= 1:
            ais.add(i)
    bis = set()
    for i in range(density):
        # if ycoor[i] > -0.5+(b+0.3) and ycoor[i] < 0.5-(b+0.3):
        if ycoor[i] > -0.2 and ycoor[i] < 0.1:
            bis.add(i)
    iss = ais.intersection(bis)
    left_ind = min(iss)
    right_ind = max(iss)

    return left_ind, right_ind


def getindexunstable(xcoor, ycoor, b, density):
    xline = list(xcoor)
    maxind = xline.index(max(xline))
    sis = set()
    for i in range(density):
        # if ycoor[i] > -0.3+(b+0.3) and ycoor[i] < 0.7:
        if ycoor[i] > -0.2 and ycoor[i] < 0.2:  # We look more closely to where tangence happens to make it more precise
            sis.add(i)
    tis = set()
    for i in range(density):
        if xcoor[i] > 1:
            tis.add(i)
    iss = sis.intersection(tis)
    leftind = min(iss)
    return leftind, maxind


def getstablecut(b, N):
    # left_cut_b = 0
    # if N == 15:
    #     if -0.35 < b <= -0.25:
    #         right_cut_b = (-b) * 10 ** -(N - 3) * 0.5
    #     if -0.4 < b <= -0.35:
    #         right_cut_b = (-b) * 10 ** -(N - 4)
    #     if -0.5 < b <= -0.4:
    #         right_cut_b = (-b) * 10 ** -(N - 5)
    #     if -0.6 < b <= -0.5:
    #         right_cut_b = (-b) * 10 ** -(N - 6)
    #     if b <= -0.6:
    #         right_cut_b = (-b) * 10 ** -(N - 7) * 2
    # elif N == 20:
    if N == 15:
        if -0.3 < b <= -0.25:
            right_cut_b = (-b) * 10 ** -(N - 2)
        if -0.35 < b <= -0.3:
            right_cut_b = (-b * 0.2) * 10 ** -(N - 3)
        if -0.4 < b <= -0.35:
            right_cut_b = (-b * 0.1) * 10 ** -(N - 4)
        if -0.46 <= b <= -0.4:
            right_cut_b = (-b * 0.1) * 10 ** -(N - 5)
        if -0.5 < b < -0.46:
            right_cut_b = (-b * 0.4) * 10 ** -(N - 5)
        if -0.55 < b <= -0.5:
            right_cut_b = (-b * 0.2) * 10 ** -(N - 6)
        if -0.6 < b <= -0.55:
            right_cut_b = (-b * 0.7) * 10 ** -(N - 6)
        if -0.65 < b <= -0.6:
            right_cut_b = (-b * 0.3) * 10 ** -(N - 7)
        if -0.7 < b <= -0.65:
            right_cut_b = (-b * 0.2) * 10 ** -(N - 8)
        if b <= -0.7:
            right_cut_b = (-b) * 10 ** -(N - 8)
        left_cut_b = 0

    intlen = right_cut_b - left_cut_b
    return left_cut_b, right_cut_b, intlen


def getstablecuttan(xcoor, ycoor, b, density, N, precision):
    if (precision == 'arbitrary'):
        lcut = mpf(
            getstablecut(b, N)[0] + getstablecut(b, N)[2] * getindexstable(xcoor, ycoor, b, density)[0] / density)
        rcut = mpf(
            getstablecut(b, N)[0] + getstablecut(b, N)[2] * getindexstable(xcoor, ycoor, b, density)[1] / density)
    elif (precision == 'standard'):
        lcut = getstablecut(b, N)[0] + getstablecut(b, N)[2] * getindexstable(xcoor, ycoor, b, density)[0] / density
        rcut = getstablecut(b, N)[0] + getstablecut(b, N)[2] * getindexstable(xcoor, ycoor, b, density)[1] / density
    return lcut, rcut


def getunstablecut(b, N):
    left_cut = 0
    if N == 15:
        right_cut = (5 * 10 ** -(N - 8))
    elif N == 20:
        right_cut = (5 * 10 ** -(N - 11))
    intelen = right_cut - left_cut
    return left_cut, right_cut, intelen


def getunstablecuttan(xcoor, ycoor, b, density, N):
    lcut = getunstablecut(b, N)[0] + getunstablecut(b, N)[2] * getindexunstable(xcoor, ycoor, b, density)[0] / density
    rcut = getunstablecut(b, N)[0] + getunstablecut(b, N)[2] * 1.2 * (
                2 * getindexunstable(xcoor, ycoor, b, density)[1] - getindexunstable(xcoor, ycoor, b, density)[
            0]) / density
    return lcut, rcut

def tangency(a, b, N):
    density = 5000
    density1 = 10
    rho = 0.2
    p1, p2 = getPeriodic(a, b)
    p = p2
    evecUn = getUnstableLin(a, b, p)
    dx = evecUn[0] * rho
    dy = evecUn[1] * rho
    evecSt = getStableLin(a, b, p)  # eigenvector for stable manifold
    dx_b = evecSt[0] * rho
    dy_b = evecSt[1] * rho
    xNew_b1 = \
    getPoints(a, b, getunstablecut(b, N)[0], getunstablecut(b, N)[1], p, density, dx, dy, N, 'forwards', 'standard')[0]
    yNew_b1 = \
    getPoints(a, b, getunstablecut(b, N)[0], getunstablecut(b, N)[1], p, density, dx, dy, N, 'forwards', 'standard')[1]
    xNew1 = \
    getPoints(a, b, getstablecut(b, N)[0], getstablecut(b, N)[1], p, density, dx_b, dy_b, N, 'backwards', 'arbitrary')[
        0]
    yNew1 = \
    getPoints(a, b, getstablecut(b, N)[0], getstablecut(b, N)[1], p, density, dx_b, dy_b, N, 'backwards', 'arbitrary')[
        1]
    xstable = getPoints(a, b, getstablecuttan(xNew1, yNew1, b, density, N, 'arbitrary')[0],
                        getstablecuttan(xNew1, yNew1, b, density, N, 'arbitrary')[1], p, density1, dx_b, dy_b, N,
                        'backwards', 'arbitrary')[0]
    ystable = getPoints(a, b, getstablecuttan(xNew1, yNew1, b, density, N, 'arbitrary')[0],
                        getstablecuttan(xNew1, yNew1, b, density, N, 'arbitrary')[1], p, density1, dx_b, dy_b, N,
                        'backwards', 'arbitrary')[1]
    xunstable = getPoints(a, b, getunstablecuttan(xNew_b1, yNew_b1, b, density, N)[0],
                          getunstablecuttan(xNew_b1, yNew_b1, b, density, N)[1], p, density1, dx, dy, N, 'forwards',
                          'standard')[0]
    yunstable = getPoints(a, b, getunstablecuttan(xNew_b1, yNew_b1, b, density, N)[0],
                          getunstablecuttan(xNew_b1, yNew_b1, b, density, N)[1], p, density1, dx, dy, N, 'forwards',
                          'standard')[1]
    model = np.poly1d(np.polyfit(xstable, ystable, 1))
    lmodel = list(model.c)
    coef = ["a", "b"]
    result = dict(zip(coef, lmodel))
    ###
    model1 = np.poly1d(np.polyfit(yunstable, xunstable, 2))
    lmodel1 = list(model1.c)
    print('HERE',lmodel1)
    coef1 = ["c", "d", "e"]
    result1 = dict(zip(coef1, lmodel1))
    y1 = ((1 / result["a"]) - result1["d"]) / (2 * result1["c"])
    x1 = result1["c"] * y1 ** 2 + result1["d"] * y1 + result1["e"]
    x2 = (1 / result["a"]) * y1 - result["b"] / result["a"]
    return ((x1 + x2) / 2, y1)

def distance(a, b, N):
    density = 5000
    density1 = 10
    rho = 0.2
    p1, p2 = getPeriodic(a, b)
    p = p2
    evecUn = getUnstableLin(a, b, p)
    dx = evecUn[0] * rho
    dy = evecUn[1] * rho
    evecSt = getStableLin(a, b, p)  # eigenvector for stable manifold
    dx_b = evecSt[0] * rho
    dy_b = evecSt[1] * rho
    xNew_b1 = \
    getPoints(a, b, getunstablecut(b, N)[0], getunstablecut(b, N)[1], p, density, dx, dy, N, 'forwards', 'standard')[0]
    yNew_b1 = \
    getPoints(a, b, getunstablecut(b, N)[0], getunstablecut(b, N)[1], p, density, dx, dy, N, 'forwards', 'standard')[1]
    xNew1 = \
    getPoints(a, b, getstablecut(b, N)[0], getstablecut(b, N)[1], p, density, dx_b, dy_b, N, 'backwards', 'arbitrary')[
        0]
    yNew1 = \
    getPoints(a, b, getstablecut(b, N)[0], getstablecut(b, N)[1], p, density, dx_b, dy_b, N, 'backwards', 'arbitrary')[
        1]
    xstable = getPoints(a, b, getstablecuttan(xNew1, yNew1, b, density, N, 'arbitrary')[0],
                        getstablecuttan(xNew1, yNew1, b, density, N, 'arbitrary')[1], p, density1, dx_b, dy_b, N,
                        'backwards', 'arbitrary')[0]
    ystable = getPoints(a, b, getstablecuttan(xNew1, yNew1, b, density, N, 'arbitrary')[0],
                        getstablecuttan(xNew1, yNew1, b, density, N, 'arbitrary')[1], p, density1, dx_b, dy_b, N,
                        'backwards', 'arbitrary')[1]
    xunstable = getPoints(a, b, getunstablecuttan(xNew_b1, yNew_b1, b, density, N)[0],
                          getunstablecuttan(xNew_b1, yNew_b1, b, density, N)[1], p, density1, dx, dy, N, 'forwards',
                          'standard')[0]
    yunstable = getPoints(a, b, getunstablecuttan(xNew_b1, yNew_b1, b, density, N)[0],
                          getunstablecuttan(xNew_b1, yNew_b1, b, density, N)[1], p, density1, dx, dy, N, 'forwards',
                          'standard')[1]
    model = np.poly1d(np.polyfit(xstable, ystable, 1))
    lmodel = list(model.c)
    coef = ["a", "b"]
    result = dict(zip(coef, lmodel))
    ###
    model1 = np.poly1d(np.polyfit(yunstable, xunstable, 2))
    lmodel1 = list(model1.c)
    coef1 = ["c", "d", "e"]
    result1 = dict(zip(coef1, lmodel1))
    y1 = ((1 / result["a"]) - result1["d"]) / (2 * result1["c"])
    x1 = result1["c"] * y1 ** 2 + result1["d"] * y1 + result1["e"]
    x2 = (1 / result["a"]) * y1 - result["b"] / result["a"]
    return (x1 - x2)

def BisectionMethod(distance, b, start,end,tolerance,maxIter, regulaFalsi):
    K=1
    while K <= maxIter:
        if K == 1:
            fstart = distance(start,b,M)
            fend = distance(end,b,M)
        if(regulaFalsi):
            midpoint = (start*fend-end*fstart)/(fend-fstart)
        else:
            midpoint = (start+end)/2
        fmidpoint = distance(midpoint,b,M)
        print('midpoint:',midpoint)
        print('fmidpoint:',fmidpoint)
        if((start+end)/2 < tolerance or abs(fmidpoint) < tolerance): #or d(midpoint,b,M) == 0
            return midpoint,K
        else:
            if(np.sign(fmidpoint) != np.sign(fstart)):
                end = midpoint
                fend = fmidpoint
            elif(np.sign(fmidpoint) != np.sign(fend)):
                start = midpoint
                fstart = fmidpoint
            else:
                print(K, b)
                break
        K=K+1
    return midpoint,K

def save_to_file(arr,filename):
    arr = [str(i) for i in arr]
    with open(filename,'w') as f:
            for i in arr:
                f.write(str(i)+"\n")


# save



def read_to_arr(filename):

    with open(filename,'r') as f:
        lines = f.readlines()
    arr = [float(line.replace("\n","")) for line in lines]
    return arr

# read
# new_arr = read_to_arr("file.txt")

# print(new_arr)
# %%
b = -0.3
a = 2.4385 * b + 2.052  ### cuts work for change of a under 0.05, work for change of b under 0.005
density = 20000
density1 = 10
rho = 0.2
p1, p2 = getPeriodic(a, b)
p = p2
N = 15
evecUn = getUnstableLin(a, b, p)
dx = evecUn[0] * rho
dy = evecUn[1] * rho
evecSt = getStableLin(a, b, p)  # eigenvector for stable manifold
dx_b = evecSt[0] * rho
dy_b = evecSt[1] * rho
xNew_b1 = \
getPoints(a, b, getunstablecut(b, N)[0], getunstablecut(b, N)[1], p, density, dx, dy, N, 'forwards', 'standard')[0]
yNew_b1 = \
getPoints(a, b, getunstablecut(b, N)[0], getunstablecut(b, N)[1], p, density, dx, dy, N, 'forwards', 'standard')[1]
xNew1 = \
getPoints(a, b, getstablecut(b, N)[0], getstablecut(b, N)[1], p, density, dx_b, dy_b, N, 'backwards', 'arbitrary')[0]
yNew1 = \
getPoints(a, b, getstablecut(b, N)[0], getstablecut(b, N)[1], p, density, dx_b, dy_b, N, 'backwards', 'arbitrary')[1]
xNew2 = getPoints(a, b, getstablecuttan(xNew1, yNew1, b, density, N, 'arbitrary')[0],
                  getstablecuttan(xNew1, yNew1, b, density, N, 'arbitrary')[1], p, density1, dx_b, dy_b, N, 'backwards',
                  'arbitrary')[0]
yNew2 = getPoints(a, b, getstablecuttan(xNew1, yNew1, b, density, N, 'arbitrary')[0],
                  getstablecuttan(xNew1, yNew1, b, density, N, 'arbitrary')[1], p, density1, dx_b, dy_b, N, 'backwards',
                  'arbitrary')[1]
xNew_b2 = getPoints(a, b, getunstablecuttan(xNew_b1, yNew_b1, b, density, N)[0],
                    getunstablecuttan(xNew_b1, yNew_b1, b, density, N)[1], p, density1, dx, dy, N, 'forwards',
                    'standard')[0]
yNew_b2 = getPoints(a, b, getunstablecuttan(xNew_b1, yNew_b1, b, density, N)[0],
                    getunstablecuttan(xNew_b1, yNew_b1, b, density, N)[1], p, density1, dx, dy, N, 'forwards',
                    'standard')[1]

# %%
b= -0.45
a = 2.43858*b+2.052
print('A', a)
print(distance(a, b, 15))

# %%
# Set up plot
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(1, 1, 1)
# plt.title('Approximation of Hénon Orbit, # Iterations = ' + str(N) + ', parameters a= ' + str(a) + ', b = ' + str(b))
# lim = 2
# xmid = 1.1
# ymid = 0
# plt.xlim([xmid - lim, xmid + lim])
# plt.ylim([ymid - lim, ymid + lim])
# ax.plot(xNew1, yNew1, ',', color='blue', alpha=0.9, markersize=1)
# ax.plot(xNew_b1, yNew_b1, ',', color='red', alpha=0.9, markersize=1)
# ax.plot(xNew2, yNew2, 'x', color='blue', alpha=0.9, markersize=5)
# ax.plot(xNew_b2, yNew_b2, 'x', color='red', alpha=0.9, markersize=5)
# %% this works for density = 5000
if N == 15:
    if -0.3 < b <= -0.25:
        right_cut_b = (-b) * 10 ** -(N - 2)
    if -0.35 < b <= -0.3:
        right_cut_b = (-b * 0.2) * 10 ** -(N - 3)
    if -0.4 < b <= -0.35:
        right_cut_b = (-b * 0.1) * 10 ** -(N - 4)
    if -0.46 <= b <= -0.4:
        right_cut_b = (-b * 0.1) * 10 ** -(N - 5)
    if -0.5 < b < -0.46:
        right_cut_b = (-b * 0.4) * 10 ** -(N - 5)
    if -0.55 < b <= -0.5:
        right_cut_b = (-b * 0.2) * 10 ** -(N - 6)
    if -0.6 < b <= -0.55:
        right_cut_b = (-b * 0.7) * 10 ** -(N - 6)
    if -0.65 < b <= -0.6:
        right_cut_b = (-b * 0.3) * 10 ** -(N - 7)
    if -0.7 < b <= -0.65:
        right_cut_b = (-b * 0.2) * 10 ** -(N - 8)
    if b <= -0.7:
        right_cut_b = (-b) * 10 ** -(N - 8)
    left_cut_b = 0
# C = np.zeros(3)
# B = np.linspace(-0.7, -0.65, 3)
# A = np.zeros(3)
# for i in range(3):
#     print('NEW POINT',i)
#     if i == 0:
#         a = 0.34499400000000024
#         A[0], C[0] = BisectionMethod(distance, -0.7, a, a+0.05, 0.00005,50,True)
#
#     if i > 0:
#         A[i], C[i] = BisectionMethod(distance, B[i], A[i - 1], A[i - 1]+0.05, 0.00005,50,True)
#     # A = [NewtonsMethod(d, B[i], A[i - 1], 10 ** -3) for i in range(1000)]
# print(B)
# print(A)
# save_to_file(B,"../qualityHenonParameters/bValues7421.txt")
# save_to_file(A,"../qualityHenonParameters/aValues7421.txt")
# X = []
# Y = []
# for i in range(len(B)):
#     xcord, ycord = tangency(A[i],B[i],15)
#     X.append(xcord)
#     Y.append(ycord)
# save_to_file(X,"../qualityHenonParameters/xValues7421.txt")
# save_to_file(Y,"../qualityHenonParameters/yValues7421.txt")
# print(C)
print('tangency')
print(tangency(0.95,-0.45,15))
plt.title('Regula falsi 50 points small interval')
# plt.plot(B, A)
plt.show()
