# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 14:58:14 2021

@author: rzou
"""

import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
import scipy.linalg as la


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


def d(a, b, N):
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
    return (x1 - x2)


# print(d(1.1,-.45,10))
# print(d(1.3125,-0.3,13))
delta = 0.0001
M = 10


# a = 1.31
# b= -0.3
############################ derivative of d ###############
def derd(a, b):
    return ((d(a + delta, b, M)) - (d(a - delta, b, M))) / (2 * delta)


# print(derd(1.31,-0.3))
# b = -0.35
# a = 1.17
# a1 = a - d(a,b,M) / derd(a,b)
# print("a1 is",a1)

def NewtonsMethod(d, b, a, tolerance):
    # i=0
    counter = 0
    while True:
        counter = counter + 1
        # =i+1
        a1 = a - d(a, b, M) / derd(a, b)
        dis = abs(d(a1, b, M))
        if dis < tolerance:
            # print(i)
            break

        a = a1
    return a, counter


# print(NewtonsMethod(d, -0.3, 0.0001))
# print("By Newton's method, we get for b = -0.3, the approximate a is",NewtonsMethod(d, a, 10**-5),M)
###### negative distance means no intersection ##########
# print("And the distance(error) is",d(NewtonsMethod(d, a, 10**-5),M))

# (b)=a
# dens = 5000
# dens1 = 10000
b1 = -0.6
b2 = -0.3

b4 = -0.7
b5 = -0.2
# C = np.zeros(dens)
# B = np.linspace(b1,b2,dens)  ############# 0.3-0.6
# B1 = np.linspace(b3,b2,dens) ############## 0.4-0.6
# B2 = np.linspace(b4, b5, dens1)  ###### 0.2-0.7
# C2 = np.zeros(dens1)  ### 0.7-0.2
# A2 = np.zeros(dens1)  ### 0.7-0.2
# A= np.zeros(dens)   ######## -0.4
# A1 = np.zeros(dens)   ######### -0.6
# for i in range(dens1):
#     # if i == 0:
#     a = 0.339
#     A2[0], C2[0] = NewtonsMethod(d, -0.7, a, 0.00001)
#     # if i == 1:
#
#     A2[1], C2[1] = NewtonsMethod(d, B2[1], a, 0.00001)
#     # if i == 1:
#     # a= 0.95
#     # A[1],C[1]=NewtonsMethod(d, -0.45+1/dens,a, 0.0001)
#     if i > 1:
#         slope = (A2[i - 1] - A2[i - 2]) / ((b5 - b4) / dens1)
#         A2[i], C2[i] = NewtonsMethod(d, B2[i], slope * B2[i] + (A2[i - 1] - slope * B2[i - 1]), 0.00001)
# # A = [NewtonsMethod(d,B[i],A[i-1],10**-3) for i in range(dens)]
#
# print(B2)
# print(A2)
# print(C2)
# print(B)
# print(A1)
# plt.plot(B,A1)


# print(A)
# %%
dens2 = 1000
C = np.zeros(dens2)
A = np.zeros(dens2)
B = np.linspace(b4, b1, dens2)
# for i in range(dens2):
#     # if i == 0:
#     a = 0.339
#     A[0], C[0] = NewtonsMethod(d, -0.7, a, 0.0001)
#     # if i == 1:
#
#     A[1], C[1] = NewtonsMethod(d, B[1], a, 0.0001)
#     # if i == 1:
#     # a= 0.95
#     # A[1],C[1]=NewtonsMethod(d, -0.45+1/dens,a, 0.0001)
#     if i > 1:
#         slope = (A[i - 1] - A[i - 2]) / ((b1 - b4) / dens2)
#         A[i], C[i] = NewtonsMethod(d, B[i], slope * B[i] + (A[i - 1] - slope * B[i - 1]), 0.0001)

print(B)
print(A)
print(C)

# %%
plt.plot(B, A)
plt.plot(B, C)
plt.show()


# %% store section
# def save_to_file(arr, filename):
#     arr = [str(i) for i in arr]
#     with open(filename, 'w') as f:
#         for i in arr:
#             f.write(str(i) + "\n")
#
#
# # save
# save_to_file(A, "a values for b 0.7-0.6,dens 1000.txt")

# # -*- coding: utf-8 -*-
# """
# Created on Sun Jun 27 14:58:14 2021
#
# @author: rzou
# """
#
# import numpy as np
# import matplotlib.pyplot as plt
# import mpmath as mp
# import scipy.linalg as la
#
# mp.dps = 30
#
# def hMap(a, b, x, y):
#     return 1 - a * x ** 2 + b * y
#
#
# # −b−1(1 − ayn2 − bxn)
# def backwards_hMap(a, b, x, y):
#     return -(1 / b) * (1 - a * y ** 2 - x)
#
#
# def getPeriodic(a, b):
#     p1 = (-1 * (1 - b) + np.sqrt((1 - b) ** 2 + 4 * a)) / (2 * a)
#     p2 = (-1 * (1 - b) - np.sqrt((1 - b) ** 2 + 4 * a)) / (2 * a)
#     return p1, p2
#
#
# def getUnstableLin(a, b, x):
#     evecUn = [-a * x + np.sqrt(b + a ** 2 * x ** 2), 1]
#     return evecUn
#
#
# def getStableLin(a, b, x):
#     evecSt = [-a * x - np.sqrt(b + a ** 2 * x ** 2), 1]
#     return evecSt
#
#
# def eigvalue(a, b, x):
#     A = np.array([[-2 * a * x, b], [1, 0]])
#     results = la.eig(A)
#     return results[0]
#
def getPoints(a,b,lCut, rCut, p, number, dx, dy, iterations, mapVersion, precision):
    if(precision == 'arbitrary'):
        if (mapVersion == 'forwards'):

            xPointsLine = mp.linspace(p + (lCut * dx), p + (dx * rCut), number)
            yPointsLine = mp.linspace(p + (lCut * dy), p + (dy * rCut), number)
            xOld = xPointsLine
            yOld = yPointsLine
            for i in range(iterations):
                yNew = [hMap(a, b, xOld[j], yOld[j]) for j in range(number)]
                xNew = yOld
                xOld = xNew
                yOld = yNew
            floatYNew = [float(yNew[k]) for k in range(number)] ##ends use of multiple precision
            floatXnew = [float(xNew[k]) for k in range(number)]
        elif (mapVersion == 'backwards'):


            xPointsLine = mp.linspace(p + (lCut * dx), p + (dx * rCut), number)
            yPointsLine = mp.linspace(p + (lCut * dy), p + (dy * rCut), number)
            xOld = xPointsLine
            yOld = yPointsLine
            for i in range(iterations):
                yNew = [backwards_hMap(a, b, xOld[j], yOld[j]) for j in range(number)]
                xNew = yOld
                xOld = xNew
                yOld = yNew
            floatYNew = [float(yNew[k]) for k in range(number)] ##ends use of multiple precision
            floatXnew = [float(xNew[k]) for k in range(number)]
    elif(precision == 'standard'):
        if (mapVersion == 'forwards'):

            xPointsLine = np.linspace(p + (lCut * dx), p + (dx * rCut), number)
            yPointsLine = np.linspace(p + (lCut * dy), p + (dy * rCut), number)
            xOld = xPointsLine
            yOld = yPointsLine
            for i in range(iterations):
                yNew = [hMap(a, b, xOld[j], yOld[j]) for j in range(number)]
                xNew = yOld
                xOld = xNew
                yOld = yNew
            floatYNew = yNew
            floatXnew = xNew
        elif (mapVersion == 'backwards'):

            xPointsLine = np.linspace(p + (lCut * dx), p + (dx * rCut), number)
            yPointsLine = np.linspace(p + (lCut * dy), p + (dy * rCut), number)
            xOld = xPointsLine
            yOld = yPointsLine
            for i in range(iterations):
                yNew = [backwards_hMap(a, b, xOld[j], yOld[j]) for j in range(number)]
                xNew = yOld
                xOld = xNew
                yOld = yNew
            floatYNew = yNew
            floatXnew = xNew
    return floatYNew, floatXnew
#
# def d(a, b, N):
#     density = 10000
#     density1 = 5
#     rho = 0.2
#     p1, p2 = getPeriodic(a, b)
#     p = p2
#     ############ for stable  #############
#     # eigstable = eigvalue(a,b,p)[0].real
#     left_cut_b = 0
#     if b > -0.32:
#         right_cut_b = 5 * 10 ** -(N - 2)
#     if -0.45 < b <= -0.32:
#         right_cut_b = 5 * 10 ** -(N - 3)
#     if -0.65 < b <= -0.45:
#         right_cut_b = 9 * 10 ** -(N - 4)
#     if b <= -0.65:
#         right_cut_b = 1 * 10 ** -(N - 6)
#     evecSt = getStableLin(a, b, p)  # eigenvector for stable manifold
#
#     dx_b = evecSt[0] * rho
#     dy_b = evecSt[1] * rho
#
#     # testxline = np.linspace(p + (left_cut_b * dx_b), p + (dx_b * right_cut_b), density)
#     # testyline = np.linspace(p + (left_cut_b * dy_b), p + (dy_b * right_cut_b), density)
#     # xi_b = testxline
#     # yi_b = testyline
#     #
#     # if (True):  # to conveniently hide backward if needed
#     #     for i in range(N):
#     #         xOld_b = xi_b
#     #         yOld_b = yi_b
#     #         yNew_b = backwards_hMap(a, b, xOld_b, yOld_b)
#     #         xNew_b = yOld_b
#     #         xi_b = xNew_b
#     #         yi_b = yNew_b
#     # lyNew_b = list(yNew_b)
#     # lxNew_b = list(xNew_b)
#
#     lyNew_b, lxNew_b = getPoints(a,b,left_cut_b,right_cut_b,p,density,dx_b,dy_b, N,'backwards', 'standard')
#     ais = set()
#     for i in range(density):
#         if lxNew_b[i] > 1:
#             ais.add(i)
#     bis = set()
#     for i in range(density):
#         if lyNew_b[i] > -0.5 + (b + 0.3) and lyNew_b[i] < 0.5 - (b + 0.3):
#             bis.add(i)
#
#     iss = ais.intersection(bis)
#     print(ais)
#     print(bis)
#     left_ind = min(iss)
#
#     right_ind = max(iss)
#
#     intelen = right_cut_b - left_cut_b
#
#     left_cut_b1 = left_cut_b + intelen * left_ind / density
#     right_cut_b1 = left_cut_b + intelen * right_ind / density
#
#     # xi_b1 = np.linspace(p + (left_cut_b1 * dx_b), p + (dx_b * right_cut_b1), density1)
#     # yi_b1 = np.linspace(p + (left_cut_b1 * dy_b), p + (dy_b * right_cut_b1), density1)
#     #
#     # xNew_b1, yNew_b1 = xi_b1, yi_b1
#
#     # if (True):  # to conveniently hide backward if needed
#     #     for i in range(N):
#     #         xOld_b1 = xi_b1
#     #         yOld_b1 = yi_b1
#     #         yNew_b1 = backwards_hMap(a, b, xOld_b1, yOld_b1)
#     #         xNew_b1 = yOld_b1
#     #         xi_b1 = xNew_b1
#     #         yi_b1 = yNew_b1
#     #
#     # lxNew_b1 = list(xNew_b1)
#     # lyNew_b1 = list(yNew_b1)
#     lyNew_b1, lxNew_b1 = getPoints(a,b,left_cut_b1,right_cut_b1,p,density1,dx_b,dy_b, N, 'backwards', 'standard')
#
#     model1 = np.poly1d(np.polyfit(lxNew_b1, lyNew_b1, 1))
#     lmodel1 = list(model1.c)
#     coef1 = ["al", "bl"]
#     result1 = dict(zip(coef1, lmodel1))
#     ############################# for unstable ##################################
#     # eigunstable = eigvalue(a,b,p)[1].real
#     left_cut = 0
#     right_cut = 1 * 10 ** -(N - 8)  # 0.7 too large 0.07 too small
#     evecUn = getUnstableLin(a, b, p)
#     dx = evecUn[0] * rho
#     dy = evecUn[1] * rho
#     # testxline = np.linspace(p + (left_cut * dx), p + (dx * right_cut), density)
#     # testyline = np.linspace(p + (left_cut * dy), p + (dy * right_cut), density)
#     # xi = testxline
#     # yi = testyline
#     # # xNew, yNew = xi, yi
#     # for i in range(N):
#     #     xOld = xi
#     #     yOld = yi
#     #     xNew = hMap(a, b, xOld, yOld)
#     #     yNew = xOld
#     #     xi = xNew
#     #     yi = yNew
#     # lxNew = list(xNew)
#     # lyNew = list(yNew)
#     lyNew, lxNew = getPoints(a,b,left_cut,right_cut,p,density,dx,dy, N, 'forwards', 'standard')
#
#     inx = lxNew.index(max(lxNew))
#     intelen = right_cut - left_cut
#     sis = set()
#     tis = set()
#     for i in range(density):
#         if lyNew[i] > -0.5 + (b + 0.3) and lyNew[i] < 0.5 - (b + 0.3):
#             sis.add(i)
#
#     for i in range(density):
#         if lxNew[i] > 1:
#             tis.add(i)
#     iss = sis.intersection(tis)
#     leftind = min(iss)
#
#     left_cut1 = left_cut + intelen * leftind / density
#     right_cut1 = left_cut + intelen * (2 * inx - leftind) / density
#
#     # xline = np.linspace(p + (left_cut1 * dx), p + (dx * right_cut1), density1)
#     # yline = np.linspace(p + (left_cut1 * dy), p + (dy * right_cut1), density1)
#     # xi1 = xline
#     # yi1 = yline
#     # # xNew, yNew = xi, yi
#     # for i in range(N):
#     #     xOld1 = xi1
#     #     yOld1 = yi1
#     #     xNew1 = hMap(a, b, xOld1, yOld1)
#     #     yNew1 = xOld1
#     #     xi1 = xNew1
#     #     yi1 = yNew1
#     # quadxNew = list(xNew1)
#     # quadyNew = list(yNew1)
#
#     quadyNew, quadxNew = getPoints(a,b,left_cut1,right_cut1,p,density1,dx,dy, N,'forwards', 'standard')
#
#     model = np.poly1d(np.polyfit(quadyNew, quadxNew, 2))
#     lmodel = list(model.c)
#     coef = ["aq", "bq", "cq"]
#     result = dict(zip(coef, lmodel))
#     # result["aq"] is my coefficient
#     y1 = ((1 / result1["al"]) - result["bq"]) / (2 * result["aq"])
#     x1 = result["aq"] * y1 ** 2 + result["bq"] * y1 + result["cq"]
#     x2 = (1 / result1["al"]) * y1 - result1["bl"] / result1["al"]
#     return (x1 - x2)
#
#
# # print(d(1.1,-.45,10))
# # print(d(1.3125,-0.3,13))
# delta = 0.0001
# M = 10
#
#
# # a = 1.31
# # b= -0.3
# ############################ derivative of d ###############
# def derd(a, b):
#     return ((d(a + delta, b, M)) - (d(a - delta, b, M))) / (2 * delta)
#
#
# # print(derd(1.31,-0.3))
# # b = -0.35
# # a = 1.17
# # a1 = a - d(a,b,M) / derd(a,b)
# # print("a1 is",a1)
#
# def NewtonsMethod(d, b, a, tolerance):
#     # i=0
#     counter = 0
#     while True:
#         counter = counter + 1
#         # =i+1
#         a1 = a - d(a, b, M) / derd(a, b)
#         dis = abs(d(a1, b, M))
#         if dis < tolerance:
#             # print(i)
#             break
#
#         a = a1
#     return a, counter
#
#
# # print(NewtonsMethod(d, -0.3, 0.0001))
# # print("By Newton's method, we get for b = -0.3, the approximate a is",NewtonsMethod(d, a, 10**-5),M)
# ###### negative distance means no intersection ##########
# # print("And the distance(error) is",d(NewtonsMethod(d, a, 10**-5),M))
#
# # (b)=a
# dens = 5000
# dens1 = 10000
# b1 = -0.6
# b2 = -0.3
#
# b4 = -0.7
# b5 = -0.2
# # C = np.zeros(dens)
# # B = np.linspace(b1,b2,dens)  ############# 0.3-0.6
# # B1 = np.linspace(b3,b2,dens) ############## 0.4-0.6
# B2 = np.linspace(b4, b5, dens1)  ###### 0.2-0.7
# C2 = np.zeros(dens1)  ### 0.7-0.2
# A2 = np.zeros(dens1)  ### 0.7-0.2
# # A= np.zeros(dens)   ######## -0.4
# # A1 = np.zeros(dens)   ######### -0.6
# for i in range(dens1):
#     # if i == 0:
#     a = 0.339
#     A2[0], C2[0] = NewtonsMethod(d, -0.7, a, 0.00001)
#     # if i == 1:
#
#     A2[1], C2[1] = NewtonsMethod(d, B2[1], a, 0.00001)
#     # if i == 1:
#     # a= 0.95
#     # A[1],C[1]=NewtonsMethod(d, -0.45+1/dens,a, 0.0001)
#     if i > 1:
#         slope = (A2[i - 1] - A2[i - 2]) / ((b5 - b4) / dens1)
#         A2[i], C2[i] = NewtonsMethod(d, B2[i], slope * B2[i] + (A2[i - 1] - slope * B2[i - 1]), 0.00001)
# # A = [NewtonsMethod(d,B[i],A[i-1],10**-3) for i in range(dens)]
#
# print(B2)
# print(A2)
# print(C2)
# # print(B)
# # print(A1)
# # plt.plot(B,A1)
#
#
# # print(A)
# # %%
# dens2 = 1000
# C = np.zeros(dens2)
# A = np.zeros(dens2)
# B = np.linspace(b4, b1, dens2)
# for i in range(dens2):
#     # if i == 0:
#     a = 0.339
#     A[0], C[0] = NewtonsMethod(d, -0.7, a, 0.0001)
#     # if i == 1:
#
#     A[1], C[1] = NewtonsMethod(d, B[1], a, 0.0001)
#     # if i == 1:
#     # a= 0.95
#     # A[1],C[1]=NewtonsMethod(d, -0.45+1/dens,a, 0.0001)
#     if i > 1:
#         slope = (A[i - 1] - A[i - 2]) / ((b1 - b4) / dens2)
#         A[i], C[i] = NewtonsMethod(d, B[i], slope * B[i] + (A[i - 1] - slope * B[i - 1]), 0.0001)
#
# print(B)
# print(A)
# print(C)
#
# # %%
# plt.plot(B, A)
# plt.plot(B, C)
#
#
# # # %% store section
# # def save_to_file(arr, filename):
# #     arr = [str(i) for i in arr]
# #     with open(filename, 'w') as f:
# #         for i in arr:
# #             f.write(str(i) + "\n")
# #
# #
# # # save
# # save_to_file(A, "a values for b 0.7-0.6,dens 1000.txt")
