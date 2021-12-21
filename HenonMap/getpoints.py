import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

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


# a,b = 1.3212113142620452, -0.3
# a= 1.3146 good value
a, b = 1.3146, -0.3  ### cuts work for change of a under 0.05, work for change of b under 0.005
density = 50000
density1 = 50
rho = 0.2
p1, p2 = getPeriodic(a, b)

p = p2

# cut for unstable #############################
N = 4
if N == 3:
    left_cut = 0.1
    right_cut = 0.25
if N == 4:
    left_cut = 0.0
    right_cut = 0.77
# left_cut = 0.02
# right_cut = 0.07
if N == 5:
    left_cut = 9 * 10 ** -3
    right_cut = 1.5 * 10 ** -2
if N == 6:
    left_cut = 2.2 * 10 ** -3
    right_cut = 4 * 10 ** -3
if N == 7:
    left_cut = 6 * 10 ** -4
    right_cut = 1 * 10 ** -3
if N == 8:
    left_cut = 1.5 * 10 ** -4
    right_cut = 2.5 * 10 ** -4
if N == 9:
    left_cut = 3.5 * 10 ** -5
    right_cut = 7 * 10 ** -5
if N == 10:
    left_cut = 9 * 10 ** -6
    right_cut = 1.6 * 10 ** -5
if N == 11:
    left_cut = 2.5 * 10 ** -6
    right_cut = 4 * 10 ** -6
if N == 12:
    left_cut = 5.5 * 10 ** -7
    right_cut = 1.2 * 10 ** -6
if N == 13:
    left_cut = 1.7 * 10 ** -7
    right_cut = 3. * 10 ** -7
if N == 14:
    left_cut = 4.5 * 10 ** -8
    right_cut = 7.4 * 10 ** -8
if N == 15:
    left_cut = 3.8 * 10 ** -6
    right_cut = 3.9 * 10 ** -6
if N == 16:
    left_cut = 7 * 10 ** -7
    right_cut = 7.6 * 10 ** -7
if N == 17:
    left_cut = 7.85 * 10 ** -7
    right_cut = 8.05 * 10 ** -7
if N == 18:
    left_cut = 4.65 * 10 ** -8
    right_cut = 5.15 * 10 ** -8
if N == 19:
    left_cut = 1.70 * 10 ** -8
    right_cut = 1.79 * 10 ** -8
if N == 20:
    left_cut = 4.4 * 10 ** -9
    right_cut = 4.62 * 10 ** -9
if N == 21:
    left_cut = 8.3 * 10 ** -10
    right_cut = 8.9 * 10 ** -10
if N == 22:
    left_cut = 9.1 * 10 ** -10
    right_cut = 9.5 * 10 ** -10
if N == 23:
    left_cut = 7.5 * 10 ** -11
    right_cut = 8.2 * 10 ** -11
if N == 24:
    left_cut = 7.15 * 10 ** -11
    right_cut = 7.45 * 10 ** -11
if N == 25:
    left_cut = 5.15 * 10 ** -12
    right_cut = 5.45 * 10 ** -12
if N == 26:
    left_cut = 9.7 * 10 ** -13
    right_cut = 1.03 * 10 ** -12
if N == 27:
    left_cut = 3.45 * 10 ** -13
    right_cut = 3.65 * 10 ** -13
if N == 28:
    left_cut = 6.45 * 10 ** -14
    right_cut = 6.9 * 10 ** -14
if N == 29:
    left_cut = 1.7 * 10 ** -14
    right_cut = 1.82 * 10 ** -14
if N == 30:
    # left_cut =6*10**-15
    # right_cut = 7.2*10**-15
    left_cut = (8.1 * 10 ** -14)  ## density = 10 for best result
    right_cut = 8.4 * 10 ** -14
# np.longdouble()
########################### cut for stable $##############
if N == 3:
    left_cut_b = 8.5 * 10 ** -2
    right_cut_b = 9.7 * 10 ** -2
if N == 4:
    left_cut_b = 6.9 * 10 ** -3
    right_cut_b = 7.8 * 10 ** -3
if N == 5:
    left_cut_b = 5.1 * 10 ** -4
    right_cut_b = 6.1 * 10 ** -4
if N == 6:
    left_cut_b = 3.9 * 10 ** -5
    right_cut_b = 4.7 * 10 ** -5
if N == 7:
    left_cut_b = 3.1 * 10 ** -6
    right_cut_b = 3.8 * 10 ** -6
if N == 8:
    left_cut_b = 2.4 * 10 ** -7
    right_cut_b = 2.9 * 10 ** -7
if N == 9:
    left_cut_b = 1.9 * 10 ** -8
    right_cut_b = 2.2 * 10 ** -8
if N == 10:
    left_cut_b = 1.4 * 10 ** -9
    right_cut_b = 1.7 * 10 ** -9
if N == 11:
    left_cut_b = 1.2 * 10 ** -10
    right_cut_b = 1.3 * 10 ** -10
if N == 12:
    left_cut_b = 8.9 * 10 ** -12
    right_cut_b = 10 * 10 ** -12
if N == 13:
    left_cut_b = 6.939 * 10 ** -13
    right_cut_b = 7.593 * 10 ** -13
if N == 14:
    left_cut_b = 5.5 * 10 ** -14
    right_cut_b = 6. * 10 ** -14
if N == 15:
    # left_cut_b =3.58*10**-15
    # right_cut_b =9.9*10**-15
    left_cut_b = 1.57162855 * 10 ** -9  # works for a =1.3212113142620452
    right_cut_b = 1.57167525 * 10 ** -9

if N == 16:
    left_cut_b = 1.57164 * 10 ** -9
    right_cut_b = 1.57165000 * 10 ** -9
if N == 17:
    left_cut_b = 0.0000000000007
    right_cut_b = .0000000000009
if N == 18:
    left_cut_b = 0.0000000000007
    right_cut_b = .0000000000009
if N == 19:
    left_cut_b = 0.0000000000007
    right_cut_b = .0000000000009
if N == 20:
    left_cut_b = 0.0000000000007
    right_cut_b = .0000000000009
if N > 20:
    left_cut_b = 0
    right_cut_b = 0

################################Forward time (Unstable/red)
# direction vector in x and y
evecUn = getUnstableLin(a, b, p)
dx = evecUn[0] * rho
dy = evecUn[1] * rho
xline = np.linspace(p + (left_cut * dx), p + (dx * right_cut), density)
yline = np.linspace(p + (left_cut * dy), p + (dy * right_cut), density)
xi = xline
yi = yline
xNew, yNew = xi, yi
for i in range(N):
    xOld = xi
    yOld = yi
    xNew = hMap(a, b, xOld, yOld)
    yNew = xOld
    xi = xNew
    yi = yNew

# polynomial fit with degree = 2
quadxNew = list(xNew)
quadyNew = list(yNew)
model = np.poly1d(np.polyfit(quadyNew, quadxNew, 2))
lmodel = list(model.c)
coef = ["aq", "bq", "cq"]
result = dict(zip(coef, lmodel))
# result["aq"] is my coefficient


##################################Backwards time (Stable)

# direction vector in x and y
evecSt = getStableLin(a, b, p)  # eigenvector for stable manifold

dx_b = evecSt[0] * rho
dy_b = evecSt[1] * rho

xi_b = np.linspace(p + (left_cut_b * dx_b), p + (dx_b * right_cut_b), density1)
yi_b = np.linspace(p + (left_cut_b * dy_b), p + (dy_b * right_cut_b), density1)

xNew_b, yNew_b = xi_b, yi_b

if (True):  # to conveniently hide backward if needed
    for i in range(N):
        xOld_b = xi_b
        yOld_b = yi_b
        yNew_b = backwards_hMap(a, b, xOld_b, yOld_b)
        xNew_b = yOld_b
        xi_b = xNew_b
        yi_b = yNew_b

lxNewb = list(xNew_b)
lyNewb = list(yNew_b)

# polynomial fit with degree = 1
model1 = np.poly1d(np.polyfit(lxNewb, lyNewb, 1))
lmodel1 = list(model1.c)
coef1 = ["al", "bl"]
result1 = dict(zip(coef1, lmodel1))
polyline = np.linspace(-2, 2, 50)

########################################where they intersect
# Suppose intersect at (x1,y1）for Unstable Manifold,intersect at (x1,y2）for Stable Manifold note x,y switched
x1 = (-(1 / result1["al"]) - result["bq"]) / (2 * result["aq"])
y1 = result["aq"] * x1 ** 2 + result["bq"] * x1 + result["cq"]
y2 = (1 / result1["al"]) * x1 - result1["bl"] / result1["al"]

# Set up plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
plt.title('Approximation of Hénon Orbit, # Iterations = ' + str(N) + ', parameters a= ' + str(a) + ', b = ' + str(b))
lim = 3
xmid = 1
ymid = 0
plt.xlim([xmid - lim, xmid + lim])
plt.ylim([ymid - lim, ymid + lim])

tanpoint = ((y1 + y2) / 2, x1)
ax.plot(polyline, model1(polyline))
ax.plot(xNew_b, yNew_b, '.', color='blue', alpha=0.9, markersize=5)
ax.plot(xNew, yNew, '.', color='red', alpha=0.9, markersize=5)
ax.plot([p1, p2], [p1, p2], 'x', color='black', alpha=1, markersize=10)
x = np.linspace(0, 4, 400)
y = np.linspace(-1, 1, 400)
x, y = np.meshgrid(x, y)
ax.contour(x, y, (x - result["aq"] * y ** 2 - result["bq"] * y - result["cq"]), [0], colors='red', alpha=.3)
print("The formula for linear approximation is", "y1 =", result1["al"], "*x1 +", result1["bl"])
print("The formula for quadratic approximation is", "x2 =", result["aq"], "*(y2)^2 +", result["bq"], "y2 +",
      result["cq"])
print("The approximate tangence point on stable manifold is", "(", y1, ",", x1, ")")
print("The approximate tangence point on unstable manifold is", "(", y2, ",", x1, ")")
print("The distance(error) between two points is", y1 - y2)
print("By taking the average value, the approximate intersection is", tanpoint)