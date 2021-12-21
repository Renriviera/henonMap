import numpy as np
import matplotlib.pyplot as plt


def hMap(a, b, x, y):
    return 1 - a * x ** 2 + b * y


# −b−1(1 − ayn2 − bxn)
def backwards_hMap(a, b, x, y):
    return -(1 / b) * (1 - a * y ** 2 - x)


def getPeriodic(a, b):
    p1 = (-1 * (1 - b) + np.sqrt((1 - b) ** 2 + 4 * a)) / (2 * a)
    p2 = (-1 * (1 - b) - np.sqrt((1 - b) ** 2 + 4 * a)) / (2 * a)

    return p1, p2


a, b = 1.4, 0.3
N = 10
density = 7000
rho = 0.1
p1, p2 = getPeriodic(a, b)

p = p2

xbox = np.linspace(p - rho, p + rho, density)
ybox = np.linspace(p - rho, p + rho, density)

Xbox, Ybox = np.meshgrid(xbox, ybox)
Xi = Xbox
Yi = Ybox
XNew, YNew = Xi, Yi

# Forward time (Unstable)
for i in range(N):
    XOld = Xi
    YOld = Yi
    XNew = hMap(a, b, XOld, YOld)
    YNew = XOld
    Xi = XNew
    Yi = YNew

# Backwards time (Stable)
Xi_b = Xbox
Yi_b = Ybox
XNew_b, Y_new_b = Xi_b, Yi_b
if (True):  # to conveniently hide backward if needed
    for i in range(N):
        XOld_b = Xi_b
        YOld_b = Yi_b
        YNew_b = backwards_hMap(a, b, XOld_b, YOld_b)
        XNew_b = YOld_b
        Xi_b = XNew_b
        Yi_b = YNew_b

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
plt.title('Local Hénon Orbit, # Iterations = ' + str(N) + ', parameters a= ' + str(a) + ', b = ' + str(b))
plt.xlim([-2, 2])
plt.ylim([-2, 2])
ax.plot(XNew_b, YNew_b, '.', color='blue', alpha=0.9, markersize=1)
ax.plot(XNew, YNew, '.', color='red', alpha=0.9, markersize=1)
ax.plot([p1, p2], [p1, p2], 'x', color='black', alpha=1, markersize=10)
plt.show()