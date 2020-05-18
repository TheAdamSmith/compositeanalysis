import numpy as np
from numpy.linalg import inv
import Smith_Adam_Project2 as proj2
import Smith_Adam_Project1 as proj1
import Smith_Adam_Project3 as proj3
import math

def getWmn(matProp, theta, hk, p0, loadtype, a, b, x, y,  numIters):
    ABD = proj2.getABD(matProp, theta, hk)
    D = np.zeros([3, 3], dtype=np.float64)
    for i in range(3, 6):
        for j in range(3, 6):
            D[i - 3, j - 3] = ABD[i, j]
    P = np.zeros([numIters + 1, numIters + 1], dtype=np.float64)
    if (loadtype == "uniform"):
        for m in range(1, numIters + 1):
            for n in range(1, numIters + 1):
                P[m, n] = 16 * p0 / ((np.pi ** 2) * m * n)
    elif loadtype == "sine":
        for m in range(1, numIters + 1):
            for n in range(1, numIters + 1):
                P[m,n] = p0
    elif loadtype == "hydro":
        for m in range(1, numIters + 1):
            for n in range(1, numIters + 1):
                P[m, n] = 8 * p0 *(-1**(m+1))/ ((np.pi ** 2) * m * n)
    else:
        raise ValueError('Invalid Load Type')

    W = np.zeros([numIters + 1, numIters + 1], dtype=np.float64)
    for m in range(1, numIters + 1):
        for n in range(1, numIters + 1):
            den = D[0, 0] * (m * np.pi / a) ** 4 + 2 * (D[0, 1] + 2 * D[2, 2]) * ((m * np.pi / a) ** 2) * (
                        (n * np.pi / b) ** 2) + D[1, 1] * (n * np.pi / b) ** 4
            W[m, n] = P[m, n] / den
    return W

def getW(matProp, theta, hk, p0, loadtype, a, b, x, y,  numIters):


    W = getWmn(matProp, theta, hk, p0, loadtype, a, b, x, y, numIters)
    w = 0
    if loadtype == "hydro":
        mStep = 1
    else:
        mStep = 2
    if loadtype == "sine":
        numIters = 1

    for m in range(1, numIters+1, mStep):
            for n in range(1, numIters+1, 2):
                w = w + W[m,n]*math.sin(math.pi*x/a)*math.sin(math.pi*y/b)

    return w


def getCurvatures(matProp, theta, hk, p0, loadtype, a, b,  x, y, numIters):
    W = getWmn(matProp, theta, hk, p0, loadtype, a, b, x, y, numIters)
    if loadtype == "hydro":
        mStep = 1
    else:
        mStep = 2
    if loadtype == "sine":
        numIters = 1

    Kx = 0
    for m in range(1, numIters + 1, mStep):
        for n in range(1, numIters + 1, 2):
            Kx = Kx + ((math.pi**2)/(a**2))*W[m, n] * math.sin(math.pi * x / a) * math.sin(math.pi * y / b)

    Ky = 0
    for m in range(1, numIters + 1, mStep):
        for n in range(1, numIters + 1, 2):
            Ky = Ky + ((math.pi ** 2) / (b ** 2))*W[m, n] * math.sin(math.pi * x / a) * math.sin(math.pi * y / b)

    Kxy = 0
    for m in range(1, numIters + 1, mStep):
        for n in range(1, numIters + 1, 2):
            Kxy = Kxy -2*((math.pi ** 2) / (a * b))*W[m, n] * math.cos(math.pi * x / a) * math.cos(math.pi * y / b)

    K = np.matrix([Kx, Ky, Kxy])
    K = np.transpose(K)
    return K

def probDriver(matProp, theta, hk, p0, loadtype, a, b, numIters):

    np.set_printoptions(precision=3)
    print("Lamina Properites of %s Hyer SI" % matProp[1])
    print("E1          E2          v12        G12     ")

    # retrieve and print material properties
    E1, E2, v12, G12 = proj1.getProperties("graphite")
    print(E1, "     ", E2, "     ", v12, "     ", G12, "\n")

    # print Q matrix
    Q = proj1.getQ(matProp[0])
    print("For %s material Q matrix:" % matProp[1])
    print(Q)
    print("\n")

  #print laminate
    print("Lamina\tMaterial\tThickness(mm)\tOrientation")
    print("---------------------------------------------------------")
    for i in range(0, theta.size):
        print(i+1,"\t\t", matProp[i], "\t\t", hk, "\t\t", theta[i])
        print("---------------------------------------------------------")
    print("\n")

    # print Qbar
    Qbar = np.zeros((3, 3))
    for i in range(0, theta.size):
        Qbar = proj1.getQbar(matProp[i], theta[i])
        print("For Lamina %i theta = %f and %s material\n Qbar:" % (i + 1, theta[i], matProp[i]))
        print(Qbar)
        print("\n")


    # print ABD
    ABD = proj2.getABD(matProp, theta, hk)
    print("ABD Matrix:")
    print(ABD)
    print("\n")

    # print abd
    abd = proj2.getabd(matProp, theta, hk)
    print("abd matrix:")
    print(abd)
    print("\n")


    D = np.zeros([3, 3], dtype=np.float64)
    for i in range(3, 6):
        for j in range(3, 6):
            D[i - 3, j - 3] = ABD[i, j]

    print("\nD Matrix for the Laminate\n")
    print( D)
    x = np.arange(0, a + .005, .005)
    y = np.arange(0, b + .005, .005)
    W = np.zeros([x.size, y.size], dtype=np.float64)
    sigma1Map = np.zeros([x.size, y.size], dtype=np.float64)
    sigma2Map = np.zeros([x.size, y.size], dtype=np.float64)
    tauMap = np.zeros([x.size, y.size], dtype=np.float64)
    for i in range(0, x.size):
        for j in range(0, y.size):
            W[i, j] = getW(matProp, theta, hk, p0, loadtype, a, b, x[i], y[j], numIters)
            curv = getCurvatures(matProp, theta, hk, p0, loadtype, a, b, x[i], y[j], numIters)

            sigma12 = proj3.onAxisStress(matProp, theta, hk, curv)
            sigma1 = sigma12[:, 0]
            sigma2 = sigma12[:, 1]
            tau = sigma12[:, 2]

            sigma1Map[i,j] = np.amax(abs(sigma1))
            sigma2Map[i, j] = np.amax(abs(sigma2))
            tauMap[i, j] = np.amax(abs(tau))

    indW = np.unravel_index(np.argmax(abs(W), axis=None), W.shape)

    ind1 = np.unravel_index(np.argmax(sigma1Map, axis=None), sigma1Map.shape)
    ind2 = np.unravel_index(np.argmax(sigma2Map, axis=None), sigma2Map.shape)
    ind3 = np.unravel_index(np.argmax(tauMap, axis=None), tauMap.shape)
    curv = getCurvatures(matProp, theta, hk, p0, loadtype, a, b, x[indW[0]], y[indW[1]], numIters)
    print("\nMaximum Deflection\n")
    print("x\t\ty\t  W\t\tKx\t\t   Ky   \t\tKxy")
    print(x[indW[0]], y[indW[1]], "%.4f" %W[indW], curv[0], curv[1], curv[2] )
    print("\nMaximum Stresses")
    print("x\t\ty\t  Sigma1")
    print(x[ind1[0]], y[ind1[1]], "%.4E" %sigma1Map[ind1])
    print("\nx\t\ty\t  Sigma2")
    print(x[ind2[0]], y[ind2[1]], "%.4E" % sigma2Map[ind2])
    print("\nx\ty\t  Tau")
    print(x[ind3[0]], y[ind3[1]], "  %.4E" % tauMap[ind3])

print("Project 4  Case A   Smith, Adam     Spring 2018\n------------------------------------------------------\n")

theta = np.array([0, 90, 90, 0])
matProp = ["graphite", "graphite", "graphite", "graphite", "graphite", "graphite"]
hk = .15*10**-3
a = 0.125
b  = 0.1
p0 = 120 *10**3

probDriver(matProp, theta, hk, p0, "uniform", a, b, 100)
print("\n------------------------------------------------------\n")
print("\n\nProject 4  Case B   Smith, Adam     Spring 2018\n------------------------------------------------------\n")

theta = np.array([0, 90, 0])
matProp = ["graphite", "graphite", "graphite", "graphite", "graphite", "graphite"]
hk = .15*10**-3
a = 0.075
b  = 0.125
p0 = 180 *10**3

probDriver(matProp, theta, hk, p0, "uniform", a, b, 10)
print("\n------------------------------------------------------\n")

print("\n\nProject 4  Case C   Smith, Adam     Spring 2018\n------------------------------------------------------\n")

theta = np.array([0, 90, 90, 0])
matProp = ["graphite", "graphite", "graphite", "graphite", "graphite", "graphite"]
hk = .15*10**-3
a = .125
b  = 0.075
p0 = 100 *10**3

probDriver(matProp, theta, hk, p0, "hydro", a, b, 10)

