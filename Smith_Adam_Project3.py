import numpy as np
from numpy.linalg import inv
import Smith_Adam_Project2 as proj2
import Smith_Adam_Project1 as proj1
import math

def offAxisStress(matProp, theta, hk, k):

    if isinstance(hk, (np.ndarray, np.generic)):
        z = hk
    else:
        z = np.zeros(theta.size + 1)
        for i in range(theta.size + 1):
            z[i] = i * hk

    z[:] = [element - z[theta.size] / 2 for element in z]


    ep = [0, 0, 0]


    sigma = np.zeros((theta.size*2, 3, 1),dtype=np.float64)

    j = 0

    for i in range(0, theta.size):
        Qbar = proj1.getQbar(matProp[i], theta[i])
        sigma[j] = Qbar*( z[i]*k)
        j = j + 1
        sigma[j] = Qbar*( z[i+1]*k)
        j = j+1


    return sigma

def onAxisStress(matProp, theta, hk, k):
    sigma12 = np.zeros((theta.size * 2, 3, 1), dtype=np.float64)
    sigmaxy = offAxisStress(matProp, theta, hk, k)
    for i in range(0,theta.size*2):
        m = math.cos(math.radians(theta[math.floor(i/2)]))
        n = math.sin(math.radians(theta[math.floor(i/2)]))
        T = np.matrix([[m * m, n * n, 2 * m * n], [n * n, m * m, -2 * m * n], [-m * n, m * n, m * m - n * n]],
                      dtype=np.float64)
        sigma12[i] = T*sigmaxy[i]
    return sigma12

def failureCrit(matProp, theta, hk, NM):

    sigFailGraph = np.array([-1250, 1500, -200, 50, 100])
    sigFailGlass = np.array([-600, 1000, -120, 30, 70])
    sigFailGraph = sigFailGraph*10**6
    sigma12 = onAxisStress(matProp, theta, hk, NM)


    F1 = 0.1333*10**-9
    F2 = 15 *10**-9
    F11 = 0.533 *10**-18
    F22 = 100 *10**-18
    F66 = 100*10**-18


    Fac1 = np.zeros(theta.size*2)
    Fac2 = np.zeros(theta.size*2)
    Fac12 = np.zeros(theta.size*2)
    Tsai = np.zeros(theta.size * 2)

    for i in range(0, theta.size*2):
        if (sigma12[i, 0] > 0):
            Fac1[i] = sigma12[i,0]/sigFailGraph[1]
        else:
            Fac1[i] = sigma12[i, 0] / sigFailGraph[0]
        if (sigma12[i, 1] > 0):
            Fac2[i] = sigma12[i, 1] / sigFailGraph[3]
        else:
            Fac2[i] = sigma12[i, 1] / sigFailGraph[2]
        Fac12[i] = sigma12[i,2]/sigFailGraph[4]
        Tsai[i] = F1 * sigma12[i, 0] + F2 * sigma12[i, 1] + F11 * (sigma12[i, 0] ** 2) + F22 * (
                    sigma12[i, 1] ** 2) + F66 * (sigma12[i, 2] ** 2) - math.sqrt(F11 * F22) * sigma12[i, 0] * sigma12[
                      i, 1]

    return Fac1, Fac2, Fac12, Tsai

def getEpZbar(matProp, theta, hk, NM):
    sigma = onAxisStress(matProp, theta, hk, NM)
    eps_bar_Z = 0
    H = hk*theta.size
    j = 0
    for i in range(0, theta.size):
        if (matProp[i] == "graphite"):
            S13 = -1.6 * 10**-12
            S23 = -37.9*10**-12

        eps_bar_Z = eps_bar_Z + (S13*sigma[j][0]+ S23*sigma[j][1])*hk/H

        j = j+2

    return eps_bar_Z


def unitThermalLoads(matProp, theta, alpha1, alpha2):
    #initialize z
    if isinstance(hk, (np.ndarray, np.generic)):
        z = hk
    else:
        z = np.zeros(theta.size + 1)
        for i in range(theta.size + 1):
            z[i] = i * hk
    z[:] = [element - z[theta.size] / 2 for element in z]

    #initialize loads and moments
    N_hat_T= np.zeros((3,1), dtype=np.float64)
    M_hat_T = np.zeros((3,1), dtype=np.float64)
    #apply equation 11.71
    for i in range(0, theta.size):

        #sin and cos
        m = math.cos(math.radians(theta[i]))
        n = math.sin(math.radians(theta[i]))

        #on axis CTE (1/deg C)
        alphax = alpha1*(m**2) + alpha2*(n**2)
        alphay = alpha1*(n**2) + alpha2*(m**2)
        alphaxy = 2*(alpha1 - alpha2)*m*n

        #put in matrix for matrix multiplication
        Alpha_XY = np.matrix([alphax, alphay, alphaxy])
        Alpha_XY = Alpha_XY.transpose()

        #retrieve Qbar
        Qbar = proj1.getQbar(matProp[i], theta[i])

        #unit thermal load N/m/deg C
        N_hat_T = N_hat_T + Qbar*Alpha_XY*(z[i+1]-z[i])

        #unit thermal moment N-m/m/deg C
        M_hat_T = M_hat_T + Qbar*Alpha_XY*((z[i+1]**2)-(z[i]**2))

    #combine for matrix operation later
    #NM_hat_T = np.matrix([N_hat_T.item(0), N_hat_T.item(1), N_hat_T.item(2), M_hat_T.item(0), M_hat_T.item(1), M_hat_T.item(2)])

    NM_hat_T = np.zeros((6,1), dtype=np.float64)
    for i in range(0, 2):
        NM_hat_T[i] = N_hat_T[i]

    for i in range(0, 2):
        NM_hat_T[i+3] = M_hat_T[i]
    return NM_hat_T




def getSmearPoissonZ(matProp, theta, hk):
    #for x
    #apply unit Nx
    NM = np.matrix([1, 0, 0, 0, 0, 0])
    NM = NM.transpose()

    eps_bar_Z =getEpZbar(matProp, theta, hk, NM)
    strains = proj2.getStrains(matProp, theta, hk, NM)

    v_xz_bar = eps_bar_Z/strains[0]

    # for x

    # apply unit Nx
    NM = np.matrix([0, 1, 0, 0, 0, 0])
    NM = NM.transpose()

    eps_bar_Z = getEpZbar(matProp, theta, hk, NM)
    strains = proj2.getStrains(matProp, theta, hk, NM)

    v_yz_bar = eps_bar_Z / strains[1]

    return v_xz_bar, v_yz_bar

def getAlphaSmeared(matProp, theta, hk, alpha1, alpha2):

    NM_hat_T = unitThermalLoads(matProp, theta, alpha1, alpha2)
    abd = proj2.getabd(matProp, theta, hk)

    alpha_bar = abd*NM_hat_T
    alpha_bar_x = alpha_bar[0]
    alpha_bar_y = alpha_bar[1]
    alpha_bar_xy = alpha_bar[2]

    alpha_bar_Z = 0

    sigma = onAxisStress(matProp, theta, hk, NM_hat_T)
    eps_bar_Z = 0
    H = hk*theta.size
    j = 0
    alpha3 = 24.3*10**-6
    for i in range(0, theta.size):
        if (matProp[i] == "graphite"):
            S13 = -1.6 * 10**-12
            S23 = -37.9*10**-12

        alpha_bar_Z = alpha_bar_Z + (alpha3+S13*sigma[j][0] + S23*sigma[j][1])*hk/H

        j = j+2

    return alpha_bar_x, alpha_bar_y, alpha_bar_xy, alpha_bar_Z

def probDriver(matProp, theta, hk, NM):

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

    #print loads
    print("Applied Loads")
    print("Nx\t\t   Ny\t\t Nxy\tMx\t\t My\t\t Mxy")
    print(np.transpose(NM),"\n\n")

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

    # print strains
    strains = proj2.getStrains(matProp, theta, hk, NM)
    print("Strains:")
    print(strains)
    print("\n")
    print("---------------------------------------------------------")

    Exbar, Eybar, Gxybar, vxybar, vyxbar, eta_xy_x_bar, eta_xy_y_bar, eta_y_xy_bar, eta_x_xy_bar = proj2.getSmearedProp(
        matProp, theta, hk)
    print("Smeared Properties:")
    print("Exbar = %.3e" % Exbar)
    print("Eybar = %.3e" % Eybar)
    print("Gxybar = %.3e" % Gxybar)
    print("vxybar = %.3e" % vxybar)
    print("vyxbar = %.3e" % vyxbar)
    print("eta_xy_x_bar = %.3e" % eta_xy_x_bar)
    print("eta_xy_ybar = %.3e" % eta_xy_y_bar)
    print("eta_y_xy_bar = %.3e" % eta_y_xy_bar)
    print("eta_x_xy_bar = %.3e" % eta_x_xy_bar)
    print("---------------------------------------------------------\n")

    sigmaxy = offAxisStress(matProp, theta, hk, NM)

    print('Lamina \t z \t\t     sigmax \t\t sigmay \t\t tauxy')
    print("---------------------------------------------------------")
    j=0
    for i in range(0, theta.size):
        print(i+1, "\t\ttop   \t", "\t%.3e\t" % sigmaxy[j,0], "\t%.3e\t" % sigmaxy[j,1], "\t%.3e\t" % sigmaxy[j,2])
        j = j + 1
        print(i + 1, "\t\tbottom\t", "\t%.3e\t" % sigmaxy[j, 0],"\t%.3e\t" %  sigmaxy[j, 1], "\t%.3e\t" % sigmaxy[j, 2])
        j = j+1

    print("---------------------------------------------------------\n")
    epxy = np.zeros((theta.size*2, 3, 1),dtype=np.float64)

    for i in range(0, theta.size*2):

        Sbar = proj1.getSbar(matProp[math.floor(i/2)], theta[math.floor(i/2)])
        epxy[i]=Sbar*sigmaxy[i]

    print('Lamina \t z \t\t     epsx \t\t\t epsy \t\t\t gammay')
    print("---------------------------------------------------------")
    j = 0
    for i in range(0, theta.size):
        print(i + 1, "\t\ttop   \t", "\t%.3e\t" % epxy[j, 0], "\t%.3e\t" % epxy[j, 1], "\t%.3e\t" % epxy[j, 2])
        j = j + 1
        print(i + 1, "\t\tbottom\t", "\t%.3e\t" % epxy[j, 0], "\t%.3e\t" % epxy[j, 1], "\t%.3e\t" % epxy[j, 2])
        j = j + 1

    print("---------------------------------------------------------\n")



    sigma12 = onAxisStress(matProp, theta, hk, NM)

    print('Lamina \t z \t\t     sigma1\t\t sigma2 \t\t tau12')
    print("---------------------------------------------------------")
    j=0
    for i in range(0, theta.size):
        print(i+1, "\t\ttop   \t", "\t%.3e\t" % sigma12[j,0], "\t%.3e\t" % sigma12[j,1], "\t%.3e\t" % sigma12[j,2])
        j = j + 1
        print(i + 1, "\t\tbottom\t", "\t%.3e\t" % sigma12[j, 0],"\t%.3e\t" %  sigma12[j, 1], "\t%.3e\t" % sigma12[j, 2])
        j = j+1

    print("---------------------------------------------------------\n")

    ep12 = np.zeros((theta.size * 2, 3, 1), dtype=np.float64)

    for i in range(0, theta.size * 2):
        S = proj1.getS(matProp[math.floor(i / 2)])
        ep12[i] = S * sigma12[i]

    print('Lamina \t z \t\t     epsx \t\t\t epsy \t\t\t gammay')
    print("---------------------------------------------------------")
    j = 0
    for i in range(0, theta.size):
        print(i + 1, "\t\ttop   \t", "\t%.3e\t" % ep12[j, 0], "\t%.3e\t" % ep12[j, 1], "\t%.3e\t" % ep12[j, 2])
        j = j + 1
        print(i + 1, "\t\tbottom\t", "\t%.3e\t" % ep12[j, 0], "\t%.3e\t" % ep12[j, 1], "\t%.3e\t" % ep12[j, 2])
        j = j + 1

    print("---------------------------------------------------------\n")

    Fac1, Fac2, Fac12, Tsai = failureCrit(matProp, theta, hk, NM)
    print('Lamina \t z \t\t     Fac1 \t\t\t Fac2 \t\t\t Fac12\t\t\t Tsai')
    print("---------------------------------------------------------")
    j = 0
    for i in range(0, theta.size):
        print(i + 1, "\t\ttop   \t", "\t%.3e\t" % Fac1[j], "\t%.3e\t" % Fac2[j], "\t%.3e\t" % Fac12[j],
              "\t%.3e\t" % Tsai[j])
        j = j + 1
        print(i + 1, "\t\tbottom\t", "\t%.3e\t" % Fac1[j], "\t%.3e\t" % Fac2[j], "\t%.3e\t" % Fac12[j],
              "\t%.3e\t" % Tsai[j])
        j = j + 1

    print("---------------------------------------------------------\n")

def project3a():
    theta = np.array([30,-30,0,0,-30,30])
    matProp = ["graphite", "graphite", "graphite", "graphite", "graphite", "graphite"]
    NM = np.matrix([800000, -20000, 70000, 0, 0, 0])
    NM = NM.transpose()
    hk = 0.15*10**-3
    # problem 1
    print("Project 3a   Problem 1   Smith, Adam     Spring 2018"
              "\n------------------------------------------------------\n")
    probDriver(matProp, theta, hk, NM)\

    #problem 2#problem 2
    NM = np.matrix([0, 0, 0, -5.0, 20.0, 5.0])
    NM = NM.transpose()


    print("Project 3a   Problem 2   Smith, Adam     Spring 2018"
              "\n------------------------------------------------------\n")
    probDriver(matProp, theta, hk, NM)

    #Verification
    NM = np.matrix([-0.425*10**6, 0, 0, 0, 0, 0])
    NM = NM.transpose()
    print("Project 3a   Verification using table 10.7 in Hyer   Smith, Adam     Spring 2018"
              "\n------------------------------------------------------\n")
    probDriver(matProp, theta, hk, NM)


# theta = np.array([0, 90, 90, 0])
# #NM = unitThermalLoads(matProp, theta, alpha1, alpha2)
# #NM.transpose()
# #print(NM)
# NM = NM *-150
# #     print(NM)
# #probDriver(matProp, theta, hk, NM)
#
# NM = np.matrix([0, 0, 255, 0, 0, 0])
# NM = NM.transpose()
# theta = np.array([30, -30, 0, 0, -30, 30])
# matProp = ["graphite", "graphite", "graphite", "graphite", "graphite", "graphite","graphite", "graphite", "graphite", "graphite", "graphite", "graphite"]
# hk = 0.15 * 10 ** -3
# v_xz_bar, v_yz_bar = getSmearPoissonZ(matProp, theta, hk)
#
# print(v_xz_bar)
# print(v_yz_bar)
#
# alpha_bar_x, alpha_bar_y, alpha_bar_xy, alpha_bar_Z = getAlphaSmeared(matProp, theta, hk, alpha1, alpha2)
#
# print(alpha_bar_Z)