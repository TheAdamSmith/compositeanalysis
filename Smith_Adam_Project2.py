import Smith_Adam_Project1 as proj1
import numpy as np
from numpy.linalg import inv

def getABD(matProp, theta, hk):
    if isinstance(hk, (np.ndarray, np.generic)):
        z = hk
    else:
        z = np.zeros(theta.size+1)
        for i in range(theta.size+1):
            z[i] = i*hk

    z[:] = [element - z[theta.size]/2 for element in z]

    A = np.zeros((3,3), dtype=np.float64)
    B = np.zeros((3,3), dtype=np.float64)
    D = np.zeros((3,3), dtype=np.float64)
    delZ = np.zeros(z.size)
    for i in range(0, z.size-1):
        delZ[i] = (z[i+1])-(z[i])
    for i in range(0, theta.size):
        Qbar = proj1.getQbar(matProp[i], theta[i])
        A = A + ((z[i+1]-z[i])*Qbar)#*np.power(10, 6)
        B = B + ((Qbar*(np.power(z[i+1], 2) - np.power(z[i], 2)))/2)#p*np.power(10, 3)
        D = D + (Qbar*(np.power(z[i+1], 3) - np.power(z[i], 3)))/3

    ABD = np.zeros((6, 6), dtype=np.float64)

    for i in range(0, 3):
        for j in range(0, 3):
            ABD[i, j] = A.item(i, j)
    for i in range(0, 3):
        for j in range(3, 6):
            ABD[i, j] = B.item(i, j - 3)
    for i in range(3, 6):
        for j in range(0, 3):
            ABD[i, j] = B[i - 3, j]
    for i in range(3, 6):
        for j in range(3, 6):
            ABD[i, j] = D.item(i - 3, j - 3)

    return ABD

def getabd(matProp, theta, hk):

    ABD = getABD(matProp, theta, hk)
    abd = inv(ABD)

    return abd

def getStrains(matProp, theta, hk, NM):

    abd = getabd(matProp, theta, hk)
    strains = np.matmul(abd, NM)

    return strains

def getNM(matProp, theta, hk, strains):


    ABD = getABD(matProp, theta, hk)

    NM = np.matmul(ABD, strains)


    return NM



def getSmearedProp(matProp, theta, hk):
    H = hk*theta.size
    abd  = getabd(matProp, theta, hk)
    Exbar = 1/(abd[0,0]*H) #1/H*a11
    Eybar = 1/(abd[1,1]*H) #1/H*a22
    Gxybar = 1/(abd[2,2]*H) #1/H*a66
    vxybar = -abd[0,1]/abd[0,0] #-a12/a11
    vyxbar = -abd[0,1]/abd[1,1] ##-a12/a22
    eta_xy_x_bar = abd[0,2]/abd[0,0] #a16/a11
    eta_xy_y_bar = abd[0,2]/abd[1,1] #a16/a22
    eta_y_xy_bar = abd[1,1]/abd[0,2] #a22/a16
    eta_x_xy_bar = abd[0, 0] / abd[0, 2]  # a11/a16

    return Exbar, Eybar, Gxybar, vxybar, vyxbar, eta_xy_x_bar, eta_xy_y_bar, eta_y_xy_bar, eta_x_xy_bar

def problem1():

    #format matrix print
    np.set_printoptions(precision=3)
    #problem 1
    print("Project 2    Problem 1   Smith, Adam     Spring 2018"
          "\n------------------------------------------------------\n")

    #given values
    theta = np.array([0, 0, 0, 36, 36, 36])
    matProp = ["graphite", "graphite", "graphite", "graphite", "graphite", "graphite"]
    hk = 0.15*10**-3
    NM = np.matrix([1000, -200, 100, 0.05, 0.05, -0.05])
    NM = NM.transpose()


    print("Lamina Properites of %s Hyer SI" % matProp[1])
    print("E1          E2          v12        G12     ")

    # retrieve and print material properties
    E1, E2, v12, G12 = proj1.getProperties("graphite")
    print(E1, "     ", E2, "     ", v12, "     ", G12, "\n")


    #print Q matrix
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

    #print Qbar
    Qbar = np.zeros((3, 3))
    for i in range(0, theta.size):
        Qbar = proj1.getQbar(matProp[i], theta[i])
        print("For Lamina %i theta = %f and %s material\n Qbar:" %( i+1, theta[i], matProp[i]))
        print(Qbar)
        print("\n")


    #print loads
    print("Applied Loads")
    print("Nx\t\t   Ny\t\t Nxy\tMx\t\t My\t\t Mxy")
    print(np.transpose(NM),"\n\n")

    #print ABD
    ABD = getABD(matProp, theta, hk)
    print("ABD Matrix:")
    print(ABD)
    print("\n")

    #print abd
    abd = getabd(matProp, theta, hk)
    print("abd matrix:")
    print(abd)
    print("\n")

    #print strains
    strains = getStrains(matProp, theta, hk, NM)
    print("Strains:")
    print(strains)
    print("\n")
    print("---------------------------------------------------------")

def problem2():
    np.set_printoptions(precision=3)
    #problem 2
    print("Project 2    Problem 2   Smith, Adam     Spring 2018"
          "\n------------------------------------------------------\n")

    #given values
    theta = np.array([0, 0, 30, 30, 30, 30, 0, 0])
    matProp = ["graphite", "graphite", "graphite", "graphite", "graphite", "graphite", "graphite", "graphite"]
    hk = 0.15*10**-3
    strains= np.matrix([0.01, 0, 0, 1, 0, 0])
    strains= strains.transpose()

    print("Lamina Properites of %s Hyer SI" % matProp[1])
    print("E1          E2          v12        G12     ")
    # retrieve and print material properties
    E1, E2, v12, G12 = proj1.getProperties("graphite")
    print(E1, "     ", E2, "     ", v12, "     ", G12, "\n")

    #print Q
    Q = proj1.getQ(matProp[0])
    print("For %s material Q matrix:" % matProp[1])
    print(Q)
    print("\n")

    # print laminate
    print("Lamina\tMaterial\tThickness(mm)\tOrientation")
    print("---------------------------------------------------------")
    for i in range(0, theta.size):
        print(i + 1, "\t\t", matProp[i], "\t\t", hk, "\t\t", theta[i])
        print("---------------------------------------------------------")
    print("\n")

    #print Qbar
    Qbar = np.zeros((3,3))
    for i in range(0, theta.size):
        Qbar = proj1.getQbar(matProp[i], theta[i])
        print("For Lamina %i theta = %f and %s material\n Qbar:" % (i + 1, theta[i], matProp[i]))
        print(Qbar)
        print("\n")

    # print strains
    print("Enforced Strains")
    print(" Epx  Epy\tEpxy\tKx\t Ky\t Kxy")
    print(np.transpose(strains), "\n\n")

    #print ABd
    ABD = getABD(matProp, theta, hk)
    print("ABD Matrix:")
    print(ABD)
    print("\n")

    #print abd
    abd = getabd(matProp, theta, hk)
    print("abd matrix:")
    print(abd)
    print("\n")

    #print loads
    NM = getNM(matProp, theta, hk, strains)
    print("Loads and Moments")
    print(NM)
    print("\n")


    Exbar, Eybar, Gxybar, vxybar, vyxbar, eta_xy_x_bar, eta_xy_y_bar, eta_y_xy_bar, eta_x_xy_bar = getSmearedProp(matProp,theta,hk)
    print("Smeared Properties:")
    print("Exbar = ", Exbar)
    print("Eybar = ", Eybar)
    print("Gxybar = ", Gxybar)
    print("vxybar = ", vxybar)
    print("vyxbar = ", vyxbar)
    print("eta_xy_x_bar = ", eta_xy_x_bar)
    print("eta_xy_ybar = ", eta_xy_y_bar)
    print("eta_y_xy_bar = ", eta_y_xy_bar)
    print("eta_x_xy_bar = ", eta_x_xy_bar, "\n")
    print("---------------------------------------------------------")


def problem3():
    np.set_printoptions(precision=3)
    # problem 3
    print("Project 2   Problem 3   Smith, Adam     Spring 2018"
          "\n------------------------------------------------------\n")

    #given values
    theta = np.array([35, -35])
    matProp = ["graphite", "glass"]
    hk = np.array([-2*10**-3, 0, 3*10**-3])
    NM = np.matrix([1000, -200, 100, 0.05, 0.05, -0.05])
    NM = NM.transpose()

    for i in range(0, len(matProp)):
        print("Lamina Properites of %s Hyer SI" % matProp[i])
        print("E1          E2          v12        G12     ")
        # retrieve and print material properties
        E1, E2, v12, G12 = proj1.getProperties(matProp[i])
        print(E1, "     ", E2, "     ", v12, "     ", G12, "\n")

    #print Q
    for i in range(0, len(matProp)):
        Q = proj1.getQ(matProp[i])
        print("For %s material Q matrix:" % matProp[i])
        print(Q)
        print("\n")

    # print laminate
    print("Lamina\tMaterial\tThickness(mm)\tOrientation")
    print("---------------------------------------------------------")
    for i in range(0, theta.size):
        print(i + 1, "\t\t", matProp[i], "\t\t", hk, "\t\t", theta[i])
        print("---------------------------------------------------------")
    print("\n")


    #print Qbar
    Qbar = np.zeros((3, 3))
    for i in range(0, theta.size):
        Qbar = proj1.getQbar(matProp[i], theta[i])
        print("For theta = %f and %s material Qbar:" % (theta[i], matProp[i]))
        print(Qbar)
        print("\n")


    # print loads
    print("Applied Loads")
    print("Nx\t\t   Ny\t\t Nxy\tMx\t\t My\t\t Mxy")
    print(np.transpose(NM), "\n\n")

    #print ABD
    ABD = getABD(matProp, theta, hk)
    print("ABD Matrix:")
    print(ABD)
    print("\n")

    #print abd
    abd = getabd(matProp, theta, hk)
    print("abd matrix:")
    print(abd)
    print("\n")

    #print strains
    strains = getStrains(matProp, theta, hk, NM)
    print("Strains:")
    print(strains)
    print("\n")
    print("---------------------------------------------------------")

def verification():

    #verification
    print("Project 2   Verification to [+-30/0] Problem 2 HW6  Smith, Adam     Spring 2018"
          "\n------------------------------------------------------\n")
    theta = np.array([30, -30, 0])
    hk  = 0.15
    matProp = ["graphite", "graphite","graphite"]
    ABD = getABD(matProp, theta, 0.15)
    abd = getabd(matProp, theta, 0.15)
    np.set_printoptions(precision=3)

    # print laminate
    print("Lamina\tMaterial\tThickness(mm)\tOrientation")
    print("---------------------------------------------------------")
    for i in range(0, theta.size):
        print(i + 1, "\t\t", matProp[i], "\t\t", hk, "\t\t", theta[i])
        print("---------------------------------------------------------")
    print("\n")


    print("ABD:")
    print(ABD)
    print("\n")
    print("abd:")
    print(abd)
    print("\n")
    NM = np.matrix([51200, 9470, 0, 1.416, -0.609, -1.051])
    print(NM)
    print("\n")
    NM= NM.transpose()
    # print loads
    print("Applied Loads")
    print("Nx\t\t   Ny\t\t Nxy\tMx\t\t My\t\t Mxy")
    print(np.transpose(NM), "\n\n")


    print("Strains found using Applied Loads")
    strains = getStrains(matProp, theta, 0.15, NM)
    print(strains)

    print("given strains")
    strains = np.matrix([.001, 0, 0, 0, 0, 0])

    strains = strains.transpose()
    print(strains, "\n")

    NM = getNM(matProp, theta, 0.15, strains)
    print("Loads found using strains")
    print(NM)
    print("---------------------------------------------------------")


def main():
    problem1()
    problem2()
    problem3()
    verification()


#main()
