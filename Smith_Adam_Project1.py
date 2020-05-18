#Smith, Adam
#MAE 5060
#Computer Assignment 1
import math #import math for exponentials
import numpy as np
from numpy.linalg import inv

#function to retrieve material properties
def getProperties(material):
    if material == "graphite":
        E1 = 155.0 *10**9
        E2 = 12.1 *10**9
        v12 = 0.248
        G12 = 4.4 *10**9


    elif material == "glass":
        E1 = 50.0*10**9
        E2 = 15.2*10**9
        v12 = 0.254
        G12 = 4.7*10**9

    else:
        print("Unable to find properties")
    return E1, E2, v12, G12
#function to get S matrix from Eq. 4.5
def getS(matProp):
    E1, E2, v12, G12 = getProperties(matProp)
    S = np.matrix([[1/E1, -v12/E1, 0], [-v12/E1, 1/E2, 0], [0, 0, 1/G12]], dtype= np.float64)
    return S

#function to get Q matrix from S matrix using Eq. 4.16
def getQ(matProp):

    #retrieve S matrix from material properties
    S = getS(matProp)

    #Calculate values using Eq. 4.16

    #add them to a matrix
    Q = inv(S)

    return Q

#function to calculate Sbar from S matrix and tranformation matrix  using Eq. 5.26
def getSbar(matProp, theta):

    #calculate trig values
    m = math.cos(math.radians(theta))
    n = math.sin(math.radians(theta))
    T = np.matrix([[m*m, n*n, 2*m*n], [n*n, m*m, -2*m*n], [-m*n, m*n, m*m-n*n]],dtype= np.float64)
    R = np.matrix([[1,0,0], [0,1,0], [0,0,2]])
    #retrieve S matrix
    S = getS(matProp)

    #Apply Eq. 5.26

    S11 = S[0,0]*(m**4) + (2*S[0,1]+S[2,2])*(n**2)*(m**2) + S[1,1]*(n**4)
    S12 = (S[0,0]+S[1,1] -S[2,2])*(n**2)*(m**2)+S[0,1]*(n**4+m**4)
    S16 = (2*S[0,0] -2*S[0,1] - S[2,2])*n*(m**3) - (2*S[1,1] - 2*S[0,1]-S[2,2])*m*(n**3)
    S22 = S[0,0]*(n**4) + (2*S[0,1] + S[2,2])*(n**2)*(m**2) + S[1,1] *(m**4)
    S26 = (2*S[0,0] - 2*S[0,1]-S[2,2])*(m)*(n**3) - (2*S[1,1] - 2* S[0,1] - S[2,2])* n* (m**3)
    S66 = 2*(2*S[0,0] +S[1,1] -4*S[0,1]- S[2,2])*(n**2)*(m**2) +S[2,2]*(n**4+m**4)
    #add to matrix
    #Sbar = R*inv(T)*inv(R)*S*T
    Sbar = np.matrix([[S11, S12, S16], [S12, S22, S26], [S16, S26, S66]])
    return Sbar

#Function to get Qbar matrix from Q matrix and transformation matrix using Eq. 5.84
def getQbar(matProp, theta):

    #calculate trig values
    m = math.cos(math.radians(theta))
    n = math.sin(math.radians(theta))
    T = np.matrix([[m * m, n * n, 2 * m * n], [n * n, m * m, -2 * m * n], [-m * n, m * n, (m * m) - (n * n)]], dtype= np.float64)
    R = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 2]])

    #retrieve Q
    S = getQ(matProp)

    #Apply Eq. 5.84
    S11 = S[0, 0] * (m ** 4) + 2*(S[0, 1] + 2*S[2, 2]) * (n ** 2) * (m ** 2) + S[1, 1] * (n ** 4)
    S12 = (S[0, 0] + S[1, 1] - 4*S[2, 2]) * (n ** 2) * (m ** 2) + S[0, 1] * (n ** 4 + m ** 4)
    S16 = (S[0, 0] - S[0, 1] - 2*S[2, 2]) * n * (m ** 3) + (  S[0, 1]-S[1, 1] + 2*S[2, 2]) * m * (n ** 3)
    S22 = S[0, 0] * (n ** 4) + 2*(S[0, 1] + 2*S[2, 2]) * (n ** 2) * (m ** 2) + S[1, 1] * (m ** 4)
    S26 = (S[0, 0] - S[0, 1] - 2*S[2, 2]) * (m) * (n ** 3) - (S[0, 1]-S[1, 1] + 2*S[2, 2]) * n * (m ** 3)
    S66 = (S[0, 0] + S[1, 1] - 2 * S[0, 1] - 2*S[2, 2]) * (n ** 2) * (m ** 2) + S[2, 2] * (n ** 4 + m ** 4)

    #add to matrix
    #Qbar = inv(T)*Q*R*T*inv(R)

    Qbar = np.matrix([[S11, S12, S16], [S12, S22, S26], [S16, S26, S66]])

    return Qbar

#main driver function
def main():

    #format initial output
    print("\nComputer Assignment #1: Graphite Polymer Properites 41 degree lamina\n\n")
    print("Lamina Properites")
    print("E1          E2          v12        G12      theta")

    #retrieve and print material properties
    E1, E2, v12, G12 = getProperties("graphite")
    print( E1,"     ", E2, "     ", v12, "     ", G12, "     ", 41 , "\n\n")

    #calculate and print S and Q matrix
    S = getS("graphite")
    Q = getQ("graphite")
    print("S Matrix (GPa^-1):\n", S[0], "\n", S[1], "\n",  S[2], "\n\n")
    print("Q Matrix (GPa):\n", Q[0], "\n", Q[1], "\n", Q[2],"\n\n")

    #calculate and print Sbar and Qbar matrix at 41 degrees
    Sbar41 = getSbar("graphite", 41)
    Qbar41 = getQbar("graphite", 41)
    print("Sbar Matrix (GPa^-1) at 41 deg:\n", Sbar41[0], "\n", Sbar41[1], "\n", Sbar41[2], "\n\n")
    print("Qbar Matrix (GPa) at 41 deg:\n", Qbar41[0], "\n", Qbar41[1], "\n", Qbar41[2], "\n\n")

    print("Veriction Matrices at 30 deg for comparison to HW 3 Problem 3,4 and given values in book\n")

    #calculate Qbar and Sbar at 30 degrees for verification puposes
    Sbar30 = getSbar("graphite", 90)
    Qbar30 = getQbar("graphite", 90)
    print("Sbar Matrix (GPa^-1) at 30 deg:\n", Sbar30[0], "\n", Sbar30[1], "\n", Sbar30[2], "\n\n")
    print("Qbar Matrix (GPa) at 30 deg:\n", Qbar30[0], "\n", Qbar30[1], "\n", Qbar30[2], "\n\n")

#call to main
#main()