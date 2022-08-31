from turtle import hideturtle
import scipy.constants as sc
import scipy.special
import scipy
import pandas as pd
import numpy as np
import math
import json

settingsPath = open("mieSettings.txt")
settings = json.load(settingsPath)

numLayers = int(settings["numLayers"])
dielectricDataPath = np.array(settings["dielectricData"])
radii = np.array(settings["radii"]).astype(int)
dielectricColumns = np.array(settings["dielectricColumns"]).astype(int)
startWavelength = int(settings["startWavelength"])
stopWavelength = int(settings["stopWavelength"])
requestInterval = int(settings["intervalWavelength"])
outputFile = str(settings["outputFileName"])
lamda = np.arange(startWavelength, stopWavelength + 1, requestInterval)

n_m = 1.33  # real part of the refractive index of medium
k_m = 0  # imaginary part of refractive index of medium
N_m = n_m+(k_m*1j)
m_m = 1  # relative refractive index of medium
kappa = (2*sc.pi*N_m)/lamda

def interpolation(wavelengthRequested, dielectricsTemp, wavelengthsTemp):
    # For loop finds the brackets of wavelengths in the actual list the wavelengthRequest sits between
    for i in range(0, len(wavelengthsTemp)):
        if (wavelengthsTemp[i] < wavelengthRequested and wavelengthsTemp[i+1] >= wavelengthRequested):
            j = i
            break
    # If the wavelengthRequest sits between the first and second wavelength or just before the last wavelength, we run linear interpolation
    if (j == 0 or j == len(wavelengthsTemp)-1):
        # Eq. 3 from https://www.appstate.edu/~grayro/comphys/lecture4_11.pdf
        return dielectricsTemp[j]+(((dielectricsTemp[j+1]-dielectricsTemp[j])/(wavelengthsTemp[j+1]-wavelengthsTemp[j]))*(wavelengthRequested-wavelengthsTemp[j]))
    else:
        # Lagrangian 4-point interpolation (p.5) from https://www.appstate.edu/~grayro/comphys/lecture4_11.pdf
        x = wavelengthRequested
        x1 = wavelengthsTemp[j-1]
        x2 = wavelengthsTemp[j]
        x3 = wavelengthsTemp[j+1]
        x4 = wavelengthsTemp[j+2]
        return ((((x - x2)*(x - x3)*(x - x4))/((x1 - x2)*(x1 - x3)*(x1 - x4))) * dielectricsTemp[j-1]) + ((((x - x1)*(x - x3)*(x - x4))/((x2 - x1)*(x2 - x3)*(x2 - x4))) * dielectricsTemp[j]) + ((((x - x1)*(x - x2)*(x - x4))/((x3 - x1)*(x3 - x2)*(x3 - x4))) * dielectricsTemp[j+1]) + ((((x - x1)*(x - x2)*(x - x3))/((x4 - x1)*(x4 - x2)*(x4 - x3))) * dielectricsTemp[j+2])


def interpolationProcessor(start, stop, path):
    dielectricDataTemp = pd.read_csv(
        path, sep='\t', header=None, skiprows=1).values
    wavelengthsTemp = dielectricDataTemp[:, 0].astype(float)
    dielectricDataComplex = (
        dielectricDataTemp[:, 1] + (dielectricDataTemp[:, 2]*1j))/N_m

    if (wavelengthsTemp[0] > start):
        print("Error: Please make sure start wavelength is greater than or equal to the first wavelength in your dielectric file:" & path & ".")
    elif (wavelengthsTemp[0] == start):
        startIndex = 0
    else:
        for i in range(0, len(wavelengthsTemp)):
            if (wavelengthsTemp[i] < start and wavelengthsTemp[i+1] >= start):
                startIndex = i+1
                break
    if (wavelengthsTemp[len(wavelengthsTemp)-1] < stop):
        print("Error: Please make sure stop wavelength is less than or equal to the last wavelength in your dielectric files.")

    currWavelength = start + requestInterval
    newDielectricArray = [dielectricDataComplex[startIndex]]

    for i in np.arange(start + requestInterval, stop + requestInterval, requestInterval):
        if (currWavelength > stop):
            break
        realTemp = interpolation(
            currWavelength, dielectricDataComplex.real, wavelengthsTemp)
        imagTemp = interpolation(
            currWavelength, dielectricDataComplex.imag, wavelengthsTemp)
        newDielectricArray.append((realTemp + (imagTemp*1j)))
        currWavelength = currWavelength + requestInterval
    newDielectricArray = np.array(newDielectricArray)
    return newDielectricArray


firstDielColumn = interpolationProcessor(
    startWavelength, stopWavelength, dielectricDataPath[0])
dielectricArray = np.array(firstDielColumn).reshape(len(firstDielColumn), 1)
for x in range(1, len(dielectricDataPath)):
    test = interpolationProcessor(
        startWavelength, stopWavelength, dielectricDataPath[x])
    dielectricArray = np.hstack(
        (dielectricArray, np.array(test).reshape(len(test), 1)))
# last column is relative refractive index of medium
dielectricArray = np.hstack(
    (dielectricArray, np.ones((len(dielectricArray), 1))))


def rA(layerIndex, radiiIndex):  # Radii Adjustment
    # layerIndex = index of desired refractive index (innermost = 1)
    # radiiIndex = index of radius (innermost = 1)
    # The idea of this function is to produce m_l*r_i for any layerIndex (l) and radiiIndex (i)
    return (dielectricArray[:, layerIndex-1]*2*sc.pi*radii[radiiIndex-1]*N_m)/lamda


# The next two variables are chosen from Eq 30 from Yang 2003
# The number of terms, Nmax, in the partial-wave expansion
# is a function of the size parameters. Theory and experiment
# show that a good choice for the number of terms is given by
# Nmax = max(Nstop, |rA(j,j)|, |rA(j+1,j)|) + 15
largest = 0
# This for loop finds the largest of all |rA(j,j)| and |rA(j+1,j)|
for j in range(1, numLayers+1):
    if (max(abs(rA(j, j))) > largest):
        largest = max(abs(rA(j, j)))
    if (max(abs(rA(j+1, j))) > largest):
        largest = max(abs(rA(j, j)))

# radial adjustment without dielectric component
xL = max(kappa*radii[len(radii)-1])
if ((xL <= 8)):
    Nstop = np.round(xL + 4*(xL)**(1/3) + 1)
elif ((xL <= 4200)):
    Nstop = np.round(xL + 4.05*(xL)**(1/3) + 2)
elif ((xL <= 20000)):
    Nstop = np.round(xL + 4*(xL)**(1/3) + 2)
else:
    print("Sphere too large. Please make total radius smaller.")

lastTerm = int(max(largest, Nstop.real) + 15)


def psi(n, z):
    return z*scipy.special.spherical_jn(n, z)
    # First kind of Spherical Bessel Function where n is the order and z is the parameter


def Dpsi(n, z):
    return (z*scipy.special.spherical_jn(n, z, True))+(scipy.special.spherical_jn(n, z))
    # Derivative of psi using product rule


def spherical_hn1(n, z):
    return scipy.special.spherical_jn(n, z)+(1j*scipy.special.spherical_yn(n, z))
    # First kind of Spherical Hankel function
    # not in a python library, had to find alternate definition: https://en.wikipedia.org/wiki/Bessel_function#Hankel_functions:_H(1)α,_H(2)α


def spherical_hn2(n, z):
    return scipy.special.spherical_jn(n, z)-(1j*scipy.special.spherical_yn(n, z))
    # from https://en.wikipedia.org/wiki/Bessel_function#Hankel_functions:_H(1)α,_H(2)α


def zeta(n, z):
    return z*(spherical_hn1(n, z))


def Dzeta(n, z):
    return (scipy.special.spherical_jn(n, z)+(1j*scipy.special.spherical_yn(n, z)))+(z*(-spherical_hn1(n+1, z)+(n/z)*spherical_hn1(n, z)))


def D1(n, z):
    return Dpsi(n, z)/psi(n, z)


def D3(n, z):
    return Dzeta(n, z)/zeta(n, z)


def RR(n, z):
    return psi(n, z)/zeta(n, z)


def ratioQ(n, l):
    return ((psi(n, rA(l ,l-1)))/(zeta(n, rA(l ,l-1))))/((psi(n, rA(l ,l)))/(zeta(n, rA(l ,l))))


def G1(n, l):
    return dielectricArray[:, l]*Hal_new(n, l-1)-dielectricArray[:, l-1]*D1(n, rA(l, l-1))


def G2(n, l):
    return dielectricArray[:, l]*Hal_new(n, l-1)-dielectricArray[:, l-1]*D3(n, rA(l, l-1))


def G1hat(n, l):
    return dielectricArray[:, l-1]*Hbl_new(n, l-1)-dielectricArray[:, l]*D1(n, rA(l, l-1))


def G2hat(n, l):
    return dielectricArray[:, l-1]*Hbl_new(n, l-1)-dielectricArray[:, l]*D3(n, rA(l, l-1))


def Hal_new(n, l):
    if (l == 1):
        return D1(n, rA(1, 1))
    else:
        return ((G2(n, l)*D1(n, rA(l, l)))-(ratioQ(n, l)*G1(n, l)*D3(n, rA(l, l))))/(G2(n, l)-(ratioQ(n, l)*G1(n, l)))
    
    
def Hbl_new(n, l):
    if (l == 1):
        return D1(n, rA(1, 1))
    else:
        return ((G2hat(n, l)*D1(n, rA(l, l)))-(ratioQ(n, l)*G1hat(n, l)*D3(n, rA(l, l))))/(G2hat(n, l)-(ratioQ(n, l)*G1hat(n, l)))


def an(n, L):
    return (((Hal_new(n, L)/dielectricArray[:, L])+(n/xL))*psi(n, xL)-psi(n-1, xL))/(((Hal_new(n, L)/dielectricArray[:, L])+(n/xL))*zeta(n, xL)-zeta(n-1, xL))


def bn(n, L):
    return (((Hbl_new(n, L)*dielectricArray[:, L])+(n/xL))*psi(n, xL)-psi(n-1, xL))/(((Hbl_new(n, L)*dielectricArray[:, L])+(n/xL))*zeta(n, xL)-zeta(n-1, xL))


def Qext(L):
    i = L-1
    # incase any of the outer layers are the same as the medium
    while i > (-1):
        if (dielectricArray[:, i] != m_m).all():
            solid = i  # set the solid outer layer index (core = 1)
            break
        i -= 1
    if (solid == 0):
        return 0
    else:
        sumd = 0
        for n in range(1, lastTerm+1):
            sumd = sumd + (2*n+1)*(an(n, solid+2)+bn(n, solid+2)).real
        return (sumd*2)/(kappa**2*radii[solid]**2)


def Qsca(L):
    i = L-1
    # incase any of the outer layers are the same as the medium
    while i > (-1):
        if (dielectricArray[:, i] != m_m).all():
            solid = i  # set the solid outer layer index (core = 1)
            break
        i -= 1
    if (solid == 0):
        return 0
    else:
        sumd = 0
        for n in range(1, lastTerm+1):
            sumd = sumd + (2*n+1)*(np.abs(an(n, solid+2))+np.abs(bn(n, solid+2)))
        return (sumd*2)/(kappa**2*radii[solid]**2)


def Qabs(L):
    return Qext(L)-Qsca(L)


def Qnf(l):
    sumd = 0
    for n in range(1, lastTerm+1):
        sumd = sumd + (((np.abs(Al_new(n, l)))**2)*(((n+1)*(np.abs(spherical_hn2(n-1, rA(l, l-1))))**2)+(n*(np.abs(
            spherical_hn2(n+1, rA(l, l-1))))**2)))+(((2*n)+1)*((np.abs(Bl_new(n, l)))**2)*(np.abs(spherical_hn2(n, rA(l, l-1))))**2)
    return 0.5*sumd


file = open(outputFile, "w+")
qnf_header = "Qnf_1"
if (numLayers > 1):
    for i in range(2, numLayers + 1):
        qnf_header = "\t".join([qnf_header, "Qnf_" + str(i)])
header = "\t".join(["#Lambda", "Qext", "Qsca", "Qabs", qnf_header])

file.write(header)
file.write("\n")

list1 = lamda.real
list2 = Qext(numLayers).real
list3 = Qsca(numLayers).real
list4 = Qabs(numLayers).real
Qnfl = np.array(Qnf(2)).reshape(len(Qnf(2)), 1)
if (numLayers > 1):
    for i in range(3, numLayers + 2):
        Qnfl = np.hstack((Qnfl, np.array(Qnf(i)).reshape(len(Qnf(i)), 1)))

for i in range(0, len(lamda)):
    QnfText = str(Qnfl[i][0])
    for j in range(1, Qnfl.shape[1]):
        QnfText = "\t".join([QnfText, str(Qnfl[i][j])])
    text = "\t".join([str(list1[i]),
                     str(list2[i]),
                     str(list3[i]),
                     str(list4[i]),
                     str(QnfText)])
    file.write(text+"\n")
file.close()

print("Processing is finished. File " + outputFile + " created")