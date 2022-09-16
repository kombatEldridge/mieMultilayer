#########################################################################
#                                                                       #
#           The intention of this version is to copy all                #
#           availible resources from scattnlay (publicly sourced)       #
#           and investigate our own additions to the project.           #
#                                                                       #
#########################################################################

################## CHANGE LOG ##################
# All changes come from mieMultilayer_1.0.py
# Changed all instances of "dielectric" to "refractive index"


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
refractiveDataPath = np.array(settings["refractiveData"])
radii = np.array(settings["radii"]).astype(int)
refractiveColumns = np.array(settings["refractiveColumns"]).astype(int)
startWavelength = int(settings["startWavelength"])
stopWavelength = int(settings["stopWavelength"])
requestInterval = int(settings["intervalWavelength"])
outputFile = str(settings["outputFileName"])

lamda = np.arange(startWavelength, stopWavelength + requestInterval, requestInterval)

n_m = 1.33  # real part of the refractive index of medium
k_m = 0  # imaginary part of refractive index of medium
N_m = n_m+(k_m*1j)
m_m = 1  # relative refractive index of medium
kappa = (2*sc.pi*N_m)/lamda

def interpolation(wavelengthRequested, refractivesTemp, wavelengthsTemp):
    # For loop finds the brackets of wavelengths in the actual list the wavelengthRequest sits between
    for i in range(0, len(wavelengthsTemp)):
        if (wavelengthsTemp[i] < wavelengthRequested and wavelengthsTemp[i+1] >= wavelengthRequested):
            j = i
            break
    # If the wavelengthRequest sits between the first and second wavelength or just before the last wavelength, we run linear interpolation
    if (j == 0 or j == len(wavelengthsTemp)-2):
        # Eq. 3 from https://www.appstate.edu/~grayro/comphys/lecture4_11.pdf
        return refractivesTemp[j]+(((refractivesTemp[j+1]-refractivesTemp[j])/(wavelengthsTemp[j+1]-wavelengthsTemp[j]))*(wavelengthRequested-wavelengthsTemp[j]))
    else:
        # Lagrangian 4-point interpolation (p.5) from https://www.appstate.edu/~grayro/comphys/lecture4_11.pdf
        x = wavelengthRequested
        x1 = wavelengthsTemp[j-1]
        x2 = wavelengthsTemp[j]
        x3 = wavelengthsTemp[j+1]
        x4 = wavelengthsTemp[j+2]
        return ((((x - x2)*(x - x3)*(x - x4))/((x1 - x2)*(x1 - x3)*(x1 - x4))) * refractivesTemp[j-1]) + ((((x - x1)*(x - x3)*(x - x4))/((x2 - x1)*(x2 - x3)*(x2 - x4))) * refractivesTemp[j]) + ((((x - x1)*(x - x2)*(x - x4))/((x3 - x1)*(x3 - x2)*(x3 - x4))) * refractivesTemp[j+1]) + ((((x - x1)*(x - x2)*(x - x3))/((x4 - x1)*(x4 - x2)*(x4 - x3))) * refractivesTemp[j+2])


def interpolationProcessor(start, stop, path):
    refractiveDataTemp = pd.read_csv(
        path, sep='\t', header=None, skiprows=1).values
    wavelengthsTemp = refractiveDataTemp[:, 0].astype(float)
    refractiveDataComplex = (
        refractiveDataTemp[:, 1] + (refractiveDataTemp[:, 2]*1j))/N_m

    if (wavelengthsTemp[0] > start):
        print("Error: Please make sure start wavelength is greater than or equal to the first wavelength in your refractive file:" & path & ".")
    elif (wavelengthsTemp[0] == start):
        startIndex = 0
    else:
        for i in range(0, len(wavelengthsTemp)):
            if (wavelengthsTemp[i] < start and wavelengthsTemp[i+1] >= start):
                startIndex = i+1
                break
    if (wavelengthsTemp[len(wavelengthsTemp)-1] < stop):
        print("Error: Please make sure stop wavelength is less than or equal to the last wavelength in your refractive files.")

    currWavelength = start + requestInterval
    newrefractiveArray = [refractiveDataComplex[startIndex]]

    for i in np.arange(start + requestInterval, stop + requestInterval, requestInterval):
        if (currWavelength > stop):
            break
        realTemp = interpolation(
            currWavelength, refractiveDataComplex.real, wavelengthsTemp)
        imagTemp = interpolation(
            currWavelength, refractiveDataComplex.imag, wavelengthsTemp)
        newrefractiveArray.append((realTemp + (imagTemp*1j)))
        currWavelength = currWavelength + requestInterval
    newrefractiveArray = np.array(newrefractiveArray)
    return newrefractiveArray


firstDielColumn = interpolationProcessor(
    startWavelength, stopWavelength, refractiveDataPath[0])
refractiveArray = np.array(firstDielColumn).reshape(len(firstDielColumn), 1)
for x in range(1, len(refractiveDataPath)):
    test = interpolationProcessor(
        startWavelength, stopWavelength, refractiveDataPath[x])
    refractiveArray = np.hstack(
        (refractiveArray, np.array(test).reshape(len(test), 1)))
# last column is relative refractive index of medium
refractiveArray = np.hstack(
    (refractiveArray, np.ones((len(refractiveArray), 1))))


def rA(layerIndex, radiiIndex):  # Radii Adjustment
    # layerIndex = index of desired refractive index (innermost = 1)
    # radiiIndex = index of radius (innermost = 1)
    # The idea of this function is to produce m_l*r_i for any layerIndex (l) and radiiIndex (i)
    return (refractiveArray[:, layerIndex-1]*2*sc.pi*radii[radiiIndex-1]*N_m)/lamda


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

# radial adjustment without refractive component
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


def phi(n, z):
    return z*scipy.special.spherical_jn(n, z)
    # First kind of Spherical Bessel Function where n is the order and z is the parameter


def Dphi(n, z):
    return (z*scipy.special.spherical_jn(n, z, True))+(scipy.special.spherical_jn(n, z))
    # Derivative of phi using product rule


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
    return Dphi(n, z)/phi(n, z)


def D3(n, z):
    return Dzeta(n, z)/zeta(n, z)


def RR(n, z):
    return phi(n, z)/zeta(n, z)


def Hal(n, l):
    return ((RR(n, rA(l, l))*D1(n, rA(l, l)))-(Al(n, l)*D3(n, rA(l, l))))/(RR(n, rA(l, l))-Al(n, l))


def Al(n, l):
    if (l == 2):
        return (RR(n, rA(2, 1)))*(((refractiveArray[:, 1]*D1(n, rA(1, 1)))-(refractiveArray[:, 0]*D1(n, rA(2, 1))))/((refractiveArray[:, 1]*D1(n, rA(1, 1)))-(refractiveArray[:, 0]*D3(n, rA(2, 1)))))
    else:
        return (RR(n, rA(l, l-1)))*(((refractiveArray[:, l-1]*Hal(n, l-1))-(refractiveArray[:, l-2]*D1(n, rA(l, l-1))))/((refractiveArray[:, l-1]*Hal(n, l-1))-(refractiveArray[:, l-2]*D3(n, rA(l, l-1)))))


def Hbl(n, l):
    return ((RR(n, rA(l, l))*D1(n, rA(l, l)))-(Bl(n, l)*D3(n, rA(l, l))))/(RR(n, rA(l, l))-Bl(n, l))


def Bl(n, l):
    if (l == 2):
        return (RR(n, rA(2, 1)))*(((refractiveArray[:, 0]*D1(n, rA(1, 1)))-(refractiveArray[:, 1]*D1(n, rA(2, 1))))/((refractiveArray[:, 0]*D1(n, rA(1, 1)))-(refractiveArray[:, 1]*D3(n, rA(2, 1)))))
    else:
        return (RR(n, rA(l, l-1)))*(((refractiveArray[:, l-2]*Hbl(n, l-1))-(refractiveArray[:, l-1]*D1(n, rA(l, l-1))))/((refractiveArray[:, l-2]*Hbl(n, l-1))-(refractiveArray[:, l-1]*D3(n, rA(l, l-1)))))

def cext(L):
    sumd = 0
    for n in range(1, lastTerm+1):
        sumd = sumd+((2*n)+1)*(Al(n, L+1).real+Bl(n, L+1).real)
    return ((2*sc.pi*sumd)/kappa**2).real


def csca(L):
    sumd = 0
    for n in range(1, lastTerm+1):
        sumd = sumd+((2*n)+1)*((np.abs(Al(n, L+1))**2)+(np.abs(Bl(n, L+1))**2))
    return ((2*sc.pi*sumd)/kappa**2).real


def cabs(L):
    return cext(L)-csca(L)


def Qext(L):
    i = L-1
    # incase any of the outer layers are the same as the medium
    while i > (-1):
        if (refractiveArray[:, i] != m_m).all():
            solid = i  # set the solid outer layer index (core = 1)
            break
        i -= 1
    if (solid == 0):
        return 0
    else:
        return (cext(L))/(sc.pi*(radii[solid]**2))


def Qsca(L):
    i = L-1
    # incase any of the outer layers are the same as the medium
    while i > (-1):
        if (refractiveArray[:, i] != m_m).all():
            solid = i  # set the solid outer layer index (core = 1)
            break
        i -= 1
    if (solid == 0):
        return 0
    else:
        return (csca(L))/(sc.pi*(radii[solid]**2))


def Qabs(L):
    return Qext(L)-Qsca(L)


def Qnf(l):
    sumd = 0
    for n in range(1, lastTerm+1):
        sumd = sumd + (((np.abs(Al(n, l)))**2)*(((n+1)*(np.abs(spherical_hn2(n-1, rA(l, l-1))))**2)+(n*(np.abs(
            spherical_hn2(n+1, rA(l, l-1))))**2)))+(((2*n)+1)*((np.abs(Bl(n, l)))**2)*(np.abs(spherical_hn2(n, rA(l, l-1))))**2)
    return 0.5*sumd


file = open(outputFile, "w+")
qnf_header = "Qnf_1"
if (numLayers > 1):
    for i in range(2, numLayers + 1):
        qnf_header = "\t".join([qnf_header, "Qnf_" + str(i)])
header = "\t".join(["#Lambda", "Qext", "Qsca", "Qabs",
                   "Cext", "Csca", "Cabs", qnf_header])

file.write(header)
file.write("\n")

list1 = lamda.real
list2 = Qext(numLayers).real
list3 = Qsca(numLayers).real
list4 = Qabs(numLayers).real
list5 = cext(numLayers).real
list6 = csca(numLayers).real
list7 = cabs(numLayers).real
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
                     str(list5[i]),
                     str(list6[i]),
                     str(list7[i]),
                     str(QnfText)])
    file.write(text+"\n")
file.close()

print("Processing is finished. File " + outputFile + " created")