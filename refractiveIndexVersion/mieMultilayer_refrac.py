import scipy.constants as sc
import scipy.special
import scipy
import pandas as pd
import numpy as np
import math
import json

settingsPath = open("mieSettingsRefrac.txt")
settings = json.load(settingsPath)

numLayers = int(settings["numLayers"])
lamda = int(settings["wavelength"])
radii = np.array(settings["radii"]).astype(int)
refractiveStart = int(settings["refractiveStart"])
refractiveStop = int(settings["refractiveStop"])
refractiveInterval = float(settings["refractiveInterval"])
refracArray = np.arange(refractiveStart, refractiveStop, refractiveInterval)
refracArray = refracArray.reshape(len(refracArray), 1)
for x in range(1, numLayers):
    refracArray = np.hstack((refracArray, refracArray))
outputFile = str(settings["outputFileName"])

if(str(settings["varyRealImag"]) == "real"):
    imagComponent = int(settings["constantComponent"])
    refracArray = refracArray + imagComponent*1j
    refracArray = np.hstack((refracArray, np.ones((len(refracArray), 1))))

if(str(settings["varyRealImag"]) == "imag"):
    realComponent = int(settings["constantComponent"])
    refracArray = realComponent + refracArray*1j
    refracArray = np.hstack((refracArray, np.ones((len(refracArray), 1))))

n_m = 1.33  # real part of the refractive index of medium
k_m = 0  # imaginary part of refractive index of medium
N_m = n_m+(k_m*1j)
m_m = 1  # relative refractive index of medium
kappa = (2*sc.pi*N_m)/lamda


def rA(layerIndex, radiiIndex):  # Radii Adjustment
    # layerIndex = index of desired refractive index (innermost = 1)
    # radiiIndex = index of radius (innermost = 1)
    # The idea of this function is to produce m_l*r_i for any layerIndex (l) and radiiIndex (i)
    return (refracArray[:, layerIndex-1]*2*sc.pi*radii[radiiIndex-1]*N_m)/lamda


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
xL = kappa*radii[len(radii)-1]
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
        return (RR(n, rA(2, 1)))*(((refracArray[:, 1]*D1(n, rA(1, 1)))-(refracArray[:, 0]*D1(n, rA(2, 1))))/((refracArray[:, 1]*D1(n, rA(1, 1)))-(refracArray[:, 0]*D3(n, rA(2, 1)))))
    else:
        return (RR(n, rA(l, l-1)))*(((refracArray[:, l-1]*Hal(n, l-1))-(refracArray[:, l-2]*D1(n, rA(l, l-1))))/((refracArray[:, l-1]*Hal(n, l-1))-(refracArray[:, l-2]*D3(n, rA(l, l-1)))))


def Hbl(n, l):
    return ((RR(n, rA(l, l))*D1(n, rA(l, l)))-(Bl(n, l)*D3(n, rA(l, l))))/(RR(n, rA(l, l))-Bl(n, l))


def Bl(n, l):
    if (l == 2):
        return (RR(n, rA(2, 1)))*(((refracArray[:, 0]*D1(n, rA(1, 1)))-(refracArray[:, 1]*D1(n, rA(2, 1))))/((refracArray[:, 0]*D1(n, rA(1, 1)))-(refracArray[:, 1]*D3(n, rA(2, 1)))))
    else:
        return (RR(n, rA(l, l-1)))*(((refracArray[:, l-2]*Hbl(n, l-1))-(refracArray[:, l-1]*D1(n, rA(l, l-1))))/((refracArray[:, l-2]*Hbl(n, l-1))-(refracArray[:, l-1]*D3(n, rA(l, l-1)))))

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
    return (cext(L))/(sc.pi*(radii[L-1]**2))


def Qsca(L):
    return (csca(L))/(sc.pi*(radii[L-1]**2))


def Qabs(L):
    return Qext(L)-Qsca(L)


def Qnf(l):
    sumd = 0
    for n in range(1, lastTerm+1):
        sumd = sumd + (((np.abs(Al(n, l)))**2)*(((n+1)*(np.abs(spherical_hn2(n-1, rA(l, l-1))))**2)+(n*(np.abs(
            spherical_hn2(n+1, rA(l, l-1))))**2)))+(((2*n)+1)*((np.abs(Bl(n, l)))**2)*(np.abs(spherical_hn2(n, rA(l, l-1))))**2)
    return 0.5*sumd


file = open(outputFile, "w+")
header = "\t".join(["#Refrac", "Qext", "Qsca", "Qabs", "Qnf"])

file.write(header)
file.write("\n")
print(header)
list1 = np.around(refracArray[:,0], 2)
list2 = Qext(numLayers).real
list3 = Qsca(numLayers).real
list4 = Qabs(numLayers).real
Qnfl = np.array(Qnf(numLayers+1))
for i in range(0, len(refracArray)):
    text = "\t".join([str(list1[i]),
                     str(list2[i]),
                     str(list3[i]),
                     str(list4[i]),
                     str(Qnfl[i])])
    file.write(text+"\n")
    print(text)
file.close()

print("\nProcessing is finished. File " + outputFile + " created")