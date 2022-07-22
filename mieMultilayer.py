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
startWavelength = int(settings["wavelengthInterval"]["startWavelength"])
stopWavelength = int(settings["wavelengthInterval"]["stopWavelength"])
requestInterval = int(settings["wavelengthInterval"]["intervalWavelength"])
outputFile = str(settings["outputFileName"])

lamda = np.arange(startWavelength, stopWavelength + requestInterval, requestInterval)

n_m = 1.33 # real part of the refractive index of medium
k_m = 0 # imaginary part of refractive index of medium
N_m = n_m+(k_m*1j)
m_m = 1 # relative refractive index of medium

def interpolation(x, y, lamda1):
    for i in range(0,len(lamda1)):
        if(lamda1[i] < x and lamda1[i+1] >= x):
            j=i
            break
    return y[j]+(((y[j+1]-y[j])/(lamda1[j+1]-lamda1[j]))*(x-lamda1[j]))

def interpolationComplete(start, stop, path):
    data1 = pd.read_csv(path, sep='\t', header=None, skiprows=1).values 
    lamda1 = data1[:,0].astype(float)

    mi = []
    temp = (data1[:,1] + (data1[:,2]*1j))/N_m
    mi.append(temp)  
    
    if(lamda1[0] == start):
        startIndex = 0
    else:
        for i in range(0,len(lamda1)):
            if(lamda1[i] < start and lamda1[i+1] >= start):
                startIndex = i+1
                break
    if(lamda1[len(lamda1)-1] < stop):
        print("Error: Please make sure stop wavelength is less than or equal to the last wavelength in your dielectric files.")

    mi_new = []
    curr = start + requestInterval
    ml = [mi[0][startIndex]]
    
    for i in np.arange(start + requestInterval, stop + requestInterval, requestInterval):
        if(curr > stop):
            break
        temp1 = interpolation(curr, mi[0].real, lamda1)
        temp2 = interpolation(curr, mi[0].imag, lamda1)
        ml.append((temp1 + (temp2*1j)))
        curr = curr + requestInterval
    mi_new.append(np.array(ml))
    mi_new = mi_new[0]
    return mi_new

xyz = interpolationComplete(startWavelength, stopWavelength, dielectricDataPath[0])
dielectricArray = np.array(xyz).reshape(len(xyz),1)
for x in range(1, len(dielectricDataPath)):
    test = interpolationComplete(startWavelength, stopWavelength, dielectricDataPath[x])
    dielectricArray = np.hstack((dielectricArray, np.array(test).reshape(len(test),1)))
dielectricArray = np.hstack((dielectricArray, np.ones((len(dielectricArray), 1)))) #last column is relative refractive index of medium

def rA(l, i): #Radii Adjustment
    # l = index of desired refractive index (innermost = 1)
    # i = index of radius (innermost = 1)
    # The idea of this function is to produce m_l*r_i for any l and i
    return (dielectricArray[:,l-1]*2*sc.pi*radii[i-1]*N_m)/lamda

#The next two variables are chosen from Eq 30 from Yang 2003
largest=0
for j in range(1,numLayers+1):
    if(max(rA(j,j).real) > largest):
        largest = max(rA(j,j).real)
    if(max(rA(j+1,j).real) > largest):
        largest = max(rA(j+1,j).real)
lastTerm = math.ceil(abs(largest+4.05*(largest**(1/3))+2))

def phi(n,z): 
    return z*scipy.special.spherical_jn(n,z)
    #Spherical Bessel Function where n is the order and z is the parameter

def Dphi(n,z): 
    return (z*scipy.special.spherical_jn(n,z, True))+(scipy.special.spherical_jn(n,z))
    #Derivative of phi using product rule

def spherical_hn1(n,z):
    return scipy.special.spherical_jn(n,z)+(1j*scipy.special.spherical_yn(n,z))
    #Spherical Hankel function (not in a python library, had to find alternate definition https://en.wikipedia.org/wiki/Bessel_function#Hankel_functions:_H(1)α,_H(2)α)

def SphericalHankelH2(n,z):
    return scipy.special.spherical_jn(n,z)-(1j*scipy.special.spherical_yn(n,z))
    #from https://en.wikipedia.org/wiki/Bessel_function#Hankel_functions:_H(1)α,_H(2)α

def zeta(n,z): 
    return z*(spherical_hn1(n,z))

def Dzeta(n,z): 
    return (scipy.special.spherical_jn(n,z)+(1j*scipy.special.spherical_yn(n,z)))+(z*(-spherical_hn1(n+1,z)+(n/z)*spherical_hn1(n,z)))

def D1(n,z): 
    return Dphi(n,z)/phi(n,z)

def D3(n,z): 
    return Dzeta(n,z)/zeta(n,z)

def RR(n,z): 
    return phi(n,z)/zeta(n,z)

def Hal(n, l):
    return ((RR(n,rA(l,l))*D1(n,rA(l,l)))-(Al(n,l)*D3(n,rA(l,l))))/(RR(n,rA(l,l))-Al(n,l))
    
def Al(n, l):
    if(l==2):
        return (RR(n,rA(2,1)))*(((dielectricArray[:,1]*D1(n,rA(1,1)))-(dielectricArray[:,0]*D1(n,rA(2,1))))/((dielectricArray[:,1]*D1(n,rA(1,1)))-(dielectricArray[:,0]*D3(n,rA(2,1)))))
    else:
        return (RR(n,rA(l,l-1)))*(((dielectricArray[:,l-1]*Hal(n,l-1))-(dielectricArray[:,l-2]*D1(n,rA(l,l-1))))/((dielectricArray[:,l-1]*Hal(n,l-1))-(dielectricArray[:,l-2]*D3(n,rA(l,l-1)))))

def Hbl(n, l):
    return ((RR(n,rA(l,l))*D1(n,rA(l,l)))-(Bl(n,l)*D3(n,rA(l,l))))/(RR(n,rA(l,l))-Bl(n,l))

def Bl(n, l):
    if(l==2):
        return (RR(n,rA(2,1)))*(((dielectricArray[:,0]*D1(n,rA(1,1)))-(dielectricArray[:,1]*D1(n,rA(2,1))))/((dielectricArray[:,0]*D1(n,rA(1,1)))-(dielectricArray[:,1]*D3(n,rA(2,1)))))
    else:
        return (RR(n,rA(l,l-1)))*(((dielectricArray[:,l-2]*Hbl(n,l-1))-(dielectricArray[:,l-1]*D1(n,rA(l,l-1))))/((dielectricArray[:,l-2]*Hbl(n,l-1))-(dielectricArray[:,l-1]*D3(n,rA(l,l-1)))))

kappa = (2*sc.pi*N_m)/lamda

def cext(L): 
    sumd=0
    for n in np.arange(1, lastTerm+1):
        sumd=sumd+((2*n)+1)*(Al(n,L+1).real+Bl(n,L+1).real)
    return ((2*sc.pi*sumd)/kappa**2).real

def csca(L):
    sumd=0
    for n in np.arange(1, lastTerm+1):
        sumd=sumd+((2*n)+1)*((np.abs(Al(n,L+1))**2)+(np.abs(Bl(n,L+1))**2))
    return ((2*sc.pi*sumd)/kappa**2).real

def cabs(L): 
    return cext(L)-csca(L)

def Qext(L):
    i = L-1
    #incase any of the outer layers are the same as the medium
    while i>(-1):
        if (dielectricArray[:,i]!=m_m).all():
            solid = i #set the solid outer layer index (core = 1)
            break
        i-=1
    if(solid==0):
        return 0
    else:
        return (cext(L))/(sc.pi*(radii[solid]**2))
    
def Qsca(L):
    i = L-1
    #incase any of the outer layers are the same as the medium
    while i>(-1):
        if (dielectricArray[:,i]!=m_m).all():
            solid = i #set the solid outer layer index (core = 1)
            break
        i-=1
    if(solid==0):
        return 0
    else:
        return (csca(L))/(sc.pi*(radii[solid]**2))

def Qabs(L): 
    return Qext(L)-Qsca(L)

def Qnf(l):
    sumd = 0
    for n in np.arange(1, lastTerm+1):
        sumd = sumd + (((np.abs(Al(n,l)))**2)*(((n+1)*(np.abs(SphericalHankelH2(n-1,rA(l,l-1))))**2)+(n*(np.abs(SphericalHankelH2(n+1,rA(l,l-1))))**2)))+(((2*n)+1)*((np.abs(Bl(n,l)))**2)*(np.abs(SphericalHankelH2(n,rA(l,l-1))))**2)
    return 0.5*sumd

file = open(outputFile, "w+")
qnf_header = "Qnf_1"
if(numLayers > 1):
    for i in range(2,numLayers + 1):
        qnf_header = "\t".join([qnf_header,"Qnf_" + str(i)])
header = "\t".join(["Lambda","Qext","Qsca","Qabs","Cext","Csca","Cabs", qnf_header])

file.write(header)
file.write("\n")

list1 = lamda.real
list2 = Qext(numLayers).real
list3 = Qsca(numLayers).real
list4 = Qabs(numLayers).real
list5 = cext(numLayers).real
list6 = csca(numLayers).real
list7 = cabs(numLayers).real
Qnfl = np.array(Qnf(2)).reshape(len(Qnf(2)),1)
if(numLayers > 1):
    for i in range(3,numLayers + 2):
        Qnfl = np.hstack((Qnfl, np.array(Qnf(i)).reshape(len(Qnf(i)),1)))
    
for i in range(0,len(lamda)):
    QnfText = str(Qnfl[i][0])
    for j in range(1, Qnfl.shape[1]):
        QnfText = "\t".join([QnfText,str(Qnfl[i][j])])
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