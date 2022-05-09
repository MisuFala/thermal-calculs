import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import multiprocess as mp

def init(n, LargeurResistance, HauteurResistance, T0, LargeurCube, Tinf):
    delta = LargeurCube/n
    Xmin = round((n - LargeurResistance/delta)/2)
    Xmax = round((n + LargeurResistance/delta)/2)
    Zmin = round((n - HauteurResistance/delta)/2)
    Zmax = round((n + HauteurResistance/delta)/2)
    T = np.ones((n+1, n+1))
    T=T*Tinf
    for i in range(Xmin, Xmax+1):
        for j in range(Zmin, Zmax+1):
            T[i,j]=T0
    return (T, delta, Xmin, Xmax, Zmin, Zmax)

def calculnorm(T, i, j):
    return (T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1])/4

def calculgauche(T, j):
    return T[1,j]/2 + (T[0,j+1] + T[0,j-1])/4

def calculdroite(T, j, n):
    return T[n-1,j]/2 + (T[n,j+1] + T[n,j-1])/4

def calculbas(T, i):
    return T[i,1]/2 + (T[i+1,0] + T[i-1,0])/4

def calcul00(T):
    return (T[1,0] + T[0,1])/2

def calculn0(T, n):
    return (T[n-1,0] + T[n,1])/2

def calcul0n(T, h, delta, n, L, Tinf):
    return ((T[1,n] + T[0,n-1])/(2+h*delta/(2*L)) + Tinf*h*delta/(4*L+h*delta))

def calculnn(T, h, delta, n, L, Tinf):
    return ((T[n-1,n] + T[n,n-1])/(2+h*delta/(2*L)) + Tinf*h*delta/(4*L+h*delta))

def calculin(T, h, delta, n, L, Tinf, i):
    return ((T[i+1,n] + T[i-1,n])/(4*(1+h*delta/(2*L))) + T[i,n-1]/(2*(1+h*delta/(2*L))) + Tinf*h*delta/(2*(L+h*delta/2)))

def calculTotal(T, h, delta, n, L, Tinf, Xmin, Xmax, Zmin, Zmax):
    for i in range(0, n+1):
        for j in range(0, n+1):
            if i<=Xmax and i>=Xmin and j<=Zmax and j>=Zmin:
                T[i, j]=T[i, j]
            elif j==0:
                if i==0:
                    T[i, j]=calcul00(T)
                elif i==n:
                    T[i, j]=calculn0(T,n)
                else:
                    T[i, j]=calculbas(T,i)
            elif j==n:
                if i==0:
                    T[i, j]=calcul0n(T, h, delta, n, L, Tinf)
                elif i==n:
                    T[i, j]=calculnn(T, h, delta, n, L, Tinf)
                else:
                    T[i,j]=calculin(T, h, delta, n, L, Tinf, i)
            else:
                if i==0:
                    T[i,j]=calculgauche(T, j)
                elif i==n:
                    T[i,j]=calculdroite(T, j, n)
                else:
                    T[i,j]=calculnorm(T, i, j)
    return T

def iteration(n, LargeurResistance, HauteurResistance, T0, LargeurCube, Tinf, h, L, nbIter):
    T, delta, Xmin, Xmax, Zmin, Zmax=init(n, LargeurResistance, HauteurResistance, T0, LargeurCube, Tinf)
    if __name__ == '__main__':
        pool=mp.Pool(processes=4)
        n=100
        with concurrent.futures.ProcessPoolExecutor() as executor:
            T = executor.map(calculTotal, (T, h, delta, n, L, Tinf, Xmin, Xmax, Zmin, Zmax))
        print(T)
    #for u in range(nbIter):
        #T=calculTotal(T, h, delta, n, L, Tinf, Xmin, Xmax, Zmin, Zmax)
    return T

n=50
T=iteration(n, 1, 1, 150, 10, 18, 20, 0.6, 1000)
print(T)
T=np.rot90(T,3)


##
XB = np.linspace(0,n,n+1)
YB = np.linspace(0, n, n+1)
X,Y = np.meshgrid(XB, YB)
cmap1=plt.get_cmap('turbo', lut=13)
plt.imshow(T, cmap=cmap1, origin='lower', vmin=T.min(), vmax=T.max())
plt.colorbar()
plt.contour(T, levels=13, colors='black', origin='lower')
plt.show()
plt.close()

##
import multiprocess as mp
import concurrent.futures
if __name__ == '__main__':
    pool=mp.Pool(processes=4)
    n=100
    with concurrent.futures.ProcessPoolExecutor() as executor:
        T = executor.map(iteration, (n, 1, 1, 150, 10, 18, 20, 0.6, 1000))
    print(T)




