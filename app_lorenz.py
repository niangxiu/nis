# use single direction and N tiem segments to solve Lorentz problem
import numpy as np
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nilss import *


rho_lb = 20
rho_ub = 26
sigma = 10
beta = 8. / 3.
rho_arr = np.arange(rho_lb, rho_ub + 1)
J_arr = np.zeros(rho_arr.shape)
dJdrho_arr = np.zeros(rho_arr.shape)


def ddt(uwvs, rho):
    u = uwvs[0]
    x, y, z = u
    w = uwvs[1]
    vstar = uwvs[2]
    dudt = np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
    Df = np.array([[-sigma, sigma, 0],[rho - z,-1,-x],[y,x,-beta]])
    dwdt = np.dot(Df, w.T)
    dfdrho = np.array([0, x, 0])
    dvstardt = np.dot(Df, vstar) + dfdrho
    return np.array([dudt, dwdt.T, dvstardt])



# parameter passing to nilss
dt = 0.001
nseg = 10 #number of segments on time interval
T_seg = 2 # length of each segment
T_ps = 10 # time of pre-smoothing
nc = 3 # number of component in u
nus = 1 # number of unstable direction


# functions passing to nilss
def fJJu(u, rho):
    x, y, z = u
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z]),\
            z, np.array([0,0,1])


def RK4(u, w, vstar, rho):
    # integrate u, w, and vstar to the next time step
    uwvs = np.array([u, w, vstar])
    k0 = dt * ddt(uwvs, rho) 
    k1 = dt * ddt(uwvs + 0.5 * k0, rho)
    k2 = dt * ddt(uwvs + 0.5 * k1, rho)
    k3 = dt * ddt(uwvs + k2, rho)
    uwvs_new = uwvs + (k0 + 2*k1 + 2*k2 + k3) / 6.0
    return uwvs_new


def Euler(u, w, vstar, rho):
    uwvs = np.array([u, w, vstar])
    k0 = dt * ddt(uwvs, rho) 
    uwvs_new = uwvs + k0 
    return uwvs_new


# main loop
for i, rho in enumerate(rho_arr):
    print(rho)
    u0 =  (np.random.rand(nc)-0.5) * 100 + np.array([0,0,50]) #[-10, -10, 60]
    J, dJdrho = nilss(dt, nseg, T_seg, T_ps, u0, nus, rho, Euler, fJJu)
    J_arr[i] = J
    dJdrho_arr[i] = dJdrho


# plot preparations
plt.rc('axes', labelsize='xx-large',  labelpad=12)
plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')
plt.rc('legend', fontsize='xx-large')
plt.rc('font', family='sans-serif')


# plot J vs r
plt.figure(figsize=[12,12])
plt.subplot(2,1,1)
plt.plot(rho_arr, J_arr)
plt.xlabel(r'$\rho$')
plt.ylabel(r'$\langle J \rangle$')


# plot dJdrho vs r
plt.subplot(2,1,2)
plt.plot(rho_arr, dJdrho_arr)
plt.xlabel(r'$\rho$')
plt.ylabel(r'$d \langle J \rangle / d \rho$')
plt.ylim([0,2.0])
plt.savefig('lorenz.png')
plt.show()
