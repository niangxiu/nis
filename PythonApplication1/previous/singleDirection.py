# use only single direction projection to solve the Lorentz problem.
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

rho_lb = 62
rho_ub = 64
Nrho = rho_ub - rho_lb + 1
navg = 2 # number of segments on time interval 
J_arr = np.zeros([Nrho, navg])
dJdrho_arr = np.zeros([Nrho, navg])
J_arr_avg = np.zeros(Nrho)
dJdrho_arr_avg = np.zeros(Nrho)
rho_arr = np.zeros(Nrho)

for rr in range(rho_lb, rho_ub+1):
    print(rr)
    rho = rr
    sigma = 10
    beta = 8./3.
    T = 2
    nT = 10
    dt = 0.005
    nc =3 # number of component in u
    nUnstable = 1 
    nn = int(T/dt/2)
    N = 2*nn + 1
    Df = np.zeros([N,3,3])
    drhof = np.zeros([N,3])
    u = np.zeros([N,3])
    v = np.zeros([N,3])
    vstar = np.zeros([N,3])
    w = np.zeros([nUnstable,N,nc])

    # give start value, u at t=0; v*,w at t = nn
    u[0] = [-10, -10, 60]

    # push forward u to a stable attractor
    for j in range(0,5):
        for i in range(1, 2*nn+1):    
            [x, y ,z] = u[i-1]
            dudt = np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])
            u[i] = u[i-1] + dudt*dt
        u0 = u[-1]
        u = np.zeros([N,3])
        u[0] = u0
    pass

    for iavg in range(0, navg):

        for i in range(1, 2*nn+1):
            # push forward u to i-th step
            [x, y ,z] = u[i-1]
            dudt = np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])
            u[i] = u[i-1] + dudt*dt
            # get Df and drhof
            Dftemp = np.array([[-sigma, sigma, 0],[rho-z,-1,-x],[y,x,-beta]])
            Df[i-1] = Dftemp * dt + np.identity(3)
            drhof[i-1] = np.array([0, x*dt, 0])


        # intialize v* and w
        vstar[0] = [0, 0, 0]
        w[0,0] = [0, 0, 1]

        for iT in range(0, nT):
            # find w
            for i in range(1, 2*nn+1):
                w[0,i] = np.dot(Df[i-1], w[0,i-1])

            # find vstar
            for i in range(1, 2*nn+1):
                vstar[i] = np.dot(Df[i-1], vstar[i-1]) + drhof[i-1]

            # calculate M, rhs, lbd, and v
            M = np.zeros([nUnstable, nUnstable])
            for i in range(0, nUnstable):
                for j in range(0, i+1):
                    M[i,j] = np.einsum(w[i],[0,1],w[j],[0,1])
                    M[j,i] = M[i,j]

        rhs = np.zeros(nUnstable)
        rhs[0] = np.einsum(w[0], [0,1], vstar, [0,1])
        
        lbd = np.linalg.solve(M, rhs)
        v = vstar - np.einsum(lbd, [0], w, [0,1,2], [1,2])
        vstar[0] = v[0]

        # calculate rho and dJ/drho
        J_arr[rr-rho_lb,iavg] = np.sum(u[:,2]) / N
        dJdrho = np.sum(v[:,2]) / N
        rho_arr[rr-rho_lb] = rho
        dJdrho_arr[rr-rho_lb,iavg] = dJdrho

        # use the end state as initial state of next segment
        u0 = u[-1]
        u = np.zeros([N,3])
        u[0] = u0

    J_arr_avg = np.average(J_arr,axis=1)
    dJdrho_arr_avg = np.average(dJdrho_arr,axis=1)
    pass

## plot u
#mpl.rcParams['legend.fontsize'] = 10
#fig = plt.figure()
#ax = fig.gca(projection = '3d')
#ax.plot(u[:,0],u[:,1],u[:,2])
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z')
#plt.show()

# plot J vs r
fig = plt.figure()
ax = fig.gca()
ax.plot(rho_arr, J_arr_avg)
ax.set_xlabel('rho')
ax.set_ylabel('J')
plt.show()

# plot dJdrho vs r
fig = plt.figure()
plt.plot(rho_arr, dJdrho_arr_avg)
plt.xlabel('rho')
plt.ylabel('dJdrho')
plt.ylim([0,1.5])

plt.show()
pass

pass