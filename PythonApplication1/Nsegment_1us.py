# use single direction and N tiem segments to solve Lorentz problem
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

rho_lb = 30
rho_ub = 30
Nrho = rho_ub - rho_lb + 1 # number of rho to be calculated
J_arr = np.zeros(Nrho)
dJdrho_arr = np.zeros(Nrho)
rho_arr = np.zeros(Nrho)

for rr in range(rho_lb, rho_ub + 1):

    print(rr)
    rho = rr
    sigma = 10
    beta = 8. / 3.
    
    nseg = 5 # number of segments on time interval
    T = 2 # length of each segment
    dt = 0.005
    nc = 3 # number of component in u
    nus = 2 # number of unstable direction
    nstep = int(T / dt) # number of step in each time segment
    Df = np.zeros([nseg, nstep, nc, nc])
    dfdrho = np.zeros([nseg, nstep, nc]) #
    u = np.zeros([nseg, nstep, nc])
    v = np.zeros([nseg, nstep, nc])
    vstar = np.zeros([nseg, nstep, nc])
    w = np.zeros([nseg, nstep, nc])

    # push forward u to a stable attractor
    u_temp = np.zeros([nstep, nc])
    u_temp[0] = [-10, -10, 60]
    for j in range(0,5):
        for i in range(1, nstep):    
            [x, y ,z] = u_temp[i - 1]
            dudt = np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
            u_temp[i] = u_temp[i - 1] + dudt * dt
        u0 = u_temp[-1]
        u_temp = np.zeros([nstep,3])
        u_temp[0] = u0
    
    # assign u[0,0], v*[0,0], w[0,0]
    u[0,0] = u0
    vstar[0,0] = [0, 0, 0]
    w[0,0] = [1, 0, 0]
    #w[0,0,1] = [0, 1, 0]

    # find u, w and vstar on each segment
    for iseg in range(0, nseg):
        
        # find u, Df, dfdrho
        for i in range(1, nstep):
            # push forward u to i-th step
            [x, y ,z] = u[iseg, i - 1]
            dudt = np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
            u[iseg, i] = u[iseg, i - 1] + dudt * dt
            # get Df and dfdrho todo: maybe bug in overlap timestep
            Dftemp = np.array([[-sigma, sigma, 0],[rho - z,-1,-x],[y,x,-beta]])
            Df[iseg, i - 1] = Dftemp * dt + np.identity(3)
            dfdrho[iseg, i - 1] = np.array([0, x * dt, 0])
        
        # find w and v*
        for i in range(1, nstep):    
            w[iseg, i] = np.dot(Df[iseg,i - 1], w[iseg, i - 1])
            vstar[iseg,i] = np.dot(Df[iseg,i - 1], vstar[iseg,i - 1]) + dfdrho[iseg,i - 1]

        # get u, and renormalize v* and w for next segment
        if iseg < nseg - 1:
            u[iseg + 1, 0] = u[iseg, -1]
            

            w[iseg + 1, 0] = w[iseg, -1] / np.linalg.norm(w[iseg,-1])
            
            vstar[iseg + 1,0] = vstar[iseg,-1] 
            vstar[iseg + 1,0] -= \
                    np.dot(vstar[iseg,-1], w[iseg,-1]) \
                    / np.dot(w[iseg,-1], w[iseg,-1]) \
                    * w[iseg,-1]

    # calculate lbd
    M = np.zeros([(2 * nseg - 1), (2 * nseg - 1)])
    rhs = np.zeros((2 * nseg - 1))
    # dL/dmu = 0: nus*(N-1) equations
    for i in range(0, nseg - 1):
        # Diagonal 1
        M[i,i] = np.dot(w[i,-1], w[i + 1,0])
        # Diagonal 2
        M[i,i + 1] = - np.dot(w[i + 1,0], w[i + 1,0])
        # rhs 1
        rhs[i] = np.dot(vstar[i + 1,0] - vstar[i,-1], w[i + 1,0])
    # dL/dlbd = 0: next N equations
    for i in range(0, nseg):
        # Diagonal 3
        M[i + nseg - 1, i] = 2 * np.einsum(w[i],[0,1],w[i],[0,1])
        # Diagonal 4
        if i < nseg - 1:
            M[i + nseg - 1, i + nseg] = -np.dot(w[i,-1], w[i + 1,0])
        # Diagonal 5
        if i > 0:
            M[i + nseg - 1, i + nseg - 1] = np.dot(w[i,0],w[i,0])
        # rhs 2
        rhs[i + nseg - 1] = -2 * np.einsum(vstar[i],[0,1], w[i],[0,1])
    #plt.spy(M)
    #plt.show
    lbd = np.linalg.solve(M, rhs)
    lbd = lbd[:nseg]

    # calculate v
    for iseg in range(0, nseg):
        v[iseg] = vstar[iseg] + lbd[iseg] * w[iseg]
#    v = vstar + np.einsum(lbd,[0], w, [0,1,2], [1,2])
       

    # calculate rho and dJ/drho
    rho_arr[rr - rho_lb] = rr
    J_arr[rr - rho_lb] = np.einsum(u[:,:,2],[0,1],[]) / (nstep * nseg)
    dJdrho_arr[rr - rho_lb] = np.einsum(v[:,:,2],[0,1],[]) / (nstep * nseg)


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
plt.subplot(2,1,1)
plt.plot(rho_arr, J_arr)
plt.xlabel('rho')
plt.ylabel('J')

# plot dJdrho vs r
plt.subplot(2,1,2)
plt.plot(rho_arr, dJdrho_arr)
plt.xlabel('rho')
plt.ylabel('dJdrho')
plt.ylim([0,1.5])
plt.show()

pass