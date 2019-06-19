# use single direction and N tiem segments to solve Lorentz problem
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import timeit
from scipy.sparse import csr_matrix
from parameters import *

startTime = timeit.timeit()


def pushSeg(u0, vstar0, w0):

    Df = np.zeros([nstep, nc, nc])
    dfdrho = np.zeros([nstep, nc])
    dJdu = np.zeros([nstep, nc])
    u = np.zeros([nstep, nc])
    vstar = np.zeros([nstep, nc])
    w = np.zeros([nstep, nus, nc])
   
    # assign initial value, u[0,0], v*[0,0], w[0,0]
    u[0] = u0
    vstar[-1] = vstar0
    w[-1] = w0

    # find u, Df, dfdrho
    for i in range(1, nstep):
        # push forward u to i-th step
        [x, y ,z] = u[iseg, i - 1]
        dudt = np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
        u[iseg, i] = u[iseg, i - 1] + dudt * dt
        # get Df and dfdrho todo: maybe bug in overlap timestep
        Df[iseg, i - 1] = np.array([[-sigma, sigma, 0],[rho - z,-1,-x],[y,x,-beta]])
        dfdrho[i - 1] = np.array([0, x * dt, 0])
        dJdu[i - 1] = np.array([0, 0, 1])
        
    # find w and v*
    for i in range(nstep-2,-1,-1):
        for ius in range(0, nus):
            w[i, ius] += -np.dot(Df[i].T, w[i + 1, ius]) * dt
        vstar[i] += -np.dot(Df[i].T, vstar[i + 1]) + dJdu[i]

    return [u, w, vstar]

J_arr = np.zeros(Nrho)
dJdrho_arr = np.zeros(Nrho)
rho_arr = np.zeros(Nrho)

for rr in range(rho_lb, rho_ub + 1):
    print(rr)

    #  get u0, vstar0, w0 for pre-smoothing
    u0 =  [-10, -10, 60]
    vstar0 = [0, 0, 0]
    w0 = np.zeros([nus,nc])
    for ius in range(0,nus):
        w0[ius] = np.random.rand(nc)
        w0[ius] = w0[ius] / np.linalg.norm(w0[ius])
    # push forward u to a stable attractor
    
    [u_ps, w_ps, vstar_ps] = pushSeg(nseg_ps, nstep, nus, nc, dt,u0, vstar0, w0)
    u0 = u_ps[-1,0]

    # find u, w, vstar on all segments
    [u, w, vstar] = pushSeg(nseg, nstep, nus, nc, dt,u0, vstar0, w0)
    # construct M and rhs
    M = np.zeros([nus, nus])
    rhs = np.zeros(nus)
    for ius in range(0, nus):
        for jus in range(0, nus):
            M[ius, jus] = np.dot(w[:, ius], w[6:,jus])# + eps
        # rhs 1
        rhs[ii] = np.dot(vstar[iseg + 1,0] - vstar[iseg,-1], w[iseg + 1,0,ius])

    plt.spy(M)
    plt.show()
    lbd = np.linalg.solve(M, rhs)

    # calculate v
    v = vstar
    for ius in range(0, nus):
        v+= lbd[ius] * w[:, ius,:]

    # window function
    def window(eta):
        w = 2*(1 - np.cos(np.pi * eta) ** 2)  # actually it works good for w = 2*(1 - np.cos(2*np.pi * eta) ** 2) 
        return w
    # calculate rho and dJ/drho
    rho_arr[rr - rho_lb] = rr
    J_arr[rr - rho_lb] = np.einsum(u[:,:,2],[0,1],[]) / (nstep * nseg)
    t = np.zeros([nseg*(nstep-1)+1])
    # reshape v to [nseg*(nstep-1), nc] vector: delete duplicate
    v_resu = np.zeros([nseg*(nstep-1)+1,nc])
    for iseg in range(0, nseg):
        for istep in range(0, nstep-1):
            ii = iseg*(nstep-1) + istep
            v_resu[ii] = v[iseg, istep]
            t[ii] = ii * dt
    v_resu[-1] = v[-1,-1]
    t[-1] = nseg*(nstep-1) * dt
    w = window(t / t[-1])
    dJdrho_arr[rr - rho_lb] = np.einsum(v_resu[:,2],[0],w,[0],[]) / (nseg*(nstep-1)+1)
        

    ## plot some debug info
    #plt.plot(np.linalg.norm(v_resu, axis=1))
    #plt.show()
    #plt.plot(w)
    #plt.show()

    djdrho_4deviation[i_trajec] = dJdrho_arr[rr - rho_lb]


## plot u
#mpl.rcParams['legend.fontsize'] = 10
#fig = plt.figure()
#ax = fig.gca(projection = '3d')
#ax.plot(u[:,0],u[:,1],u[:,2])
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z')
#plt.show()

plt.rc('text', usetex=True)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 24}
plt.rc('font', **font)

## plot J vs r
#plt.subplot(2,1,1)
#plt.plot(rho_arr, J_arr)
#plt.ylabel(r'$\langle J \rangle$')

## plot dJdrho vs r
#plt.subplot(2,1,2)
#plt.plot(rho_arr, dJdrho_arr)
#plt.xlabel(r'$\rho$')
#plt.ylabel(r'$d \langle J \rangle / d \rho$')
#plt.ylim([0,2.0])
#plt.savefig('withDilation_T2500.png')
#plt.show()

# plot growrate vs r
#plt.subplot(2,1,1)
#plt.plot(rho_arr,grow_rate_arr)
#plt.xlabel(r'\rho')
#plt.ylabel('growRate')
#plt.show()

# plot deviation
plt.loglog(T_total_array, deviation, linestyle='None', marker='.', markersize=10)
plt.xlabel(r'$T$')
plt.ylabel(r'std($d \langle J \rangle / d \rho$)')

plt.show()


endTime = timeit.timeit()
print endTime - startTime
print('end')