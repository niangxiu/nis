# use single direction and N tiem segments to solve Lorentz problem
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import timeit
#from scipy.sparse import csr_matrix
from parameters import *

startTime = timeit.timeit()

# window function
def window(eta):
    w = 2*(1 - np.cos(np.pi * eta) ** 2) 
    return w

# For lorentz problem: find u, w and vstar on each segment  
def pushSeg(nseg, nstep, nus, nc, dt, u0, vstar0, w0):
    rho = rr
    sigma = 10
    beta = 8. / 3.

    Df = np.zeros([nseg, nstep, nc, nc])
    dfdrho = np.zeros([nseg, nstep, nc])
    u = np.zeros([nseg, nstep, nc])
    vstar = np.zeros([nseg, nstep, nc])
    w = np.zeros([nseg, nstep, nus, nc])
   
    # assign initial value, u[0,0], v*[0,0], w[0,0]
    u[0,0] = u0
    vstar[0,0] = vstar0
    w[0,0] = w0

    # push forward
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
            for ius in range(0, nus):
                w[iseg, i, ius] = np.dot(Df[iseg,i - 1], w[iseg, i - 1, ius])
            vstar[iseg,i] = np.dot(Df[iseg,i - 1], vstar[iseg,i - 1]) + dfdrho[iseg,i - 1]

        # get u, and renormalize v* and w for next segment
        if iseg < nseg - 1:
            u[iseg + 1, 0] = u[iseg, -1]
            
            [Qtemp, Rtemp] = (np.linalg.qr(w[iseg,-1].T, 'reduced'))
            w[iseg + 1, 0] = Qtemp.T
            #w[iseg + 1, 0,0] = w[iseg, -1,0] / np.linalg.norm(w[iseg,-1,0])

            vstar[iseg + 1,0] = vstar[iseg,-1] 
            for ius in range (0, nus):
                vstar[iseg + 1,0] += \
                    - np.dot(vstar[iseg,-1], w[iseg+1,0,ius]) \
                    / np.dot(w[iseg+1,0,ius], w[iseg+1,0,ius]) \
                    * w[iseg+1,0,ius]
    return [u, w, vstar]



J_arr = np.zeros(Nrho)
dJdrho_arr = np.zeros(Nrho)
rho_arr = np.zeros(Nrho)

for rr in range(rho_lb, rho_ub + 1):
    print(rr)
    T_total = 100
#T_total_array = [10]    # 10, 20, 50, 100, 200, 500,
#deviation = np.zeros(len(T_total_array)) # standard deviation for different T_total
#for i_T in range(0,len(T_total_array)):
#    T_total = T_total_array[i_T]
#    djdrho_4deviation = np.zeros(100)

    #for i_trajec in range(0,100):
    #    print(i_T, i_trajec)
    #    rr = 28

    v = np.zeros([nseg, nstep, nc])
    
    #  get u0, vstar0, w0 for pre-smoothing
    u0 =  [-10, -10, 60]
    #u0[0] =  (np.random.rand()-0.5) * 40 
    #u0[1] =  (np.random.rand()-0.5) * 40 
    #u0[2] =  (np.random.rand()-0.5) * 40 + 20
    vstar0 = [0, 0, 0]
    w0 = np.zeros([nus,nc])
    for ius in range(0,nus):
        w0[ius] = np.random.rand(nc)
        w0[ius] = w0[ius] / np.linalg.norm(w0[ius])
    # push forward u to a stable attractor
    nseg_ps = int(T_ps/dt/nstep)
    [u_ps, w_ps, vstar_ps] = pushSeg(nseg_ps, nstep, nus, nc, dt,u0, vstar0, w0)
    # get initial value for later integration
    u0 = u_ps[-1,0]
    vstar0 = vstar_ps[-1,0]
    w0 = w_ps[-1,0]

    # find u, w, vstar on all segments
    [u, w, vstar] = pushSeg(nseg, nstep, nus, nc, dt,u0, vstar0, w0)

    # construct M
    M = np.zeros([(2 * nseg - 1)*nus, (2 * nseg - 1)*nus])
    rhs = np.zeros((2 * nseg - 1)*nus)
    # dL/dmu = 0: nus*(N-1) equations
    for iseg in range(0, nseg - 1):
        for ius in range(0, nus):
            ii0 = iseg * nus # the starting idex of iseg
            ii1 = (iseg+1) * nus # the startind index of iseg+1
            ii  = ii0 + ius # the number of equation we are dealing with
            # Diagonal 1 and 2
            for k in range(0, nus):
                M[ii, ii0 + k] = np.dot(w[iseg,-1,k], w[iseg + 1,0,ius])# + eps
                M[ii, ii1 + k] = - np.dot(w[iseg + 1,0,k], w[iseg + 1,0,ius]) #+ eps
            # rhs 1
            rhs[ii] = np.dot(vstar[iseg + 1,0] - vstar[iseg,-1], w[iseg + 1,0,ius])
    
    # dL/dlbd = 0: next nus*N equations
    for iseg in range(0, nseg):
        for ius in range(0, nus):
            ii0 = iseg * nus
            ii1 = (iseg+1)*nus
            ii = ii0 + ius + (nseg-1)*nus
            jj0 = iseg * nus + (nseg)*nus
            jjm1 = (iseg-1) * nus + (nseg)*nus
            # Diagonal 3
            for k in range (0,nus):
                M[ii, ii0 + k] = 2 * np.einsum(w[iseg,:, k],[0,1],w[iseg,:, ius],[0,1])
            # Diagonal 4
            if iseg < nseg - 1:
                for k in range (0, nus):
                    M[ii, jj0 + k] = -np.dot(w[iseg,-1, ius], w[iseg + 1,0,k]) #+eps
            # Diagonal 5
            if iseg > 0:
                for k in range (0, nus):
                    M[ii, jjm1 +k] = np.dot(w[iseg,0,ius],w[iseg,0,k]) #+ eps
            # rhs 2
            rhs[ii] = -2 * np.einsum(vstar[iseg],[0,1], w[iseg,:,ius],[0,1])

    #plt.spy(M)
    #plt.show()

    #M = csr_matrix(M)
    lbd = np.linalg.solve(M, rhs)
    lbd = lbd[:nseg*nus]
    lbd = lbd.reshape([nseg, nus])

    # calculate v
    for iseg in range(0, nseg):
        v[iseg] = vstar[iseg]
        for ius in range(0, nus):
            v[iseg] += lbd[iseg, ius] * w[iseg,:, ius,:]
#    v = vstar + np.einsum(lbd,[0], w, [0,1,2], [1,2])
       

    

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
    win = window(t / t[-1])
    dJdrho_arr[rr - rho_lb] = np.einsum(v_resu[:,2],[0],win,[0],[]) / (nseg*(nstep-1)+1)
    
    # plot setting
    plt.rc('text', usetex=True)
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 36}
    plt.rc('font', **font)

    # plot u
    fig = plt.figure(figsize = (20,20))
    ax = Axes3D(fig)
    u = np.squeeze(u)
    u_resu = np.zeros([nseg*(nstep-1)+1,nc])
    for iseg in range(0, nseg):
        for istep in range(0, nstep-1):
            ii = iseg*(nstep-1) + istep
            u_resu[ii] = u[iseg, istep]
    u_resu[-1] = v[-1,-1]
    u_resu = u_resu.T
    ax.plot(u_resu[0],u_resu[1],u_resu[2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig('lorenzU3d')
    plt.close(fig)

    # plot vstar
    # reshape vstar to [nseg*(nstep-1), nc] vector: delete duplicate
    plt.rc('text', usetex=True)
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 32}
    plt.rc('font', **font)
    vstar_ = np.zeros([nseg*(nstep-1)+1,nc])
    w1_ = np.zeros([nseg*(nstep-1)+1,nc])
    for iseg in range(0, nseg):
        for istep in range(0, nstep-1):
            ii = iseg*(nstep-1) + istep
            vstar_[ii] = vstar[iseg, istep]
            w1_[ii] = w[iseg, istep, 1]
    vstar_[-1] = v[-1,-1]
    fig = plt.figure(figsize = (15,10))
    ax = fig.add_subplot(1,1,1)
    ax.semilogy(t,vstar_[:,2])
    ax.minorticks_off()
    ax.set_yticks([10**0,10**10,10**20,10**30,10**40])   
    plt.savefig('LorenzInstantDerivative.png')
    plt.close(fig)

    # plot v
    fig = plt.figure(figsize = (20,10))
    ax = fig.add_subplot(111)
    ax.plot(t, v_resu[:,2])
    plt.savefig('LorenzV')
    plt.close(fig)    

    ## plot some debug info
    #plt.plot(np.linalg.norm(v_resu, axis=1))
    #plt.show()
    #plt.plot(w)
    #plt.show()

    #djdrho_4deviation[i_trajec] = dJdrho_arr[rr - rho_lb]
    # deviation[i_T] = np.std(djdrho_4deviation)




plt.rc('text', usetex=True)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 32}
plt.rc('font', **font)

# plot J vs r
#plt.subplot(2,1,1)
fig = plt.figure(figsize=(15,10))
plt.plot(rho_arr, J_arr)
#plt.ylabel(r'$\overline{ J }$')
plt.savefig('J.png')
plt.close(fig)

# plot dJdrho vs r
#plt.subplot(2,1,2)
fig = plt.figure(figsize=(15,10))
plt.plot(rho_arr, dJdrho_arr)
#plt.xlabel(r'$\rho$')
#plt.ylabel(r'$d \overline{J} / d \rho$')
        plt.ylim([0,2.0])
        plt.savefig('dJdrho.png')
        plt.close(fig)

    # plot growrate vs r
    #plt.subplot(2,1,1)
    #plt.plot(rho_arr,grow_rate_arr)
#plt.xlabel(r'\rho')
#plt.ylabel('growRate')
#plt.show()

## plot deviation
#plt.loglog(T_total_array, deviation, linestyle='None', marker='.', markersize=10)
#plt.xlabel(r'$T$')
#plt.ylabel(r'std($d \langle J \rangle / d \rho$)')

#plt.show()


#endTime = timeit.timeit()
#print endTime - startTime
#print('end')
