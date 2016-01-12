# use single direction and N tiem segments to solve Lorentz problem
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

def ddt(uwvs):
    u = uwvs[0]
    [x, y, z] = u
    w = uwvs[1]
    vstar = uwvs[2]
    dudt = np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
    Df = np.array([[-sigma, sigma, 0],[rho - z,-1,-x],[y,x,-beta]])
    dwdt = np.dot(Df, w.T)
    dfdrho = np.array([0, x, 0])
    dvstardt = np.dot(Df, vstar) + dfdrho
    return np.array([dudt, dwdt.T, dvstardt])

def RK4(u, w, vstar):
    # integrate u, w, and vstar to the next time step
    #uwvs = np.array([u, w, vstar])
    #k0 = dt * ddt(uwvs) 
    #k1 = dt * ddt(uwvs + 0.5 * k0)
    #k2 = dt * ddt(uwvs + 0.5 * k1)
    #k3 = dt * ddt(uwvs + k2)
    #uwvs_new = uwvs + (k0 + 2*k1 + 2*k2 + k3) / 6.0
    uwvs = np.array([u, w, vstar])
    k0 = dt * ddt(uwvs) 
    uwvs_new = uwvs + k0 
    return uwvs_new
    

def pushSeg(nseg, nstep, nus, nc, dt, u0, vstar0, w0):
    # For lorentz problem
    # find u, w and vstar on each segment

    f = np.zeros([nseg, nstep, nc])
    u = np.zeros([nseg, nstep, nc])
    vstar = np.zeros([nseg, nstep, nc])
    vstar_perp = np.zeros_like(vstar)
    w = np.zeros([nseg, nstep, nus, nc])
    w_perp = np.zeros_like(w)
   
    # assign initial value, u[0,0], v*[0,0], w[0,0]
    u[0,0] = u0
    vstar[0,0] = vstar0
    w[0,0] = w0


    # push forward
    for iseg in range(0, nseg):
        for istep in range(0, nstep-1):
            u[iseg, istep+1], w[iseg, istep+1], vstar[iseg,istep+1]\
                = RK4(u[iseg, istep], w[iseg, istep], vstar[iseg, istep])
        for istep in range(0, nstep):
            [x, y, z] = u[iseg, istep]
            f[iseg, istep] = [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

        # get u, and renormalize v* and w for next segment
        if iseg < nseg - 1:
            u[iseg + 1, 0] = u[iseg, -1]
            
            [Qtemp, Rtemp] = (np.linalg.qr(w[iseg,-1].T, 'reduced'))
            w[iseg + 1, 0] = Qtemp.T
            #w[iseg + 1, 0,0] = w[iseg, -1,0] / np.linalg.norm(w[iseg,-1,0])

            vstar[iseg + 1,0] = vstar[iseg,-1] 
            for ius in range(0, nus):
                vstar[iseg + 1,0] += \
                    - np.dot(vstar[iseg,-1], w[iseg + 1,0,ius]) \
                    / np.dot(w[iseg + 1,0,ius], w[iseg + 1,0,ius]) \
                    * w[iseg + 1,0,ius]

    # calculate vstar_perp and w_perp
    for iseg in range(0, nseg):
        for i in range(0, nstep):
            vstar_perp[iseg, i] = vstar[iseg, i] - np.dot(vstar[iseg, i], f[iseg, i]) / np.dot(f[iseg, i], f[iseg, i]) * f[iseg, i]
            for ius in range(0, nus):
                w_perp[iseg, i, ius] = w[iseg, i,ius] - np.dot(w[iseg, i, ius],     f[iseg, i]) / np.dot(f[iseg, i], f[iseg, i]) * f[iseg, i]

    return [u, w, vstar, w_perp, vstar_perp, f]



rho_lb = 45
rho_ub = 76
nseg = 40 #number of segments on time interval
T_seg = 3 # length of each segment
rho = 0 # make a global variable
sigma = 10
beta = 8. / 3.
Nrho = rho_ub - rho_lb + 1 # number of rho to be calculated
J_arr = np.zeros(Nrho)
dJdrho_arr = np.zeros(Nrho)
rho_arr = np.zeros(Nrho)
grow_rate_arr = np.zeros(Nrho) # the grow rate (10^*) in a time unit for w


for rho in range(rho_lb, rho_ub + 1):

    print(rho)
    
    T_ps = 10 # time of pre-smoothing
    dt = 0.01
    nc = 3 # number of component in u
    nus = 2 # number of unstable direction
    nstep = int(T_seg / dt) # number of step in each time segment
    v = np.zeros([nseg, nstep, nc])
    v_perp = np.zeros_like(v)
    ksi = np.zeros([nseg, nstep])
    
    #  get u0, vstar0, w0 for pre-smoothing
    u0 = [-10.0, -10.0, 60.0]
    vstar0 = [0.0, 0.0, 0.0]
    w0 = np.zeros([nus,nc])
    for ius in range(0,nus):
        w0[ius] = np.random.rand(nc)
        w0[ius] = w0[ius] / np.linalg.norm(w0[ius])
    # push forward u to a stable attractor
    nseg_ps = int(T_ps / dt / nstep)
    [u_ps, w_ps, vstar_ps, temp1, temp2, temp3] = pushSeg(nseg_ps, nstep, nus, nc, dt,u0, vstar0, w0)
    # get initial value for later integration
    u0 = u_ps[-1,0]
    vstar0 = vstar_ps[-1,0]
    w0 = w_ps[-1,0]


    # find u, w, vstar on all segments
    [u, w, vstar, w_perp, vstar_perp, f] = pushSeg(nseg, nstep, nus, nc, dt,u0, vstar0, w0)
    # calculate grow rate of w
    grow_rate_arr[rho - rho_lb] = (np.log(np.linalg.norm(w[-1,-1]) / np.linalg.norm(w[-1,0]))) / T_seg
    # construct M
    M = np.zeros([(2 * nseg - 1) * nus, (2 * nseg - 1) * nus])
    rhs = np.zeros((2 * nseg - 1) * nus)
    # dL/dmu = 0: nus*(N-1) equations
    for iseg in range(0, nseg - 1):
        for ius in range(0, nus):
            ii0 = iseg * nus # the starting idex of iseg
            ii1 = (iseg + 1) * nus # the startind index of iseg+1
            ii = ii0 + ius # the number of equation we are dealing with
            # Diagonal 1 and 2
            for k in range(0, nus):
                M[ii, ii0 + k] = np.dot(w[iseg,-1,k], w[iseg + 1,0,ius])
                M[ii, ii1 + k] = - np.dot(w[iseg + 1,0,k], w[iseg + 1,0,ius])
            # rhs 1
            rhs[ii] = np.dot(vstar[iseg + 1,0] - vstar[iseg,-1], w[iseg + 1,0,ius])
    
    # dL/dlbd = 0: next nus*N equations
    for iseg in range(0, nseg):
        for ius in range(0, nus):
            ii0 = iseg * nus
            ii1 = (iseg + 1) * nus
            ii = ii0 + ius + (nseg - 1) * nus
            jj0 = iseg * nus + (nseg) * nus
            jjm1 = (iseg - 1) * nus + (nseg) * nus
            # Diagonal 3
            for k in range(0,nus):
                M[ii, ii0 + k] = 2 * np.einsum(w_perp[iseg,:, k],[0,1],w_perp[iseg,:, ius],[0,1])
            # Diagonal 46
            if iseg < nseg - 1:
                for k in range(0, nus):
                    M[ii, jj0 + k] = -np.dot(w[iseg,-1, ius], w[iseg + 1,0,k])
            # Diagonal 5
            if iseg > 0:
                for k in range(0, nus):
                    M[ii, jjm1 + k] = np.dot(w[iseg,0,ius],w[iseg,0,k])
            # rhs 2
            rhs[ii] = -2 * np.einsum(vstar_perp[iseg],[0,1], w_perp[iseg,:,ius],[0,1])

    #plt.spy(M)
    #plt.show()

    lbd = np.linalg.solve(M, rhs)
    #lbd,temp = np.linalg.lstsq(M, rhs)
    lbd = lbd[:nseg * nus]
    lbd = lbd.reshape([nseg, nus])

    # calculate v
    for iseg in range(0, nseg):
        v[iseg] = vstar[iseg]
        v_perp[iseg] = vstar_perp[iseg]
        for ius in range(0, nus):
            v[iseg] += lbd[iseg, ius] * w[iseg,:, ius,:]
            v_perp[iseg] += lbd[iseg, ius] * w_perp[iseg,:, ius,:]
    
    # calculate ksi
    for iseg in range(0, nseg):
        for i in range(0, nstep):
            ksi[iseg,i] = np.dot(v[iseg, i], f[iseg, i]) / np.dot(f[iseg, i], f[iseg, i])
       

    # window function
    def window(eta):
        w = 2 * (1 - np.cos(2 * np.pi * eta) ** 2)
        return w
    # calculate rho and dJ/drho
    rho_arr[rho - rho_lb] = rho
    J_arr[rho - rho_lb] = np.einsum(u[:,:,2],[0,1],[]) / (nstep * nseg)
    t = np.zeros([nseg * (nstep - 1) + 1])

    # reshape v, v_perp, f, ksi to [nseg*(nstep-1), nc] vector: delete duplicate
    v_resu = np.zeros([nseg * (nstep - 1) + 1,nc])
    v_resu_perp = np.zeros_like(v_resu)
    f_resu = np.zeros_like(v_resu)
    ksi_resu = np.zeros([nseg * (nstep - 1) + 1])    
    for iseg in range(0, nseg):
        for istep in range(0, nstep - 1):
            ii = iseg * (nstep - 1) + istep
            v_resu[ii] = v[iseg, istep]
            v_resu_perp[ii] = v_perp[iseg, istep]
            f_resu[ii] = f[iseg, istep]
            ksi_resu[ii] = ksi[iseg, istep]
            t[ii] = ii * dt
    v_resu[-1] = v[-1,-1]
    v_resu_perp[-1] = v_perp[-1,-1]
    f_resu[-1] = f[-1,-1]
    ksi_resu[-1] = ksi[-1,-1]

    # apply windowing
    T_total = nseg * (nstep - 1) * dt
    t[-1] = T_total
    
    ## with window
    #wdw = window(t / T_total)    
    #dJdrho_arr[rho - rho_lb] = np.einsum(v_resu[:,2],[0],wdw,[0],[]) / (nseg * (nstep - 1) + 1) \
    #                        - (ksi_resu[-1]*u[-1,-1,2] -  ksi_resu[0]*u[0,0,2]) / T_total \
    #                        + (ksi_resu[-1] - ksi_resu[0]) * np.sum(u[:,:,2]) / (nseg * (nstep - 1) + 1) / T_total

    # with dilation
    dJdrho_arr[rho - rho_lb]= np.sum(v_resu_perp[:,2]) / (nseg * (nstep - 1) + 1) \
                            - (ksi_resu[-1]*u[-1,-1,2] -  ksi_resu[0]*u[0,0,2]) / T_total \
                            + (ksi_resu[-1] - ksi_resu[0]) * np.sum(u[:,:,2]) / (nseg * (nstep - 1) + 1) / T_total \
                            + np.sum(f_resu[:,2] * ksi_resu) / (nseg * (nstep - 1) + 1)  
                           

    
    ## plot u
    ##mpl.rcParams['legend.fontsize'] = 10   
    #plt.plot(np.ravel(u[:,:,0]), np.ravel(u[:,:,2]))
    #plt.xlabel('x')
    #plt.ylabel('y')
    #plt.show()

    ## plot some debug info
    #plt.subplot(3,1,1)
    #plt.plot(np.abs(v_resu))
    #plt.subplot(3,1,2)
    #plt.plot(np.abs(v_resu_perp))
    #plt.subplot(3,1,3)
    #othor = np.zeros(nseg*(nstep-1)+1)
    #for i in range(0, nseg*(nstep-1)+1):
    #    othor[i] = np.dot(v_resu_perp[i,:], f_resu[i,:])
    #plt.plot(np.abs(othor))
    #plt.show()

plt.rc('text', usetex=True)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 24}
plt.rc('font', **font)

# plot J vs r
plt.subplot(2,1,1)
plt.plot(rho_arr, J_arr)
plt.xlabel(r'$\rho$')
plt.ylabel(r'$\langle J \rangle$')

# plot dJdrho vs r
plt.subplot(2,1,2)
plt.plot(rho_arr, dJdrho_arr)
plt.xlabel(r'$\rho$')
plt.ylabel(r'$d \langle J \rangle / d \rho$')
plt.ylim([0,2.5])
plt.savefig('withDilation_T2500.png')
plt.show()

# plot growrate vs r
#plt.subplot(2,1,1)
#plt.plot(rho_arr,grow_rate_arr)
#plt.xlabel(r'\rho')
#plt.ylabel('growRate')
#plt.show()
print('end')
