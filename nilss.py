# use single direction and N tiem segments to solve Lorentz problem
import numpy as np
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
    

def pushSeg(nseg, nstep, nus, nc, dt, u0, vstar0, w0, s, integrator, fJJu):
    """
    find u, w, vstar, f, J, dJdu on each segment.
    Here we store all values at all time steps for better intuition,
    but this implementation costs a lot of memory.
    See the paper on FD-NILSS for discussion on how to reduce memory cost 
    by computing inner products via only snapshots.
    """

    u = np.zeros([nseg, nstep, nc])
    f = np.zeros([nseg, nstep, nc])
    J = np.zeros([nseg, nstep])
    dJdu = np.zeros([nseg, nstep, nc])
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
                = integrator(u[iseg, istep], w[iseg, istep], vstar[iseg, istep], s)
        for istep in range(0, nstep):
            f[iseg, istep], J[iseg, istep], dJdu[iseg, istep] = fJJu(u[iseg, istep], s)


        # get u, and renormalize v* and w for next segment
        if iseg < nseg - 1:
            u[iseg + 1, 0] = u[iseg, -1]
            
            [Qtemp, Rtemp] = (np.linalg.qr(w[iseg,-1].T, 'reduced'))
            w[iseg + 1, 0] = Qtemp.T

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
                w_perp[iseg, i, ius] = w[iseg, i,ius] - np.dot(w[iseg, i, ius], f[iseg, i]) / np.dot(f[iseg, i], f[iseg, i]) * f[iseg, i]


    return [u, w, vstar, w_perp, vstar_perp, f, J, dJdu]


def nilss(dt, nseg, T_seg, T_ps, u0, nus, s, integrator, fJJu):


    nc = len(u0)
    nstep = int(T_seg / dt) + 1 # number of step in each time segment
    v = np.zeros([nseg, nstep, nc])
    v_perp = np.zeros_like(v)
    ksi = np.zeros([nseg, nstep])
    

    #  get u0, vstar0, w0 for pre-smoothing
    vstar0 = [0.0, 0.0, 0.0]
    w0 = np.zeros([nus,nc])
    for ius in range(0,nus):
        w0[ius] = np.random.rand(nc)
        w0[ius] = w0[ius] / np.linalg.norm(w0[ius])
    # push forward u to a stable attractor
    nseg_ps = int(T_ps / dt / nstep)
    u_ps, w_ps, vstar_ps, _, _, _, _, _ = pushSeg(nseg_ps, nstep, nus, nc, dt, u0, vstar0, w0, s, integrator, fJJu)
    # get initial value for later integration: value of the first step of the last segment in presmoothing
    u0 = u_ps[-1,0]
    vstar0 = vstar_ps[-1,0]
    w0 = w_ps[-1,0]


    # find u, w, vstar on all segments
    [u, w, vstar, w_perp, vstar_perp, f, J, dJdu] = pushSeg(nseg, nstep, nus, nc, dt,u0, vstar0, w0, s, integrator, fJJu)


    # Construct the linear equation system given by the Lagrange multiplier method of NILSS problem.
    # See the paper on FD-NILSS for a neater method solving the Schur complement of this problem.
    # Also the integratino here are over counting contribution on the interface, should reduce by a half factor.
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


    #lbd = np.linalg.solve(M, rhs)
    lbd = np.linalg.lstsq(M, rhs)
    lbd = lbd[0][:nseg * nus]
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


    # reshape v, v_perp, f, ksi to [nseg*(nstep-1), nc] vector: delete duplicate at interfaces. 
    # check paper for another method which does not requires this reshaping
    # we do reshaping here to for better intuition: we first recover continuous v then use it
    v_resu = np.zeros([nseg * (nstep - 1) + 1, nc])
    v_resu_perp = np.zeros_like(v_resu)
    f_resu = np.zeros_like(v_resu)
    u_resu = np.zeros_like(v_resu)
    dJdu_resu = np.zeros_like(v_resu)
    ksi_resu = np.zeros([nseg * (nstep - 1) + 1])    #TODO:problem here
    J_resu = np.zeros_like(ksi_resu)
    for iseg in range(0, nseg):
        for istep in range(0, nstep - 1):
            ii = iseg * (nstep - 1) + istep
            u_resu[ii] = u[iseg, istep]
            v_resu[ii] = v[iseg, istep]
            v_resu_perp[ii] = v_perp[iseg, istep]
            f_resu[ii] = f[iseg, istep]
            dJdu_resu[ii] = dJdu[iseg, istep] 
            ksi_resu[ii] = ksi[iseg, istep]
            J_resu[ii] = J[iseg, istep]
    u_resu[-1] = u[-1,-1]
    v_resu[-1] = v[-1,-1]
    v_resu_perp[-1] = v_perp[-1,-1]
    f_resu[-1] = f[-1,-1]
    dJdu_resu[-1] = dJdu[-1,-1]
    ksi_resu[-1] = ksi[-1,-1]
    J_resu[-1] = J[-1,-1]


    # compute sensitivity
    N_step = nseg * (nstep - 1) + 1
    T_total = N_step * dt
    # special care on fixed points: conventional tangent method works
    if np.sum((u_resu[-1000:,2] -np.average(u_resu[-1000:,2]))**2)  < 1e-6 * np.sum(u_resu[-1000:,2] **2):
        dJds = v_resu[-1,2]
    else:
        # # another formula which uses vperp but has one more term
        # dJds2 = np.sum(v_resu_perp[:,2]) / N_step \
                # - (ksi_resu[-1] * u_resu[-1,2] -  ksi_resu[0] * u_resu[0,2]) / T_total \
                # + (ksi_resu[-1] - ksi_resu[0]) * np.sum(u_resu[:,2]) / N_step / T_total \
                # + np.sum(f_resu[:,2] * ksi_resu) / N_step   
        # the formula in the JCP paper of NILSS
        # dJds = np.sum(v_resu[:,2]) / N_step \
                # - (ksi_resu[-1] * u_resu[-1,2] -  ksi_resu[0] * u_resu[0,2]) / T_total \
                # + (ksi_resu[-1] - ksi_resu[0]) * np.sum(u_resu[:,2]) / N_step / T_total 
        dJds = np.sum(v_resu[:,2]) / N_step \
                - (ksi_resu[-1] * u_resu[-1,2] -  ksi_resu[0] * u_resu[0,2]) / T_total \
                + (ksi_resu[-1] - ksi_resu[0]) * np.sum(u_resu[:,2]) / N_step / T_total 


    # calculate Javg
    Javg = np.einsum(u[:,:,2],[0,1],[]) / (nstep * nseg)


    return Javg, dJds
