# use single direction and N tiem segments to solve Lorentz problem
import numpy as np
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from pdb import set_trace
    

def pushSeg(nseg, nstep, nus, nc, dt, u0, vstar0, w0, s, integrator, fJJu):
    """
    find u, w, vstar, f, J, dJdu on each segment.
    Here we store all values at all time steps for better intuition,
    but this implementation costs a lot of memory.
    See the paper on FD-NILSS for discussion on how to reduce memory cost 
    by computing inner products via only snapshots.
    """

    J = np.zeros([nseg, nstep])
    u = np.zeros([nseg, nstep, nc])
    f = np.zeros(u.shape)
    dJdu = np.zeros(u.shape)
    vstar = np.zeros(u.shape)
    vstar_perp = np.zeros(u.shape)
    w = np.zeros([nseg, nstep, nus, nc])
    w_perp = np.zeros(w.shape)
    Rs = [] #Rs[0] in code = R_1 in paper
    bs = [] #bs[0] in code = b_1 in paper
   
    # assign initial value, u[0,0], v*[0,0], w[0,0]
    u[0,0] = u0
    vstar[0,0] = vstar0
    w[0,0] = w0

    # push forward
    for iseg in range(0, nseg):

        # compute u, w, vstar, f, J, dJdu
        for istep in range(0, nstep-1):
            u[iseg, istep+1], w[iseg, istep+1], vstar[iseg,istep+1]\
                = integrator(u[iseg, istep], w[iseg, istep], vstar[iseg, istep], s)
        for istep in range(0, nstep):
            f[iseg, istep], J[iseg, istep], dJdu[iseg, istep] = fJJu(u[iseg, istep], s)

        # calculate vstar_perp and w_perp
        for i in range(0, nstep):
            vstar_perp[iseg, i] = vstar[iseg, i] - np.dot(vstar[iseg, i], f[iseg, i]) / np.dot(f[iseg, i], f[iseg, i]) * f[iseg, i]
            for ius in range(0, nus):
                w_perp[iseg, i, ius] = w[iseg, i,ius] - np.dot(w[iseg, i, ius], f[iseg, i]) / np.dot(f[iseg, i], f[iseg, i]) * f[iseg, i]

        # renormalize at interfaces
        Q_temp, R_temp = np.linalg.qr(w_perp[iseg,-1].T, 'reduced')
        Rs.append(R_temp)
        b_temp = Q_temp.T @ vstar_perp[iseg,-1]
        bs.append(b_temp)
        p_temp = vstar_perp[iseg,-1] - Q_temp @ b_temp
        if iseg < nseg - 1:
            u[iseg+1, 0] = u[iseg, -1]
            w[iseg+1, 0] = Q_temp.T
            vstar[iseg+1,0] = p_temp

    return [u, w, vstar, w_perp, vstar_perp, f, J, dJdu, Rs[:-1], bs[:-1], Q_temp, p_temp]


def nilss(dt, nseg, T_seg, nseg_ps, u0, nus, s, integrator, fJJu):

    nc = len(u0)
    nstep = int(round(T_seg / dt)) + 1 # number of step + 1 in each time segment
    
    # push forward u to a stable attractor
    vstar0 = [0.0, 0.0, 0.0]
    w0 = np.zeros([nus,nc])
    for ius in range(0,nus):
        w0[ius] = np.random.rand(nc)
        w0[ius] = w0[ius] / np.linalg.norm(w0[ius])
    u_ps, w_ps, vstar_ps, _, _, _, _, _, _, _, Q_ps, p_ps= pushSeg(nseg_ps, nstep, nus, nc, dt, u0, vstar0, w0, s, integrator, fJJu)
    u0 = u_ps[-1,-1]
    w0 = Q_ps.T
    vstar0 = p_ps

    # find u, w, vstar on all segments
    u, w, vstar, w_perp, vstar_perp, f, J, dJdu, Rs, bs, _, _ = pushSeg(nseg, nstep, nus, nc, dt,u0, vstar0, w0, s, integrator, fJJu)

    # a weight matrix for integration, 0.5 at interfaces
    weight = np.ones(nstep)
    weight[0] = weight[-1] = 0.5

    # compute Javg
    Javg = np.sum(J*weight[np.newaxis,:]) / (nstep-1) / nseg

    # Construct Schur complement of the Lagrange multiplier method of the NILSS problem.
    # See the paper on FD-NILSS for this neat method
    # find C^-1
    Cinvs = []
    for iseg in range(nseg):
        C_iseg = np.zeros([nus, nus])
        for i in range(nus):
            for j in range(nus):
                C_iseg[i,j] = np.sum(w_perp[iseg, :, i, :] * w_perp[iseg, :, j, :] * weight[:, np.newaxis])
        Cinvs.append(np.linalg.inv(C_iseg))
    Cinv = block_diag(*Cinvs)

    # construct d
    ds = []
    for iseg in range(nseg):
        d_iseg = np.zeros(nus)
        for i in range(nus):
            d_iseg[i] = np.sum(w_perp[iseg, :, i, :] * vstar_perp[iseg] * weight[:, np.newaxis])
        ds.append(d_iseg)
    d = np.ravel(ds) 

    # construct B, first off diagonal I, then add Rs
    B = np.eye((nseg-1)*nus, nseg*nus, k=nus)
    B[:, :-nus] -= block_diag(*Rs)

    # construct b
    b = np.ravel(bs)

    # solve
    lbd = np.linalg.solve(-B @ Cinv @ B.T, B @ Cinv @ d + b)
    a = -Cinv @ (B.T @ lbd + d)
    a = a.reshape([nseg, nus])


    # calculate v and vperp
    v = np.zeros([nseg, nstep, nc])
    v_perp = np.zeros(v.shape)
    for iseg in range(nseg):
        v[iseg] = vstar[iseg]
        v_perp[iseg] = vstar_perp[iseg]
        for ius in range(0, nus):
            v[iseg] += a[iseg, ius] * w[iseg,:, ius,:]
            v_perp[iseg] += a[iseg, ius] * w_perp[iseg,:, ius,:]
    

    # calculate ksi, only need to use last step in each segment
    ksi = np.zeros([nseg, nstep])
    for iseg in range(nseg):
        for i in (0, -1):
            ksi[iseg,i] = np.dot(v[iseg, i], f[iseg, i]) / np.dot(f[iseg, i], f[iseg, i])
        assert abs(ksi[iseg, 0]) <= 1e-5 


    # compute dJds
    dJdss = []
    for iseg in range(nseg):
        t1 = np.sum(dJdu[iseg] * v[iseg] * weight[:,np.newaxis]) / (nstep-1) / nseg
        t2 = ksi[iseg,-1] * (Javg - J[iseg,-1]) / (nstep-1) / nseg / dt
        dJdss.append(t1+t2)
    dJds = np.sum(dJdss)


    return Javg, dJds
