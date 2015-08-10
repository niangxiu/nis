import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

Nrho = 80
J_arr = np.zeros(Nrho)
dJdrho_arr = np.zeros(Nrho)
rho_arr = np.zeros(Nrho)

for rr in range(0,Nrho):
    print(rr)
    rho = rr + 3
    sigma = 10
    beta = 8./3.
    T = 10
    dt = 0.005
    nc =3 # number of component in u
    N = int(T/dt)
    Df = np.zeros([3,3])
    u = np.zeros([N,3])
    v = np.zeros([N,3])
    vstar = np.zeros([N,3])
    w = np.zeros([nc,N,nc])
    w0 = np.zeros([N,3])
    w1 = np.zeros([N,3])
    w2 = np.zeros([N,3])

    # give start value at t=0
    u[0] = [-10, -10, 60]
    vstar[0] = [0, 0, 0]
    w[0,0] = [1, 0, 0]
    w[1,0] = [0, 1, 0]
    w[2,0] = [0, 0, 1]

    for i in range(1, N):
        # push forward u to i-th step
        [x, y ,z] = u[i-1]
        dudt = np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])
        u[i] = u[i-1] + dudt*dt
        # get Df and drhof
        Dftemp = np.array([[-sigma, sigma, 0],[rho-z,-1,-x],[y,x,-beta]])
        Df = Dftemp * dt + np.identity(3)
        drhof = np.array([0, x*dt, 0])
        pass
        # push forwar vstar and w
        vstar[i] = np.dot(Df, vstar[i-1]) + drhof
        w[0,i] = np.dot(Df, w[0,i-1])
        w[1,i] = np.dot(Df, w[1,i-1])
        w[2,i] = np.dot(Df, w[2,i-1])

    # calculate M rhs lbd, and v
    M = np.zeros([nc, nc])
    for i in range(0, nc):
        for j in range(0, i+1):
            M[i,j] = np.einsum(w[i],[0,1],w[j],[0,1])
            M[j,i] = M[i,j]

    rhs = np.zeros(3)
    rhs[0] = np.einsum(w[0], [0,1], vstar, [0,1])
    rhs[1] = np.einsum(w[1], [0,1], vstar, [0,1])
    rhs[2] = np.einsum(w[2], [0,1], vstar, [0,1])

    lbd = np.linalg.solve(M, rhs)
    v = vstar - np.einsum(lbd, [0], w, [0,1,2], [1,2])

    J_arr[rr] = np.sum(u[:,2]) / N
    dJdrho = np.sum(v[:,2]) / N
    rho_arr[rr] = rho
    dJdrho_arr[rr] = dJdrho

# plot u
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.plot(u[:,0],u[:,1],u[:,2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

# plot J vs r
fig = plt.figure()
ax = fig.gca()
ax.plot(rho_arr, J_arr)
ax.set_xlabel('rho')
ax.set_ylabel('J')
plt.show()

# plot dJdrho vs r
fig = plt.figure()
ax = fig.gca()
ax.plot(rho_arr, dJdrho_arr)
ax.set_xlabel('rho')
ax.set_ylabel('dJdrho')
plt.show()
pass

pass