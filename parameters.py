# use single direction and N tiem segments to solve Lorentz problem
sigma = 10
beta = 8. / 3.

rho_lb = 28
rho_ub = 28
Nrho = rho_ub - rho_lb + 1 # number of rho to be calculated

T_total = 100
T_seg = 2 # length of each segment
T_ps = 100 # time of pre-smoothing
nseg = int(T_total/ T_seg) #number of segments on time interval

dt = 0.005
nstep = int(T_seg/dt)
nseg_ps = int(T_ps/dt/nstep)

nc = 3 # number of component in u
nus = 2 # number of unstable direction
