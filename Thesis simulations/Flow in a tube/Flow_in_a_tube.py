import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from   matplotlib import cm

## -------------------------------------------------------------------------------------------- ##
##                                     PRE-PROCESSING                                           ##
## -------------------------------------------------------------------------------------------- ##
# Data
P = 101325.0    # Pa
T = 25+273.15   # K
rho = 1000.0    # kg/m^3

Lx = 1.0        # m
Ly = 1.0        # m
nx = 5
ny = 5
nu = 1e-4       # m^2/s  (diffusion coeff)
u_in = 0.5      # m/s    (inlet velocity)
tau = 5.0       # s      (simulation time)

# Boundary conditions (no-slip)
un = 0          # m/s
us = 0          # m/s
ve = 0          # m/s
vw = 0          # m/s

# Parameters for SOR (Poisson eq)
max_iterations = 100000
beta = 1.5
max_error = 1e-4

## -------------------------------------------------------------------------------------------- ##
##                                     DATA PROCESSING                                          ##
## -------------------------------------------------------------------------------------------- ##
# if nx % 2 /= 0 || ny % 2 /= 0:
#    print('Only even number of cells can be accepted (for graphical purposes only)')

# Process the grid
hx = Lx / nx
hy = Ly / ny

# Inlet section (west side)
nin_start = 2        # first cell index (pay attention!)
nin_end = ny + 1     # last cell index (pay attention!)

# Outlet section (east side)
nout_start = 2        # first cell index (pay attention!)
nout_end = ny + 1     # last cell index (pay attention!)

# Time step
sigma = 0.7                              # safety factor for time step (stability)
dt_diff = np.minimum(hx,hy)**2/4/nu      # time step (diffusion stability) [s]
dt_conv = 4*nu/u_in**2                   # time step (convection stability) [s]
dt = sigma*np.minimum(dt_diff, dt_conv)  # time step (stability) [s]
nsteps = int(tau/dt)                     # number of steps
Re = u_in*Ly/nu                          # Reynolds' number

print('Time step:', dt, 's' )
print(' - Diffusion:', dt_diff, 's')
print(' - Convection:', dt_conv, 's')
print('Reynolds number:', Re, '\n')

# Grid construction
x = np.linspace(0, Lx, nx+1)
y = np.linspace(0, Ly, ny+1)

## -------------------------------------------------------------------------------------------- ##
##                                     MEMORY ALLOCATION                                        ##
## -------------------------------------------------------------------------------------------- ##
# Main fields (velocities and pressure)
u = np.zeros([nx+1, ny+2])
v = np.zeros([nx+2, ny+1])
p = np.zeros([nx+2, ny+2])

# Temporary velocity fields
ut = np.zeros_like(u)
vt = np.zeros_like(v)

# Fields used only for graphical purposes
uu = np.zeros([nx+1, ny+1])
vv = np.zeros([nx+1, ny+1])
pp = np.zeros([nx+1, ny+1])

# Coefficients for pressure equation
gamma = np.zeros([nx+2,ny+2]) + hx*hy / (2*hx**2 + 2*hy**2)   # internal points
gamma[1, 2:-2]  = hx*hy / (2*hx**2 +   hy**2)                 # west wall
gamma[-2, 2:-2] = hx*hy / (2*hx**2 +   hy**2)                 # east wall
gamma[2:-2, 1]  = hx*hy / (  hx**2 + 2*hy**2)                 # north wall
gamma[2:-2, -2] = hx*hy / (  hx**2 + 2*hy**2)                 # south wall
gamma[1, 1]     = hx*hy / (  hx**2 +   hy**2)                 # corners
gamma[1, -2]    = hx*hy / (  hx**2 +   hy**2)                                   
gamma[-2, 1]    = hx*hy / (  hx**2 +   hy**2) 
gamma[-2, -2]   = hx*hy / (  hx**2 +   hy**2)

# Correction of gamma taking into account inlet and outlet section
gamma[-2, nout_start-1:nout_end-1] = hx*hy / (2*hx**2 + 2*hy**2)    # Outlet section is treated as an internal cell
gamma[-2, nout_start-1]            = hx*hy / (  hx**2 + 2*hy**2)    # Corners of outlet sections
gamma[-2, nout_end-1]              = hx*hy / (  hx**2 + 2*hy**2)

# Initial conditions: set reasonable initial velocity value instead of initializing everything to zero
u[:, :] = 0.5
ut = u

## -------------------------------------------------------------------------------------------- ##
##                                         FUNCTIONS                                            ##
## -------------------------------------------------------------------------------------------- ##
# Poisson equation solver
def Pressure_Poisson(p, ut, vt, gamma, nx, ny, hx, hy, dt, beta, max_iterations, max_error):

    pn = np.empty_like(p)

    for iter in range(max_iterations):

        pn = np.copy(p)

        delta = (pn[2:, 1:-1] + pn[:-2, 1:-1])*hy/hx + (pn[1:-1, 2:] + pn[1:-1, :-2])*hx/hy
        S = (1/dt) * ( (ut[1:, 1:-1] - ut[:-1, 1:-1])*hy + (vt[1:-1, 1:] - vt[1:-1, :-1])*hx )
        p[1:-1, 1:-1] = beta * gamma[1:-1, 1:-1] * (delta - S) + (1-beta) * pn[1:-1, 1:-1]

        if iter == 100:
            print('delta', np.transpose(delta))
            print('S', np.transpose(S))
            print('p', np.transpose(p))
            print('ut', np.transpose(ut))
            print('vt', np.transpose(vt))

        # Estimate the error
        epsilon = np.abs(p[1:-1, 1:-1] - gamma[1:-1, 1:-1] * (delta - S))
        epsilon = np.sum(epsilon) / (nx*ny)

        # Check the error
        if epsilon <= max_error:
            break

    return p , iter

def Pressure_Poisson_prova(p, ut, vt, gamma, nx, ny, hx, hy, dt, beta, max_iterations, max_error):

    for iter in range(max_iterations):

        for i in range(1,nx+1):
            for j in range(1,ny+1):

                delta = (p[i+1, j] + p[i-1,j])*hy/hx + (p[i, j+1] + p[i, j-1])*hx/hy
                S = (1/dt) * ( (ut[i, j] - ut[i-1, j])*hy + (vt[i, j] - vt[i, j-1])*hx )
                p[i, j] = beta * gamma[i, j] * (delta - S) + (1-beta) * p[i, j]

        
        if iter == 100:
            print('delta', np.transpose(delta))
            print('S', np.transpose(S))
            print('p', np.transpose(p))
            print('ut', np.transpose(ut))
            print('vt', np.transpose(vt))
        

        # Estimate the error
        epsilon = np.abs(p[1:-1, 1:-1] - gamma[1:-1, 1:-1] * (delta - S))
        epsilon = np.sum(epsilon) / (nx*ny)

        # Check the error
        if epsilon <= max_error:
            break

    return p , iter

# Advection diffusion equation
def AdvectionDiffusion2D(ut, vt, u, v, nx, ny, hx, hy, dt, nu, method):

    # Temporary u-velocity
    # Centered Differencing Scheme
    if method == 'CDS':
        ue2 = ((u[2:, 1:-1] + u[1:-1, 1:-1]) / 2)**2 * hy
        uw2 = ((u[1:-1, 1:-1] + u[:-2, 1:-1]) / 2)**2 * hy
        unv = ((u[1:-1, 2:] + u[1:-1, 1:-1]) / 2) * ((v[1:-2, 1:] + v[2:-1, 1:]) / 2) * hx
        usv = ((u[1:-1, 1:-1] + u[1:-1, :-2]) / 2) * ((v[1:-2, :-1] + v[2:-1, :-1]) / 2) * hx

    if method == 'Upwind':         # DA RIGUARDARE PER TOGLIERE I CICLI FOR!!!!!!!!
        for i in range(1, nx):
            for j in range(1, ny+1):

            # Upwind
                flux_e = (u[i+1,j] + u[i,j]) / 2
                flux_w = (u[i,j] + u[i-1,j]) / 2
                flux_n = (v[i,j] + v[i+1,j]) / 2
                flux_s = (v[i,j-1] + v[i+1,j-1]) / 2

                if flux_e > 0: ufe = u[i,j]
                else: ufe = u[i+1,j]
                if flux_w > 0: ufw = u[i-1,j]
                else: ufw = u[i,j]
                if flux_n > 0: ufn = u[i,j]
                else: ufn = u[i,j+1]
                if flux_s > 0: ufs = u[i,j-1]
                else: ufs = u[i,j]

                ue2 = ufe**2 * hy
                uw2 = ufw**2 * hy
                unv = flux_n * ufn * hx
                usv = flux_s * ufs * hx

    H = hx*hy
    A = (ue2 - uw2 + unv - usv) / H

    De = nu * (u[2:, 1:-1] - u[1:-1, 1:-1]) * hy/hx
    Dw = nu * (u[1:-1, 1:-1] - u[:-2, 1:-1]) * hy/hx
    Dn = nu * (u[1:-1, 2:] - u[1:-1, 1:-1]) * hx/hy
    Ds = nu * (u[1:-1, 1:-1] - u[1:-1, :-2]) * hx/hy

    D = (De - Dw + Dn - Ds) / H

    ut[1:-1, 1:-1] = u[1:-1, 1:-1] + dt*(-A + D)

    # Temporary v-velocity
    # Centered Differencing Scheme
    if method == 'CDS':
        vn2 = ((v[1:-1, 2:] + v[1:-1, 1:-1]) / 2)**2 * hx
        vs2 = ((v[1:-1, 1:-1] + v[1:-1, :-2]) / 2)**2 * hx
        veu = ((v[2:, 1:-1] + v[1:-1, 1:-1]) / 2) * ((u[1:, 1:-2] + u[1:, 2:-1]) / 2) * hy
        vwu = ((v[1:-1, 1:-1] + v[:-2, 1:-1]) / 2) * ((u[:-1, 1:-2] + u[:-1, 2:-1]) / 2) * hy

    if method == 'Upwind':         # DA RIGUARDARE PER TOGLIERE I CICLI FOR!!!!!!
        for i in range(1, nx+1):
            for j in range(1, ny):

            # Upwind
                flux_e = (u[i,j+1] + u[i,j]) / 2
                flux_w = (u[i-1,j+1] + u[i-1,j]) / 2
                flux_n = (v[i,j+1] + v[i,j]) / 2
                flux_s = (v[i,j] + v[i,j-1]) / 2

                if flux_e > 0: vfe = v[i,j]
                else: vfe = v[i+1,j]
                if flux_w > 0: vfw = v[i-1,j]
                else: vfw = v[i,j]
                if flux_n > 0: vfn = v[i,j]
                else: vfn = v[i,j+1]
                if flux_s > 0: vfs = v[i,j-1]
                else: vfs = v[i,j]

                vn2 = vfn**2 * hx
                vs2 = vfs**2 * hx
                veu = flux_e * vfe * hy
                vwu = flux_w * vfw * hy

    H = hx*hy
    A = (vn2 - vs2 + veu - vwu) / H

    De = nu * (v[2:, 1:-1] - v[1:-1, 1:-1]) * hy/hx
    Dw = nu * (v[1:-1, 1:-1] - v[:-2, 1:-1]) * hy/hx
    Dn = nu * (v[1:-1, 2:] - v[1:-1, 1:-1]) * hx/hy
    Ds = nu * (v[1:-1, 1:-1] - v[1:-1, :-2]) * hx/hy

    D = (De - Dw + Dn - Ds) / H

    vt[1:-1, 1:-1] = v[1:-1, 1:-1] + dt*(-A + D)

    return ut, vt

## -------------------------------------------------------------------------------------------- ##
##                                     SOLUTION OVER TIME                                       ##
## -------------------------------------------------------------------------------------------- ##
t = 0.0
for it in range(1, nsteps+1):

    # Boundary conditions
    u[:, 0]  = 2*us - u[:, 1]    # south wall
    u[:, -1] = 2*un - u[:, -2]   # north wall
    v[0, :]  = 2*vw - v[1, :]    # west wall
    v[-1, :] = 2*ve - v[-2, :]   # east wall

    # Over-writing inlet conditions
    u[0, nin_start-1:nin_end] = u_in   # fixed inlet velocity

    # Over-writing outlet conditions
    u[-1, nout_start-1:nout_end] = u[-2, nout_start-1:nout_end]   # zero-gradient outlet velocity
    v[-1, nout_start-1:nout_end] = v[-2, nout_start-1:nout_end]   # zero-gradient outlet velocity

    # Advection-diffusion equation (predictor)
    ut, vt = AdvectionDiffusion2D(ut, vt, u, v, nx, ny, hx, hy, dt, nu, 'CDS')

    # Update boundary conditions for temporary velocities
    ut[0, nin_start-1:nin_end] = u_in       # fixed inlet velocity
    ut[-1, nout_start-1:nout_end] = u[-1, nout_start-1:nout_end]   # zero-gradient outlet velocity
    vt[-1, nout_start-1:nout_end] = v[-1, nout_start-1:nout_end]   # zero-gradient outlet velocity

    # Pressure equation (Poisson)
    p, iter = Pressure_Poisson(p, ut, vt, gamma, nx, ny, hx, hy, dt, beta, max_iterations, max_error)

    # Correction on the velocity
    u[1:nx, 1:ny+1] = ut[1:nx, 1:ny+1] - dt/hx * (p[2:nx+1, 1:ny+1] - p[1:nx, 1:ny+1])
    v[1:nx+1, 1:ny] = vt[1:nx+1, 1:ny] - dt/hy * (p[1:nx+1, 2:ny+1] - p[1:nx+1, 1:ny])

    # Print on the screen
    if it % 10 == 0:
        print('Step:', it, '- Time:', t,  '- Poisson iterations:', iter)

    # Advance in time
    t = t + dt

## -------------------------------------------------------------------------------------------- ##
##                                    FINAL POST-PROCESSING                                     ##
## -------------------------------------------------------------------------------------------- ##
# Field reconstruction
uu[:,:] = (u[:, 1:] + u[:, :-1]) / 2
vv[:,:] = (v[1:, :] + v[:-1, :]) /2
pp[:,:] = (p[:-1, :-1] + p[:-1, 1:] + p[1:, :-1] + p[1:, 1:]) / 4

fig = plt.figure(figsize=[1,1],dpi=400)
plt.contourf(np.transpose(pp))

fig = plt.figure(figsize=[1,1],dpi=400)
plt.contourf(np.transpose(uu))

fig = plt.figure(figsize=[1,1],dpi=400)
plt.contourf(np.transpose(vv))

plt.show()