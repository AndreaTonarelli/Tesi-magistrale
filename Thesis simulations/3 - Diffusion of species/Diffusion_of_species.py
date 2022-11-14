import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from   matplotlib import cm
from   matplotlib.patches import Rectangle
from   numba import njit

## -------------------------------------------------------------------------------------------- ##
##                                     PRE-PROCESSING                                           ##
## -------------------------------------------------------------------------------------------- ##
# Data
P = 101325.0       # Pa
T = 25+273.15      # K

Lx = 3.0           # m
Ly = 1.0           # m
nx = 360
ny = 120
nu = 5e-5          # m2/s  (dynamic viscosity of fluid)
Gamma = 1e-5       # m2/s  (diffusion coefficient of O2 in water)
u_in = 0.5         # m/s    (inlet velocity)
tau = 20.0         # s      (simulation time)
method = 'Upwind'  # Discretization method (CDS or Upwind)

# Boundary conditions (no-slip)
un = 0             # m/s
us = 0             # m/s
ve = 0             # m/s
vw = 0             # m/s

# Inlet concentrations
cO2in = 0.1       # kmol/m3

# Parameters for SOR (Poisson eq)
max_iterations = 1000000
beta = 1.8
max_error = 5e-5

## -------------------------------------------------------------------------------------------- ##
##                                     DATA PROCESSING                                          ##
## -------------------------------------------------------------------------------------------- ##
if nx % 2 != 0 and ny % 2 != 0:
    print('\nOnly even number of cells can be accepted (for graphical purposes only)')

# Process the grid
hx = Lx / nx
hy = Ly / ny

# Grid construction
x = np.linspace(0, Lx, nx+1)
y = np.linspace(0, Ly, ny+1)

# Obstacle coordinates
# Obstacle 1
x_obst1_start = 0.
x_obst1_end = 0.8
y_obst1_start = 0.
y_obst1_end = 0.2
# Obstacle 2
x_obst2_start = 0.
x_obst2_end = 0.8
y_obst2_start = 0.8
y_obst2_end = 1
# Obstacle 3
x_obst3_start = 2.2
x_obst3_end = 3.0
y_obst3_start = 0.3
y_obst3_end = 0.7

# Obstacles definition: rectangle with base xs:xe and height ys:ye
xs1 = np.where(x <= x_obst1_start)[0][-1] + 1
xe1 = np.where(x < x_obst1_end)[0][-1] + 1
ys1 = np.where(y <= y_obst1_start)[0][-1] + 1
ye1 = np.where(y < y_obst1_end)[0][-1] + 1

xs2 = np.where(x <= x_obst2_start)[0][-1] + 1
xe2 = np.where(x < x_obst2_end)[0][-1] + 1
ys2 = np.where(y <= y_obst2_start)[0][-1] + 1
ye2 = np.where(y < y_obst2_end)[0][-1] + 1

xs3 = np.where(x <= x_obst3_start)[0][-1] + 1
xe3 = np.where(x < x_obst3_end)[0][-1] + 1
ys3 = np.where(y <= y_obst3_start)[0][-1] + 1
ye3 = np.where(y < y_obst3_end)[0][-1] + 1

# Inlet section (west side)
nin_start = 1        # first cell index (pay attention!)
nin_end = ny + 1     # last cell index (pay attention!)

# Outlet section (east side)
nout_start = 2        # first cell index (pay attention!)
nout_end = ny + 1     # last cell index (pay attention!)

# Time step
sigma = 0.75                                       # safety factor for time step (stability)
dt_diff_ns = np.minimum(hx,hy)**2/4/nu             # time step (diffusion stability) [s]
dt_conv_ns = 4*nu/u_in**2                          # time step (convection stability) [s]
dt_ns      = np.minimum(dt_diff_ns, dt_conv_ns);   # time step (stability due to FD) [s]
dt_diff_sp = np.minimum(hx,hy)**2/4/Gamma;         # time step (species diffusion stability) [s]
dt_conv_sp = 4*Gamma/u_in**2;                      # time step (species convection stability) [s]
dt_sp      = np.minimum(dt_conv_sp, dt_diff_sp);   # time step (stability due to species) [s]
dt         = sigma*min(dt_ns, dt_sp);              # time step (stability) [s]
nsteps     = int(tau/dt)                           # number of steps
Re         = u_in*Ly/nu                            # Reynolds' number

print('Time step:', dt, 's' )
print(' - Diffusion (NS):', dt_diff_ns, 's')
print(' - Convection (NS):', dt_conv_ns, 's')
print(' - Diffusion (Species):', dt_diff_sp, 's')
print(' - Convection (Species):', dt_conv_sp, 's')
print('Reynolds number:', Re, '\n')

## -------------------------------------------------------------------------------------------- ##
##                                     MEMORY ALLOCATION                                        ##
## -------------------------------------------------------------------------------------------- ##
# Main fields (velocities and pressure)
u   = np.zeros([nx+1, ny+2])
v   = np.zeros([nx+2, ny+1])
p   = np.zeros([nx+2, ny+2])
cO2 = np.zeros([nx+2, ny+2])

# Temporary velocity fields
ut = np.zeros_like(u)
vt = np.zeros_like(v)

# Fields used only for graphical purposes
uu   = np.zeros([nx+1, ny+1])
vv   = np.zeros([nx+1, ny+1])
pp   = np.zeros([nx+1, ny+1])
ccO2 = np.zeros([nx+1, ny+1])

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

# Correction of gamma close to the obstacles
gamma[xs1-1, ys1:ye1+1] = hx*hy / (2*hx**2 +   hy**2)         # Obstacle 1
gamma[xe1+1, ys1:ye1+1] = hx*hy / (2*hx**2 +   hy**2)
gamma[xs1:xe1+1, ys1-1] = hx*hy / (  hx**2 + 2*hy**2)
gamma[xs1:xe1+1, ye1+1] = hx*hy / (  hx**2 + 2*hy**2)

gamma[xs2-1, ys2:ye2+1] = hx*hy / (2*hx**2 +   hy**2)         # Obstacle 2
gamma[xe2+1, ys2:ye2+1] = hx*hy / (2*hx**2 +   hy**2)
gamma[xs2:xe2+1, ys2-1] = hx*hy / (  hx**2 + 2*hy**2)
gamma[xs2:xe2+1, ye2+1] = hx*hy / (  hx**2 + 2*hy**2)

gamma[xs3-1, ys3:ye3+1] = hx*hy / (2*hx**2 +   hy**2)         # Obstacle 3
gamma[xe3+1, ys3:ye3+1] = hx*hy / (2*hx**2 +   hy**2)
gamma[xs3:xe3+1, ys3-1] = hx*hy / (  hx**2 + 2*hy**2)
gamma[xs3:xe3+1, ye3+1] = hx*hy / (  hx**2 + 2*hy**2)

# Correction of gamma close to the obstacles edges
gamma[xs1, ys1-1] = hx*hy / (hx**2 + hy**2)
gamma[xs1, ye1+1] = hx*hy / (hx**2 + hy**2)
gamma[xe1, ys1-1] = hx*hy / (hx**2 + hy**2)

gamma[xs2, ys2-1] = hx*hy / (hx**2 + hy**2)
gamma[xs2, ye2+1] = hx*hy / (hx**2 + hy**2)
gamma[xe2, ye2+1] = hx*hy / (hx**2 + hy**2)

gamma[xe3, ys3-1] = hx*hy / (hx**2 + hy**2)
gamma[xe3, ye3+1] = hx*hy / (hx**2 + hy**2)

# Flags to recognize where is necessary to solve the equations
# This flag is 1 in the cells that contain and obstacle and 0 in the others
flagu = np.zeros([nx+1, ny+2])          # u-cells corresponding to the obstacle
flagv = np.zeros([nx+2, ny+1])          # v-cells corresponding to the obstacle
flagp = np.zeros([nx+2, ny+2])          # p-cells corresponding to the obstacle

# Set flag to 1 in obstacle cells
flagu[xs1-1:xe1+1, ys1:ye1+1] = 1
flagv[xs1:xe1+1, ys1-1:ye1+1] = 1
flagp[xs1:xe1+1, ys1:ye1+1] = 1

flagu[xs2-1:xe2+1, ys2:ye2+1] = 1
flagv[xs2:xe2+1, ys2-1:ye2+1] = 1
flagp[xs2:xe2+1, ys2:ye2+1] = 1

flagu[xs3-1:xe3+1, ys3:ye3+1] = 1
flagv[xs3:xe3+1, ys3-1:ye3+1] = 1
flagp[xs3:xe3+1, ys3:ye3+1] = 1

# Initial conditions: set reasonable initial velocity value instead of initializing everything to zero
u[:, :] = 0.5                      # Internal points: fixed velocity [m/s]
u[xs1-1:xe1+1, ys1:ye1+1] = 0      # Obstacle 1
u[xs2-1:xe2+1, ys2:ye2+1] = 0      # Obstacle 2
u[xs3-1:xe3+1, ys3:ye3+1] = 0      # Obstacle 3
ut = u

## -------------------------------------------------------------------------------------------- ##
##                                         FUNCTIONS                                            ##
## -------------------------------------------------------------------------------------------- ##
# Poisson equation solver
@njit
def Pressure_Poisson(p, ut, vt, gamma, nx, ny, hx, hy, dt, beta, max_iterations, max_error, flagp):

    for iter in range(1,max_iterations+1):

        for i in range(1,nx+1):
            for j in range(1,ny+1):
                if flagp[i,j] == 0:

                    delta = (p[i+1, j] + p[i-1,j])*hy/hx + (p[i, j+1] + p[i, j-1])*hx/hy
                    S = (1/dt) * ( (ut[i, j] - ut[i-1, j])*hy + (vt[i, j] - vt[i, j-1])*hx )
                    p[i, j] = beta * gamma[i, j] * (delta - S) + (1-beta) * p[i, j]
        
        # Estimate the error
        epsilon = 0.0
        for i in range(1,nx+1):
            for j in range(1,ny+1):
                if flagp[i,j] == 0:

                    delta = (p[i+1, j] + p[i-1,j])*hy/hx + (p[i, j+1] + p[i, j-1])*hx/hy
                    S = (1/dt) * ( (ut[i, j] - ut[i-1, j])*hy + (vt[i, j] - vt[i, j-1])*hx )
                    epsilon = epsilon + np.absolute( p[i, j] - gamma[i, j] * (delta - S) )

        epsilon = epsilon / (nx*ny)

        # Check the error and stop if converged
        if epsilon <= max_error:
            break

    return p , iter

# Advection diffusion equation
@njit
def AdvectionDiffusion2D(ut, vt, u, v, nx, ny, hx, hy, dt, nu, flagu, flagv, method):

    # Temporary u-velocity
    for i in range(1, nx):
        for j in range(1, ny+1):
            if flagu[i,j] == 0:

            # Upwind
                if method == 'Upwind':
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

                # Centered Differencing Scheme
                if method == 'CDS':
                    ue2 = ((u[i+1, j] + u[i, j]) / 2)**2 * hy
                    uw2 = ((u[i, j] + u[i-1, j]) / 2)**2 * hy
                    unv = ((u[i, j+1] + u[i, j]) / 2) * ((v[i, j] + v[i+1, j]) / 2) * hx
                    usv = ((u[i, j] + u[i, j-1]) / 2) * ((v[i, j-1] + v[i+1, j-1]) / 2) * hx

                H = hx*hy
                A = (ue2 - uw2 + unv - usv) / H

                De = nu * (u[i+1, j] - u[i, j]) * hy/hx
                Dw = nu * (u[i, j] - u[i-1, j]) * hy/hx
                Dn = nu * (u[i, j+1] - u[i, j]) * hx/hy
                Ds = nu * (u[i, j] - u[i, j-1]) * hx/hy

                D = (De - Dw + Dn - Ds) / H

                ut[i, j] = u[i, j] + dt*(-A + D)

    # Temporary v-velocity
    for i in range(1, nx+1):
        for j in range(1, ny):
            if flagv[i,j] == 0:

                # Upwind
                if method == 'Upwind':
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

                # Centered Differencing Scheme
                if method == 'CDS':
                    vn2 = ((v[i, j+1] + v[i, j]) / 2)**2 * hx
                    vs2 = ((v[i, j] + v[i, j-1]) / 2)**2 * hx
                    veu = ((v[i+1, j] + v[i, j]) / 2) * ((u[i, j] + u[i, j+1]) / 2) * hy
                    vwu = ((v[i, j] + v[i-1, j]) / 2) * ((u[i-1, j] + u[i-1, j+1]) / 2) * hy

                H = hx*hy
                A = (vn2 - vs2 + veu - vwu) / H

                De = nu * (v[i+1, j] - v[i, j]) * hy/hx
                Dw = nu * (v[i, j] - v[i-1, j]) * hy/hx
                Dn = nu * (v[i, j+1] - v[i, j]) * hx/hy
                Ds = nu * (v[i, j] - v[i, j-1]) * hx/hy

                D = (De - Dw + Dn - Ds) / H

                vt[i, j] = v[i, j] + dt*(-A + D)

    return ut, vt

@njit
def AdvDiffSpecies(phi, u, v, dt, hx, hy, Gamma, nx, ny, flagp, method):
    
    phio = phi
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            if flagp[i,j] == 0:

                ue = u[i, j]
                uw = u[i-1, j]
                vn = v[i, j]
                vs = v[i, j-1]

                # Centered Differencing Scheme
                if method == 'CDS':
                    phi_e = ( phio[i+1, j] + phio[i, j] ) / 2
                    phi_w = ( phio[i, j] + phio[i-1, j] ) / 2
                    phi_n = ( phio[i, j+1] + phio[i, j] ) / 2
                    phi_s = ( phio[i, j] + phio[i, j-1] ) / 2
                
                # Upwind
                if method == 'Upwind':
                    if vn > 0: phi_n = phio[i, j]
                    else: phi_n = phio[i, j+1]
                    if vs > 0: phi_s = phio[i, j-1]
                    else: phi_s = phio[i, j]
                    if ue > 0: phi_e = phio[i, j]
                    else: phi_e = phio[i+1, j]
                    if uw > 0: phi_w = phio[i-1, j]
                    else: phi_w = phio[i, j]

                A = (ue*phi_e - uw*phi_w)*hy + (vn*phi_n - vs*phi_s)*hx

                dphi_e = ( phio[i+1, j] - phio[i, j] )
                dphi_w = ( phio[i, j] - phio[i-1, j] )
                dphi_n = ( phio[i, j+1] - phio[i, j] )
                dphi_s = ( phio[i, j] - phio[i, j-1] )

                D = Gamma * ( (dphi_e - dphi_w)*hx/hy + (dphi_n - dphi_s)*hy/hx )

                phi[i, j] = phio[i, j] + dt/(hx*hy) * (-A + D)

    return phi

def SpeciesBCs(phi, nx,ny, xs1,xe1,ys1,ye1, xs2,xe2,ys2,ye2, xs3,xe3,ys3,ye3):

    # Entire domain BCs (Dirichlet)
    phi[1:-1, 1] = phi[1:-1, 2];           # South wall
    phi[1:-1, -1] = phi[1:-1, -2];         # North wall
    phi[1, 1:-1] = phi[2, 1:-1];           # West wall
    phi[-1, 1:-1] = phi[-2, 1:-1];         # East wall

    # 1st Obstacle BCS (Dirichlet)
    phi[xs1:xe1+1, ys1] = phi[xs1:xe1+1, ys1-1];     # South wall
    phi[xs1:xe1+1, ye1] = phi[xs1:xe1+1, ye1+1];     # North wall
    phi[xs1, ys1:ye1+1] = phi[xs1-1, ys1:ye1+1];     # West wall
    phi[xe1, ys1:ye1+1] = phi[xe1+1, ys1:ye1+1];     # East wall

    # 2st Obstacle BCS (Dirichlet)
    phi[xs2:xe2+1, ys2] = phi[xs2:xe2+1, ys2-1];     # South wall
    phi[xs2:xe2+1, ye2] = phi[xs2:xe2+1, ye2+1];     # North wall
    phi[xs2, ys2:ye2+1] = phi[xs2-1, ys2:ye2+1];     # West wall
    phi[xe2, ys2:ye2+1] = phi[xe2+1, ys2:ye2+1];     # East wall

    # 3st Obstacle BCS (Dirichlet)
    phi[xs3:xe3+1, ys3] = phi[xs3:xe3+1, ys3-1];     # South wall
    phi[xs3:xe3+1, ye3] = phi[xs3:xe3+1, ye3+1];     # North wall
    phi[xs3, ys3:ye3+1] = phi[xs3-1, ys3:ye3+1];     # West wall
    phi[xe3, ys3:ye3+1] = phi[xe3+1, ys3:ye3+1];     # East wall

    return phi

@njit
def node_interp(f, grid, nx, ny, flag):

    fnode = np.zeros((nx+1, ny+1))

    if grid == 'u':
        for i in range(nx+1):
            for j in range(ny+1):
                if flag[i,j] == 0:
                        fnode[i,j] = (f[i,j] + f[i, j+1]) / 2

    if grid == 'v':
        for i in range(nx+1):
            for j in range(ny+1):
                if flag[i,j] == 0:
                        fnode[i,j] = (f[i,j] + f[i+1, j]) / 2

    if grid == 'p':
        for i in range(nx+1):
            for j in range(ny+1):
                if flag[i,j] == 0:
                        fnode[i,j] = (f[i,j] + f[i+1,j] + f[i,j+1] + f[i+1,j+1]) / 4

    return fnode

@njit
def correction_velocity(u, v, ut, vt, p, nx, ny, hx, hy, flagu, flagv):

    for i in range(1, nx):
        for j in range(1, ny+1):
            if flagu[i,j] == 0:
                u[i, j] = ut[i, j] - (dt/hx) * (p[i+1, j] - p[i, j])

    for i in range(1, nx+1):
        for j in range(1, ny):
            if flagv[i,j] == 0:
                v[i, j] = vt[i, j] - (dt/hy) * (p[i, j+1] - p[i, j])

    return u, v

## -------------------------------------------------------------------------------------------- ##
##                                     SOLUTION OVER TIME                                       ##
## -------------------------------------------------------------------------------------------- ##
t = 0.0
for it in range(1, nsteps+1):

    #----------------------------------------------------------------------------------#
    # 1. Projection Algorithm                                                          #
    #----------------------------------------------------------------------------------#
    # Boundary conditions
    u[:, 0]  = 2*us - u[:, 1]    # south wall
    u[:, -1] = 2*un - u[:, -2]   # north wall
    v[0, :]  = 2*vw - v[1, :]    # west wall
    v[-1, :] = 2*ve - v[-2, :]   # east wall

    # Boundary conditions (obstacles)
    uwall = 0
    u[xs1-1:xe1+1, ye1] = 2*uwall - u[xs1-1:xe1+1, ye1+1]    # north face
    u[xs1-1:xe1+1, ys1] = 2*uwall - u[xs1-1:xe1+1, ys1-1]    # south face
    v[xs1, ys1-1:ye1+1] = 2*uwall - v[xs1-1, ys1-1:ye1+1]    # west  face
    v[xe1, ys1-1:ye1+1] = 2*uwall - v[xe1+1, ys1-1:ye1+1]    # east  face

    u[xs2-1:xe2+1, ye2] = 2*uwall - u[xs2-1:xe2+1, ye2+1]    # north face
    u[xs2-1:xe2+1, ys2] = 2*uwall - u[xs2-1:xe2+1, ys2-1]    # south face
    v[xs2, ys2-1:ye2+1] = 2*uwall - v[xs2-1, ys2-1:ye2+1]    # west  face
    v[xe2, ys2-1:ye2+1] = 2*uwall - v[xe2+1, ys2-1:ye2+1]    # east  face

    u[xs3-1:xe3+1, ye3] = 2*uwall - u[xs3-1:xe3+1, ye3+1]    # north face
    u[xs3-1:xe3+1, ys3] = 2*uwall - u[xs3-1:xe3+1, ys3-1]    # south face
    v[xs3, ys3-1:ye3+1] = 2*uwall - v[xs3-1, ys3-1:ye3+1]    # west  face
    v[xe3, ys3-1:ye3+1] = 2*uwall - v[xe3+1, ys3-1:ye3+1]    # east  face

    # Over-writing inlet conditions
    u[0, nin_start-1:nin_end] = u_in   # fixed inlet velocity

    # Over-writing outlet conditions
    u[-1, nout_start-1:nout_end] = u[-2, nout_start-1:nout_end]   # zero-gradient outlet velocity
    v[-1, nout_start-1:nout_end] = v[-2, nout_start-1:nout_end]   # zero-gradient outlet velocity

    # Advection-diffusion equation (predictor)
    ut, vt = AdvectionDiffusion2D(ut, vt, u, v, nx, ny, hx, hy, dt, nu, flagu, flagv, method)

    # Update boundary conditions for temporary velocities
    ut[0, nin_start-1:nin_end] = u_in       # fixed inlet velocity
    ut[-1, nout_start-1:nout_end] = u[-1, nout_start-1:nout_end]   # zero-gradient outlet velocity
    vt[-1, nout_start-1:nout_end] = v[-1, nout_start-1:nout_end]   # zero-gradient outlet velocity

    # Pressure equation (Poisson)
    p, iter = Pressure_Poisson(p, ut, vt, gamma, nx, ny, hx, hy, dt, beta, max_iterations, max_error, flagp)

    # Correction on the velocity
    u, v = correction_velocity(u, v, ut, vt, p, nx, ny, hx, hy, flagu, flagv)

    # Print on the screen
    if it % 500 == 1:
        print('Step:', it, '- Time:', t,  '- Poisson iterations:', iter)

    #----------------------------------------------------------------------------------#
    # 2. Transport of species                                                          #
    #----------------------------------------------------------------------------------#
    # Impermeable walls
    cO2 = SpeciesBCs(cO2, nx, ny, xs1, xe1, ys1, ye1, xs2, xe2, ys2, ye2, xs3, xe3, ys3, ye3)

    # Inlet sections
    cO2[0, nin_start:nin_end+1] = 2*cO2in - cO2[1, nin_start:nin_end+1]

    # Advection-Diffusion equation
    cO2 = AdvDiffSpecies(cO2, u, v, dt, hx, hy, Gamma, nx, ny, flagp, method)

    # Advance in time
    t = t + dt

## -------------------------------------------------------------------------------------------- ##
##                                    FINAL POST-PROCESSING                                     ##
## -------------------------------------------------------------------------------------------- ##
# Field reconstruction
uu   = node_interp(u, 'u', nx, ny, flagu)
vv   = node_interp(v, 'v', nx, ny, flagv)
pp   = node_interp(p, 'p', nx, ny, flagp)
ccO2 = node_interp(cO2, 'p', nx, ny, flagp)

uu[xs1-1:xe1+1, ys1-1:ye1+1] = 0         # Obstacle 1
vv[xs1-1:xe1+1, ys1-1:ye1+1] = 0
pp[xs1-1:xe1+1, ys1-1:ye1+1] = 0
ccO2[xs1-1:xe1+1, ys1-1:ye1+1] = 0
uu[xs2-1:xe2+1, ys2-1:ye2+1] = 0         # Obstacle 2
vv[xs2-1:xe2+1, ys2-1:ye2+1] = 0
pp[xs2-1:xe2+1, ys2-1:ye2+1] = 0
ccO2[xs2-1:xe2+1, ys2-1:ye2+1] = 0
uu[xs3-1:xe3+1, ys3-1:ye3+1] = 0         # Obstacle 3
vv[xs3-1:xe3+1, ys3-1:ye3+1] = 0
pp[xs3-1:xe3+1, ys3-1:ye3+1] = 0
ccO2[xs3-1:xe3+1, ys3-1:ye3+1] = 0

# Creating a grid
xx,yy = np.meshgrid(x,y)

# Plotting the results
# Surface map: pressure
fig, ax = plt.subplots()
plt1 = plt.contourf(xx, yy, np.transpose(pp))
plt.colorbar(plt1)
ax.add_patch(Rectangle((x[xs1-1], y[ys1-1]), x[xe1-xs1+1], y[ye1-ys1+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs2-1], y[ys2-1]), x[xe2-xs2+1], y[ye2-ys2+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs3-1], y[ys3-1]), x[xe3-xs3+1], y[ye3-ys3+1], facecolor = 'grey'))
plt.title('Relative pressure [Pa]')     # Trovare un modo per portarla a P assoluta!!!!!
plt.xlabel('x [m]')
plt.ylabel('y [m]')

# Surface map: u-velocity
fig, ax = plt.subplots()
plt2 = plt.contourf(xx, yy, np.transpose(uu))
plt.colorbar(plt2)
ax.add_patch(Rectangle((x[xs1-1], y[ys1-1]), x[xe1-xs1+1], y[ye1-ys1+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs2-1], y[ys2-1]), x[xe2-xs2+1], y[ye2-ys2+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs3-1], y[ys3-1]), x[xe3-xs3+1], y[ye3-ys3+1], facecolor = 'grey'))
plt.title('u - velocity [m/s]')
plt.xlabel('x [m]')
plt.ylabel('y [m]')

# Surface map: v-velocity
fig, ax = plt.subplots()
plt3 = plt.contourf(xx, yy, np.transpose(vv))
plt.colorbar(plt3)
ax.add_patch(Rectangle((x[xs1-1], y[ys1-1]), x[xe1-xs1+1], y[ye1-ys1+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs2-1], y[ys2-1]), x[xe2-xs2+1], y[ye2-ys2+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs3-1], y[ys3-1]), x[xe3-xs3+1], y[ye3-ys3+1], facecolor = 'grey'))
plt.title('v - velocity [m/s]')
plt.xlabel('x [m]')
plt.ylabel('y [m]')

# Surface map: cO2
fig, ax = plt.subplots()
plt3 = plt.contourf(xx, yy, np.transpose(ccO2))
plt.colorbar(plt3)
ax.add_patch(Rectangle((x[xs1-1], y[ys1-1]), x[xe1-xs1+1], y[ye1-ys1+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs2-1], y[ys2-1]), x[xe2-xs2+1], y[ye2-ys2+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs3-1], y[ys3-1]), x[xe3-xs3+1], y[ye3-ys3+1], facecolor = 'grey'))
plt.title('O2 concentration [kmol/m3]')
plt.xlabel('x [m]')
plt.ylabel('y [m]')

# Streamlines
fig, ax = plt.subplots()
plt.quiver(xx[::10], yy[::10], np.transpose(uu)[::10], np.transpose(vv)[::10])
plt.streamplot(xx, yy, np.transpose(uu), np.transpose(vv), linewidth = 1)
ax.add_patch(Rectangle((x[xs1-1], y[ys1-1]), x[xe1-xs1+1], y[ye1-ys1+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs2-1], y[ys2-1]), x[xe2-xs2+1], y[ye2-ys2+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs3-1], y[ys3-1]), x[xe3-xs3+1], y[ye3-ys3+1], facecolor = 'grey'))
plt.title('Streamlines')
plt.xlabel('x [m]')
plt.ylabel('y [m]')

plt.show()