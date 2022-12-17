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
P = 101325.0    # Pa
T = 25+273.15   # K

Lx = 3.0        # m
Ly = 1.0        # m
nx = 500
ny = 150
nu = 5e-5       # m^2/s  (diffusion coeff)
u_in = 1e-3     # m/s    (inlet velocity)
tau = 20.0       # s      (simulation time)

# Boundary conditions (no-slip)
un = 0          # m/s
us = 0          # m/s
ve = 0          # m/s
vw = 0          # m/s

# Parameters for SOR (Poisson eq)
max_iterations = 1000000
beta = 1.
max_error = 1e-6

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
x_obst1_start = 0.7
x_obst1_end = 0.8
y_obst1_start = 0.2
y_obst1_end = 0.6
# Obstacle 2
x_obst2_start = 1.4
x_obst2_end = 1.5
y_obst2_start = 0.4
y_obst2_end = 0.8
# Obstacle 3
x_obst3_start = 2.1
x_obst3_end = 2.2
y_obst3_start = 0.2
y_obst3_end = 0.6
# Obstacle 4
'''x_obst4_start = x_obst1_end
x_obst4_end = x_obst4_start + (x_obst2_start-x_obst1_end)/50
y_obst4_start = y_obst1_start
y_obst4_end = y_obst1_end/2
# Obstacle 5
x_obst5_start = x_obst4_end
x_obst5_end = x_obst5_start + (x_obst2_start-x_obst1_end)/50
y_obst5_start = y_obst1_start
y_obst5_end = y_obst4_end/2
# Obstacle 6
x_obst6_start = x_obst5_end
x_obst6_end = x_obst6_start + (x_obst2_start-x_obst1_end)/50
y_obst6_start = y_obst1_start
y_obst6_end = y_obst5_end/2
# Obstacle 7
x_obst7_start = x_obst6_end
x_obst7_end = x_obst7_start + (x_obst2_start-x_obst1_end)/50
y_obst7_start = y_obst1_start
y_obst7_end = y_obst6_end/2
# Obstacle 8
x_obst8_start = x_obst2_end
x_obst8_end = x_obst8_start + (x_obst3_start-x_obst2_end)/50
y_obst8_start = y_obst3_end + (y_obst2_end-y_obst3_end)/2
y_obst8_end = y_obst2_end
# Obstacle 9
x_obst9_start = x_obst8_end
x_obst9_end = x_obst9_start + (x_obst3_start-x_obst2_end)/50
y_obst9_start = y_obst8_start + (y_obst2_end-y_obst8_start)/2
y_obst9_end = y_obst2_end
# Obstacle 10
x_obst10_start = x_obst9_end
x_obst10_end = x_obst10_start + (x_obst3_start-x_obst2_end)/50
y_obst10_start = y_obst9_start + (y_obst2_end-y_obst9_start)/2
y_obst10_end = y_obst2_end'''

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

'''xs4 = np.where(x <= x_obst4_start)[0][-1] + 1
xe4 = np.where(x < x_obst4_end)[0][-1] + 1
ys4 = np.where(y <= y_obst4_start)[0][-1] + 1
ye4 = np.where(y < y_obst4_end)[0][-1] + 1

xs5 = np.where(x <= x_obst5_start)[0][-1] + 1
xe5 = np.where(x < x_obst5_end)[0][-1] + 1
ys5 = np.where(y <= y_obst5_start)[0][-1] + 1
ye5 = np.where(y < y_obst5_end)[0][-1] + 1

xs6 = np.where(x <= x_obst6_start)[0][-1] + 1
xe6 = np.where(x < x_obst6_end)[0][-1] + 1
ys6 = np.where(y <= y_obst6_start)[0][-1] + 1
ye6 = np.where(y < y_obst6_end)[0][-1] + 1

xs7 = np.where(x <= x_obst7_start)[0][-1] + 1
xe7 = np.where(x < x_obst7_end)[0][-1] + 1
ys7 = np.where(y <= y_obst7_start)[0][-1] + 1
ye7 = np.where(y < y_obst7_end)[0][-1] + 1

xs8 = np.where(x <= x_obst8_start)[0][-1] + 1
xe8 = np.where(x < x_obst8_end)[0][-1] + 1
ys8 = np.where(y <= y_obst8_start)[0][-1] + 1
ye8 = np.where(y < y_obst8_end)[0][-1] + 1

xs9 = np.where(x <= x_obst9_start)[0][-1] + 1
xe9 = np.where(x < x_obst9_end)[0][-1] + 1
ys9 = np.where(y <= y_obst9_start)[0][-1] + 1
ye9 = np.where(y < y_obst9_end)[0][-1] + 1

xs10 = np.where(x <= x_obst10_start)[0][-1] + 1
xe10 = np.where(x < x_obst10_end)[0][-1] + 1
ys10 = np.where(y <= y_obst10_start)[0][-1] + 1
ye10 = np.where(y < y_obst10_end)[0][-1] + 1'''

# Inlet section (west side)
nin_start = 1        # first cell index (pay attention!)
nin_end = ny + 1     # last cell index (pay attention!)

# Outlet section (east side)
nout_start = 2        # first cell index (pay attention!)
nout_end = ny + 1     # last cell index (pay attention!)

# Time step
sigma = 0.5                              # safety factor for time step (stability)
dt_diff = np.minimum(hx,hy)**2/4/nu      # time step (diffusion stability) [s]
dt_conv = 4*nu/u_in**2                   # time step (convection stability) [s]
dt = sigma*np.minimum(dt_diff, dt_conv)  # time step (stability) [s]
nsteps = int(tau/dt)                     # number of steps
Re = u_in*Ly/nu                          # Reynolds' number

print('Time step:', dt, 's' )
print(' - Diffusion:', dt_diff, 's')
print(' - Convection:', dt_conv, 's')
print('Reynolds number:', Re, '\n')

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

'''gamma[xs4-1, ys4:ye4+1] = hx*hy / (2*hx**2 +   hy**2)         # Obstacle 4
gamma[xe4+1, ys4:ye4+1] = hx*hy / (2*hx**2 +   hy**2)
gamma[xs4:xe4+1, ys4-1] = hx*hy / (  hx**2 + 2*hy**2)
gamma[xs4:xe4+1, ye4+1] = hx*hy / (  hx**2 + 2*hy**2)

gamma[xs5-1, ys5:ye5+1] = hx*hy / (2*hx**2 +   hy**2)         # Obstacle 5
gamma[xe5+1, ys5:ye5+1] = hx*hy / (2*hx**2 +   hy**2)
gamma[xs5:xe5+1, ys5-1] = hx*hy / (  hx**2 + 2*hy**2)
gamma[xs5:xe5+1, ye5+1] = hx*hy / (  hx**2 + 2*hy**2)

gamma[xs6-1, ys6:ye6+1] = hx*hy / (2*hx**2 +   hy**2)         # Obstacle 6
gamma[xe6+1, ys6:ye6+1] = hx*hy / (2*hx**2 +   hy**2)
gamma[xs6:xe6+1, ys6-1] = hx*hy / (  hx**2 + 2*hy**2)
gamma[xs6:xe6+1, ye6+1] = hx*hy / (  hx**2 + 2*hy**2)

gamma[xs7-1, ys7:ye7+1] = hx*hy / (2*hx**2 +   hy**2)         # Obstacle 7
gamma[xe7+1, ys7:ye7+1] = hx*hy / (2*hx**2 +   hy**2)
gamma[xs7:xe7+1, ys7-1] = hx*hy / (  hx**2 + 2*hy**2)
gamma[xs7:xe7+1, ye7+1] = hx*hy / (  hx**2 + 2*hy**2)

gamma[xs8-1, ys8:ye8+1] = hx*hy / (2*hx**2 +   hy**2)         # Obstacle 8
gamma[xe8+1, ys8:ye8+1] = hx*hy / (2*hx**2 +   hy**2)
gamma[xs8:xe8+1, ys8-1] = hx*hy / (  hx**2 + 2*hy**2)
gamma[xs8:xe8+1, ye8+1] = hx*hy / (  hx**2 + 2*hy**2)

gamma[xs9-1, ys9:ye9+1] = hx*hy / (2*hx**2 +   hy**2)         # Obstacle 9
gamma[xe9+1, ys9:ye9+1] = hx*hy / (2*hx**2 +   hy**2)
gamma[xs9:xe9+1, ys9-1] = hx*hy / (  hx**2 + 2*hy**2)
gamma[xs9:xe9+1, ye9+1] = hx*hy / (  hx**2 + 2*hy**2)

gamma[xs10-1, ys10:ye10+1] = hx*hy / (2*hx**2 +   hy**2)         # Obstacle 10
gamma[xe10+1, ys10:ye10+1] = hx*hy / (2*hx**2 +   hy**2)
gamma[xs10:xe10+1, ys10-1] = hx*hy / (  hx**2 + 2*hy**2)
gamma[xs10:xe10+1, ye10+1] = hx*hy / (  hx**2 + 2*hy**2)'''

# Correction of gamma close to the obstacles edges
'''gamma[xs1-1, ys1] = hx*hy / (hx**2 + hy**2)

gamma[xs2-1, ye2] = hx*hy / (hx**2 + hy**2)

gamma[xs3-1, ys3] = hx*hy / (hx**2 + hy**2)
gamma[xe3+1, ys3] = hx*hy / (hx**2 + hy**2)

gamma[xe1+1, ye4+1] = hx*hy / (hx**2 + hy**2)

gamma[xe4+1, ye5+1] = hx*hy / (hx**2 + hy**2)

gamma[xe5+1, ye6+1] = hx*hy / (hx**2 + hy**2)

gamma[xe6+1, ye7+1] = hx*hy / (hx**2 + hy**2)
gamma[xe7+1, ys7] = hx*hy / (hx**2 + hy**2)

gamma[xe2+1, ys8-1] = hx*hy / (hx**2 + hy**2)

gamma[xe8+1, ys9-1] = hx*hy / (hx**2 + hy**2)

gamma[xe9+1, ys10-1] = hx*hy / (hx**2 + hy**2)
gamma[xe10+1, ye2] = hx*hy / (hx**2 + hy**2)'''

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

'''flagu[xs4-1:xe4+1, ys4:ye4+1] = 1
flagv[xs4:xe4+1, ys4-1:ye4+1] = 1
flagp[xs4:xe4+1, ys4:ye4+1] = 1

flagu[xs5-1:xe5+1, ys5:ye5+1] = 1
flagv[xs5:xe5+1, ys5-1:ye5+1] = 1
flagp[xs5:xe5+1, ys5:ye5+1] = 1

flagu[xs6-1:xe6+1, ys6:ye6+1] = 1
flagv[xs6:xe6+1, ys6-1:ye6+1] = 1
flagp[xs6:xe6+1, ys6:ye6+1] = 1

flagu[xs7-1:xe7+1, ys7:ye7+1] = 1
flagv[xs7:xe7+1, ys7-1:ye7+1] = 1
flagp[xs7:xe7+1, ys7:ye7+1] = 1

flagu[xs8-1:xe8+1, ys8:ye8+1] = 1
flagv[xs8:xe8+1, ys8-1:ye8+1] = 1
flagp[xs8:xe8+1, ys8:ye8+1] = 1

flagu[xs9-1:xe9+1, ys9:ye9+1] = 1
flagv[xs9:xe9+1, ys9-1:ye9+1] = 1
flagp[xs9:xe9+1, ys9:ye9+1] = 1

flagu[xs10-1:xe10+1, ys10:ye10+1] = 1
flagv[xs10:xe10+1, ys10-1:ye10+1] = 1
flagp[xs10:xe10+1, ys10:ye10+1] = 1'''

# Initial conditions: set reasonable initial velocity value instead of initializing everything to zero
u[:, :] = u_in                     # Internal points: fixed velocity [m/s]
u[xs1-1:xe1+1, ys1:ye1+1] = 0      # Obstacle 1
u[xs2-1:xe2+1, ys2:ye2+1] = 0      # Obstacle 2
u[xs3-1:xe3+1, ys3:ye3+1] = 0      # Obstacle 3
'''u[xs4-1:xe4+1, ys4:ye4+1] = 0      # Obstacle 4
u[xs5-1:xe5+1, ys5:ye5+1] = 0      # Obstacle 5
u[xs6-1:xe6+1, ys6:ye6+1] = 0      # Obstacle 6
u[xs7-1:xe7+1, ys7:ye7+1] = 0      # Obstacle 7
u[xs8-1:xe8+1, ys8:ye8+1] = 0      # Obstacle 8
u[xs9-1:xe9+1, ys9:ye9+1] = 0      # Obstacle 7
u[xs10-1:xe10+1, ys10:ye10+1] = 0      # Obstacle 8'''
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

    '''u[xs4-1:xe4+1, ye4] = 2*uwall - u[xs4-1:xe4+1, ye4+1]    # north face
    u[xs4-1:xe4+1, ys4] = 2*uwall - u[xs4-1:xe4+1, ys4-1]    # south face
    v[xs4, ys4-1:ye4+1] = 2*uwall - v[xs4-1, ys4-1:ye4+1]    # west  face
    v[xe4, ys4-1:ye4+1] = 2*uwall - v[xe4+1, ys4-1:ye4+1]    # east  face

    u[xs5-1:xe5+1, ye5] = 2*uwall - u[xs5-1:xe5+1, ye5+1]    # north face
    u[xs5-1:xe5+1, ys5] = 2*uwall - u[xs5-1:xe5+1, ys5-1]    # south face
    v[xs5, ys5-1:ye5+1] = 2*uwall - v[xs5-1, ys5-1:ye5+1]    # west  face
    v[xe5, ys5-1:ye5+1] = 2*uwall - v[xe5+1, ys5-1:ye5+1]    # east  face

    u[xs6-1:xe6+1, ye6] = 2*uwall - u[xs6-1:xe6+1, ye6+1]    # north face
    u[xs6-1:xe6+1, ys6] = 2*uwall - u[xs6-1:xe6+1, ys6-1]    # south face
    v[xs6, ys6-1:ye6+1] = 2*uwall - v[xs6-1, ys6-1:ye6+1]    # west  face
    v[xe6, ys6-1:ye6+1] = 2*uwall - v[xe6+1, ys6-1:ye6+1]    # east  face

    u[xs7-1:xe7+1, ye7] = 2*uwall - u[xs7-1:xe7+1, ye7+1]    # north face
    u[xs7-1:xe7+1, ys7] = 2*uwall - u[xs7-1:xe7+1, ys7-1]    # south face
    v[xs7, ys7-1:ye7+1] = 2*uwall - v[xs7-1, ys7-1:ye7+1]    # west  face
    v[xe7, ys7-1:ye7+1] = 2*uwall - v[xe7+1, ys7-1:ye7+1]    # east  face

    u[xs8-1:xe8+1, ye8] = 2*uwall - u[xs8-1:xe8+1, ye8+1]    # north face
    u[xs8-1:xe8+1, ys8] = 2*uwall - u[xs8-1:xe8+1, ys8-1]    # south face
    v[xs8, ys8-1:ye8+1] = 2*uwall - v[xs8-1, ys8-1:ye8+1]    # west  face
    v[xe8, ys8-1:ye8+1] = 2*uwall - v[xe8+1, ys8-1:ye8+1]    # east  face

    u[xs9-1:xe9+1, ye9] = 2*uwall - u[xs9-1:xe9+1, ye9+1]    # north face
    u[xs9-1:xe9+1, ys9] = 2*uwall - u[xs9-1:xe9+1, ys9-1]    # south face
    v[xs9, ys9-1:ye9+1] = 2*uwall - v[xs9-1, ys9-1:ye9+1]    # west  face
    v[xe9, ys9-1:ye9+1] = 2*uwall - v[xe9+1, ys9-1:ye9+1]    # east  face

    u[xs10-1:xe10+1, ye10] = 2*uwall - u[xs10-1:xe10+1, ye10+1]    # north face
    u[xs10-1:xe10+1, ys10] = 2*uwall - u[xs10-1:xe10+1, ys10-1]    # south face
    v[xs10, ys10-1:ye10+1] = 2*uwall - v[xs10-1, ys10-1:ye10+1]    # west  face
    v[xe10, ys10-1:ye10+1] = 2*uwall - v[xe10+1, ys10-1:ye10+1]    # east  face'''

    # Over-writing inlet conditions
    u[0, nin_start-1:nin_end] = u_in   # fixed inlet velocity

    # Over-writing outlet conditions
    u[-1, nout_start-1:nout_end] = u[-2, nout_start-1:nout_end]   # zero-gradient outlet velocity
    v[-1, nout_start-1:nout_end] = v[-2, nout_start-1:nout_end]   # zero-gradient outlet velocity

    # Advection-diffusion equation (predictor)
    ut, vt = AdvectionDiffusion2D(ut, vt, u, v, nx, ny, hx, hy, dt, nu, flagu, flagv, 'Upwind')

    # Update boundary conditions for temporary velocities
    ut[0, nin_start-1:nin_end] = u_in       # fixed inlet velocity
    ut[-1, nout_start-1:nout_end] = u[-1, nout_start-1:nout_end]   # zero-gradient outlet velocity
    vt[-1, nout_start-1:nout_end] = v[-1, nout_start-1:nout_end]   # zero-gradient outlet velocity

    # Pressure equation (Poisson)
    p, iter = Pressure_Poisson(p, ut, vt, gamma, nx, ny, hx, hy, dt, beta, max_iterations, max_error, flagp)

    # Correction on the velocity
    u, v = correction_velocity(u, v, ut, vt, p, nx, ny, hx, hy, flagu, flagv)

    # Print on the screen
    if it % 20 == 1:
        print('Step:', it, '- Time:', t,  '- Poisson iterations:', iter)

    # Advance in time
    t = t + dt

## -------------------------------------------------------------------------------------------- ##
##                                    FINAL POST-PROCESSING                                     ##
## -------------------------------------------------------------------------------------------- ##
# Field reconstruction
uu = node_interp(u, 'u', nx, ny, flagu)
vv = node_interp(v, 'v', nx, ny, flagv)
pp = node_interp(p, 'p', nx, ny, flagp)

uu[xs1-1:xe1+1, ys1-1:ye1+1] = 0         # Obstacle 1
vv[xs1-1:xe1+1, ys1-1:ye1+1] = 0
pp[xs1-1:xe1+1, ys1-1:ye1+1] = 0
uu[xs2-1:xe2+1, ys2-1:ye2+1] = 0         # Obstacle 2
vv[xs2-1:xe2+1, ys2-1:ye2+1] = 0
pp[xs2-1:xe2+1, ys2-1:ye2+1] = 0
uu[xs3-1:xe3+1, ys3-1:ye3+1] = 0         # Obstacle 3
vv[xs3-1:xe3+1, ys3-1:ye3+1] = 0
pp[xs3-1:xe3+1, ys3-1:ye3+1] = 0
'''uu[xs4-1:xe4+1, ys4-1:ye4+1] = 0         # Obstacle 4
vv[xs4-1:xe4+1, ys4-1:ye4+1] = 0
pp[xs4-1:xe4+1, ys4-1:ye4+1] = 0
uu[xs5-1:xe5+1, ys5-1:ye5+1] = 0         # Obstacle 5
vv[xs5-1:xe5+1, ys5-1:ye5+1] = 0
pp[xs5-1:xe5+1, ys5-1:ye5+1] = 0
uu[xs6-1:xe6+1, ys6-1:ye6+1] = 0         # Obstacle 6
vv[xs6-1:xe6+1, ys6-1:ye6+1] = 0
pp[xs6-1:xe6+1, ys6-1:ye6+1] = 0
uu[xs7-1:xe7+1, ys7-1:ye7+1] = 0         # Obstacle 7
vv[xs7-1:xe7+1, ys7-1:ye7+1] = 0
pp[xs7-1:xe7+1, ys7-1:ye7+1] = 0
uu[xs8-1:xe8+1, ys8-1:ye8+1] = 0         # Obstacle 8
vv[xs8-1:xe8+1, ys8-1:ye8+1] = 0
pp[xs8-1:xe8+1, ys8-1:ye8+1] = 0
uu[xs9-1:xe9+1, ys9-1:ye9+1] = 0         # Obstacle 9
vv[xs9-1:xe9+1, ys9-1:ye9+1] = 0
pp[xs9-1:xe9+1, ys9-1:ye9+1] = 0
uu[xs10-1:xe10+1, ys10-1:ye10+1] = 0         # Obstacle 10
vv[xs10-1:xe10+1, ys10-1:ye10+1] = 0
pp[xs10-1:xe10+1, ys10-1:ye10+1] = 0'''

# Creating a grid
xx,yy = np.meshgrid(x,y)

# Plotting the results
# fig, = plt.figure(figsize=[1,1],dpi=100)
fig, ax = plt.subplots()
plt1 = plt.contourf(xx, yy, np.transpose(pp))
plt.colorbar(plt1)
ax.add_patch(Rectangle((x[xs1-1], y[ys1-1]), x[xe1-xs1+1], y[ye1-ys1+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs2-1], y[ys2-1]), x[xe2-xs2+1], y[ye2-ys2+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs3-1], y[ys3-1]), x[xe3-xs3+1], y[ye3-ys3+1], facecolor = 'grey'))
'''ax.add_patch(Rectangle((x[xs4-1], y[ys4-1]), x[xe4-xs4+1], y[ye4-ys4+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs5-1], y[ys5-1]), x[xe5-xs5+1], y[ye5-ys5+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs6-1], y[ys6-1]), x[xe6-xs6+1], y[ye6-ys6+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs7-1], y[ys7-1]), x[xe7-xs7+1], y[ye7-ys7+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs8-1], y[ys8-1]), x[xe8-xs8+1], y[ye8-ys8+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs9-1], y[ys9-1]), x[xe9-xs9+1], y[ye9-ys9+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs10-1], y[ys10-1]), x[xe10-xs10+1], y[ye10-ys10+1], facecolor = 'grey'))'''
plt.title('Relative pressure [Pa]')     # Trovare un modo per portarla a P assoluta!!!!!
plt.xlabel('x [m]')
plt.ylabel('y [m]')

fig, ax = plt.subplots()
plt2 = plt.contourf(xx, yy, np.transpose(uu))
plt.colorbar(plt2)
ax.add_patch(Rectangle((x[xs1-1], y[ys1-1]), x[xe1-xs1+1], y[ye1-ys1+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs2-1], y[ys2-1]), x[xe2-xs2+1], y[ye2-ys2+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs3-1], y[ys3-1]), x[xe3-xs3+1], y[ye3-ys3+1], facecolor = 'grey'))
'''ax.add_patch(Rectangle((x[xs4-1], y[ys4-1]), x[xe4-xs4+1], y[ye4-ys4+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs5-1], y[ys5-1]), x[xe5-xs5+1], y[ye5-ys5+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs6-1], y[ys6-1]), x[xe6-xs6+1], y[ye6-ys6+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs7-1], y[ys7-1]), x[xe7-xs7+1], y[ye7-ys7+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs8-1], y[ys8-1]), x[xe8-xs8+1], y[ye8-ys8+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs9-1], y[ys9-1]), x[xe9-xs9+1], y[ye9-ys9+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs10-1], y[ys10-1]), x[xe10-xs10+1], y[ye10-ys10+1], facecolor = 'grey'))'''
plt.title('u - velocity [m/s]')
plt.xlabel('x [m]')
plt.ylabel('y [m]')

fig, ax = plt.subplots()
plt3 = plt.contourf(xx, yy, np.transpose(vv))
plt.colorbar(plt3)
ax.add_patch(Rectangle((x[xs1-1], y[ys1-1]), x[xe1-xs1+1], y[ye1-ys1+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs2-1], y[ys2-1]), x[xe2-xs2+1], y[ye2-ys2+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs3-1], y[ys3-1]), x[xe3-xs3+1], y[ye3-ys3+1], facecolor = 'grey'))
'''ax.add_patch(Rectangle((x[xs4-1], y[ys4-1]), x[xe4-xs4+1], y[ye4-ys4+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs5-1], y[ys5-1]), x[xe5-xs5+1], y[ye5-ys5+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs6-1], y[ys6-1]), x[xe6-xs6+1], y[ye6-ys6+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs7-1], y[ys7-1]), x[xe7-xs7+1], y[ye7-ys7+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs8-1], y[ys8-1]), x[xe8-xs8+1], y[ye8-ys8+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs9-1], y[ys9-1]), x[xe9-xs9+1], y[ye9-ys9+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs10-1], y[ys10-1]), x[xe10-xs10+1], y[ye10-ys10+1], facecolor = 'grey'))'''
plt.title('v - velocity [m/s]')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlim(0,Lx)
plt.ylim(0,Ly)

fig, ax = plt.subplots()
plt.streamplot(xx, yy, np.transpose(uu), np.transpose(vv), linewidth = 1)
ax.add_patch(Rectangle((x[xs1-1], y[ys1-1]), x[xe1-xs1+1], y[ye1-ys1+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs2-1], y[ys2-1]), x[xe2-xs2+1], y[ye2-ys2+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs3-1], y[ys3-1]), x[xe3-xs3+1], y[ye3-ys3+1], facecolor = 'grey'))
'''ax.add_patch(Rectangle((x[xs4-1], y[ys4-1]), x[xe4-xs4+1], y[ye4-ys4+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs5-1], y[ys5-1]), x[xe5-xs5+1], y[ye5-ys5+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs6-1], y[ys6-1]), x[xe6-xs6+1], y[ye6-ys6+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs7-1], y[ys7-1]), x[xe7-xs7+1], y[ye7-ys7+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs8-1], y[ys8-1]), x[xe8-xs8+1], y[ye8-ys8+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs9-1], y[ys9-1]), x[xe9-xs9+1], y[ye9-ys9+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs10-1], y[ys10-1]), x[xe10-xs10+1], y[ye10-ys10+1], facecolor = 'grey'))'''
plt.title('Streamline')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlim(0,Lx)
plt.ylim(0,Ly)

plt.show()