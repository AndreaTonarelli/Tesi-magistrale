import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from   matplotlib import cm
from   matplotlib.patches import Rectangle
from   numba import njit
from   functions import *
plt.style.use(['science', 'no-latex'])

## -------------------------------------------------------------------------------------------- ##
##                                     PRE-PROCESSING                                           ##
## -------------------------------------------------------------------------------------------- ##
# Data
P = 101325.0       # Pa
T = 25+273.15      # K

Lx = 3.0           # m
Ly = 1.0           # m
nx = 150
ny = 50
nu = 1e-6          # m2/s  (dynamic viscosity of fluid)
Gamma = 1e-3       # m2/s  (diffusion coefficient of O2 in water)
u_in = 0.1         # m/s   (inlet velocity)
tau = 20.0         # s     (simulation time)
method = 'Upwind'  # Discretization method (CDS or Upwind)

# Boundary conditions (no-slip)
un = 0             # m/s
us = 0             # m/s
ve = 0             # m/s
vw = 0             # m/s

# Inlet concentrations
cO2in = 0.1       # kmol/m3

# Immobilized cells parameters
cB0 = 0.01        # kmol/m3
thick = 0.05      # m (thickness of cells layer)
k = 1             # m3/kmol/s (reaction constant) -> r = k*cO2*cB

# Parameters for SOR (Poisson eq)
max_iterations = 1000000
beta = 1.85
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

# Find cells corresponding to thickness
th = int(np.ceil(thick/hx))    # it finds the index not the positions!

# Obstacle coordinates
# Obstacle 1
x_obst1_start = 0.3
x_obst1_end = 0.4
y_obst1_start = 0.
y_obst1_end = 0.5
# Obstacle 2
x_obst2_start = 0.6
x_obst2_end = 0.7
y_obst2_start = 0.5
y_obst2_end = 1
# Obstacle 3
x_obst3_start = 0.9
x_obst3_end = 1.0
y_obst3_start = 0.
y_obst3_end = 0.5
# Obstacle 4
x_obst4_start = 1.2
x_obst4_end = 1.3
y_obst4_start = 0.5
y_obst4_end = 1.0
# Obstacle 5
x_obst5_start = 1.5
x_obst5_end = 1.6
y_obst5_start = 0.
y_obst5_end = 0.5
# Obstacle 6
x_obst6_start = 1.8
x_obst6_end = 1.9
y_obst6_start = 0.5
y_obst6_end = 1.0
# Obstacle 7
x_obst7_start = 2.1
x_obst7_end = 2.2
y_obst7_start = 0.
y_obst7_end = 0.5
# Obstacle 8
x_obst8_start = 2.4
x_obst8_end = 2.5
y_obst8_start = 0.5
y_obst8_end = 1.0
# Obstacle 9
x_obst9_start = 2.7
x_obst9_end = 2.8
y_obst9_start = 0.
y_obst9_end = 0.5

# Obstacles definition: rectangle with base xs:xe and height ys:ye
# Obstacle 1
xs1 = np.where(x <= x_obst1_start)[0][-1] + 1
xe1 = np.where(x < x_obst1_end)[0][-1] + 1
ys1 = np.where(y <= y_obst1_start)[0][-1] + 1
ye1 = np.where(y < y_obst1_end)[0][-1] + 1
# Obstacle 2
xs2 = np.where(x <= x_obst2_start)[0][-1] + 1
xe2 = np.where(x < x_obst2_end)[0][-1] + 1
ys2 = np.where(y <= y_obst2_start)[0][-1] + 1
ye2 = np.where(y < y_obst2_end)[0][-1] + 1
# Obstacle 3
xs3 = np.where(x <= x_obst3_start)[0][-1] + 1
xe3 = np.where(x < x_obst3_end)[0][-1] + 1
ys3 = np.where(y <= y_obst3_start)[0][-1] + 1
ye3 = np.where(y < y_obst3_end)[0][-1] + 1
# Obstacle 4
xs4 = np.where(x <= x_obst4_start)[0][-1] + 1
xe4 = np.where(x < x_obst4_end)[0][-1] + 1
ys4 = np.where(y <= y_obst4_start)[0][-1] + 1
ye4 = np.where(y < y_obst4_end)[0][-1] + 1
# Obstacle 5
xs5 = np.where(x <= x_obst5_start)[0][-1] + 1
xe5 = np.where(x < x_obst5_end)[0][-1] + 1
ys5 = np.where(y <= y_obst5_start)[0][-1] + 1
ye5 = np.where(y < y_obst5_end)[0][-1] + 1
# Obstacle 6
xs6 = np.where(x <= x_obst6_start)[0][-1] + 1
xe6 = np.where(x < x_obst6_end)[0][-1] + 1
ys6 = np.where(y <= y_obst6_start)[0][-1] + 1
ye6 = np.where(y < y_obst6_end)[0][-1] + 1
# Obstacle 7
xs7 = np.where(x <= x_obst7_start)[0][-1] + 1
xe7 = np.where(x < x_obst7_end)[0][-1] + 1
ys7 = np.where(y <= y_obst7_start)[0][-1] + 1
ye7 = np.where(y < y_obst7_end)[0][-1] + 1
# Obstacle 8
xs8 = np.where(x <= x_obst8_start)[0][-1] + 1
xe8 = np.where(x < x_obst8_end)[0][-1] + 1
ys8 = np.where(y <= y_obst8_start)[0][-1] + 1
ye8 = np.where(y < y_obst8_end)[0][-1] + 1
# Obstacle 9
xs9 = np.where(x <= x_obst9_start)[0][-1] + 1
xe9 = np.where(x < x_obst9_end)[0][-1] + 1
ys9 = np.where(y <= y_obst9_start)[0][-1] + 1
ye9 = np.where(y < y_obst9_end)[0][-1] + 1

# Inlet section (west side)
nin_start = 2        # first cell index (pay attention cause it is in MATLAB notation!)
nin_end = ny + 1     # last cell index (pay attention cause it is in MATLAB notation!)

# Outlet section (east side)
nout_start = 2       # first cell index (pay attention cause it is in MATLAB notation!)
nout_end = ny+1      # last cell index (pay attention cause it is in MATLAB notation!)

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
cB  = np.zeros([nx+2, ny+2])

# Temporary velocity fields
ut = np.zeros_like(u)
vt = np.zeros_like(v)

# Fields used only for graphical purposes
uu   = np.zeros([nx+1, ny+1])
vv   = np.zeros([nx+1, ny+1])
pp   = np.zeros([nx+1, ny+1])
ccO2 = np.zeros([nx+1, ny+1])
ccB  = np.zeros([nx+1, ny+1])

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
# Obstacle 1
gamma[xs1-1, ys1:ye1+1] = hx*hy / (2*hx**2 +   hy**2)
gamma[xe1+1, ys1:ye1+1] = hx*hy / (2*hx**2 +   hy**2)
gamma[xs1:xe1+1, ys1-1] = hx*hy / (  hx**2 + 2*hy**2)
gamma[xs1:xe1+1, ye1+1] = hx*hy / (  hx**2 + 2*hy**2)
# Obstacle 2
gamma[xs2-1, ys2:ye2+1] = hx*hy / (2*hx**2 +   hy**2)
gamma[xe2+1, ys2:ye2+1] = hx*hy / (2*hx**2 +   hy**2)
gamma[xs2:xe2+1, ys2-1] = hx*hy / (  hx**2 + 2*hy**2)
gamma[xs2:xe2+1, ye2+1] = hx*hy / (  hx**2 + 2*hy**2)
# Obstacle 3
gamma[xs3-1, ys3:ye3+1] = hx*hy / (2*hx**2 +   hy**2)
gamma[xe3+1, ys3:ye3+1] = hx*hy / (2*hx**2 +   hy**2)
gamma[xs3:xe3+1, ys3-1] = hx*hy / (  hx**2 + 2*hy**2)
gamma[xs3:xe3+1, ye3+1] = hx*hy / (  hx**2 + 2*hy**2)
# Obstacle 4
gamma[xs4-1, ys4:ye4+1] = hx*hy / (2*hx**2 +   hy**2)
gamma[xe4+1, ys4:ye4+1] = hx*hy / (2*hx**2 +   hy**2)
gamma[xs4:xe4+1, ys4-1] = hx*hy / (  hx**2 + 2*hy**2)
gamma[xs4:xe4+1, ye4+1] = hx*hy / (  hx**2 + 2*hy**2)
# Obstacle 5
gamma[xs5-1, ys5:ye5+1] = hx*hy / (2*hx**2 +   hy**2)
gamma[xe5+1, ys5:ye5+1] = hx*hy / (2*hx**2 +   hy**2)
gamma[xs5:xe5+1, ys5-1] = hx*hy / (  hx**2 + 2*hy**2)
gamma[xs5:xe5+1, ye5+1] = hx*hy / (  hx**2 + 2*hy**2)
# Obstacle 6
gamma[xs6-1, ys6:ye6+1] = hx*hy / (2*hx**2 +   hy**2)
gamma[xe6+1, ys6:ye6+1] = hx*hy / (2*hx**2 +   hy**2)
gamma[xs6:xe6+1, ys6-1] = hx*hy / (  hx**2 + 2*hy**2)
gamma[xs6:xe6+1, ye6+1] = hx*hy / (  hx**2 + 2*hy**2)
# Obstacle 7
gamma[xs7-1, ys7:ye7+1] = hx*hy / (2*hx**2 +   hy**2)
gamma[xe7+1, ys7:ye7+1] = hx*hy / (2*hx**2 +   hy**2)
gamma[xs7:xe7+1, ys7-1] = hx*hy / (  hx**2 + 2*hy**2)
gamma[xs7:xe7+1, ye7+1] = hx*hy / (  hx**2 + 2*hy**2)
# Obstacle 8
gamma[xs8-1, ys8:ye8+1] = hx*hy / (2*hx**2 +   hy**2)
gamma[xe8+1, ys8:ye8+1] = hx*hy / (2*hx**2 +   hy**2)
gamma[xs8:xe8+1, ys8-1] = hx*hy / (  hx**2 + 2*hy**2)
gamma[xs8:xe8+1, ye8+1] = hx*hy / (  hx**2 + 2*hy**2)
# Obstacle 9
gamma[xs9-1, ys9:ye9+1] = hx*hy / (2*hx**2 +   hy**2)
gamma[xe9+1, ys9:ye9+1] = hx*hy / (2*hx**2 +   hy**2)
gamma[xs9:xe9+1, ys9-1] = hx*hy / (  hx**2 + 2*hy**2)
gamma[xs9:xe9+1, ye9+1] = hx*hy / (  hx**2 + 2*hy**2)

# Correction of gamma close to the obstacles edges
gamma[xs1-1, ys1] = hx*hy / (hx**2 + hy**2)          # Obstacle 1
gamma[xe1+1, ys1] = hx*hy / (hx**2 + hy**2)

gamma[xs2-1, ye2] = hx*hy / (hx**2 + hy**2)          # Obstacle 2
gamma[xe2+1, ye2] = hx*hy / (hx**2 + hy**2)

gamma[xs3-1, ys3] = hx*hy / (hx**2 + hy**2)          # Obstacle 3
gamma[xe3-1, ys3] = hx*hy / (hx**2 + hy**2)

gamma[xs4-1, ye4] = hx*hy / (hx**2 + hy**2)          # Obstacle 4
gamma[xe4-1, ye4] = hx*hy / (hx**2 + hy**2)

gamma[xs5-1, ys5] = hx*hy / (hx**2 + hy**2)          # Obstacle 5
gamma[xe5-1, ys5] = hx*hy / (hx**2 + hy**2)

gamma[xs6-1, ye6] = hx*hy / (hx**2 + hy**2)          # Obstacle 6
gamma[xe6-1, ye6] = hx*hy / (hx**2 + hy**2)

gamma[xs7-1, ys7] = hx*hy / (hx**2 + hy**2)          # Obstacle 7
gamma[xe7-1, ys7] = hx*hy / (hx**2 + hy**2)

gamma[xs8-1, ye8] = hx*hy / (hx**2 + hy**2)          # Obstacle 8
gamma[xe8-1, ye8] = hx*hy / (hx**2 + hy**2)

gamma[xs9-1, ys9] = hx*hy / (hx**2 + hy**2)          # Obstacle 9
gamma[xe9-1, ys9] = hx*hy / (hx**2 + hy**2)

# Flags to recognize where is necessary to solve the equations
# This flag is 1 in the cells that contain and obstacle and 0 in the others
flagu = np.zeros([nx+1, ny+2])          # u-cells corresponding to the obstacle
flagv = np.zeros([nx+2, ny+1])          # v-cells corresponding to the obstacle
flagp = np.zeros([nx+2, ny+2])          # p-cells corresponding to the obstacle

# Set flag to 1 in obstacle cells
flagu[xs1-1:xe1+1, ys1:ye1+1] = 1       # Obstacle 1
flagv[xs1:xe1+1, ys1-1:ye1+1] = 1
flagp[xs1:xe1+1, ys1:ye1+1] = 1

flagu[xs2-1:xe2+1, ys2:ye2+1] = 1       # Obstacle 2
flagv[xs2:xe2+1, ys2-1:ye2+1] = 1
flagp[xs2:xe2+1, ys2:ye2+1] = 1

flagu[xs3-1:xe3+1, ys3:ye3+1] = 1       # Obstacle 3
flagv[xs3:xe3+1, ys3-1:ye3+1] = 1
flagp[xs3:xe3+1, ys3:ye3+1] = 1

flagu[xs4-1:xe4+1, ys4:ye4+1] = 1       # Obstacle 4
flagv[xs4:xe4+1, ys4-1:ye4+1] = 1
flagp[xs4:xe4+1, ys4:ye4+1] = 1

flagu[xs5-1:xe5+1, ys5:ye5+1] = 1       # Obstacle 5
flagv[xs5:xe5+1, ys5-1:ye5+1] = 1
flagp[xs5:xe5+1, ys5:ye5+1] = 1

flagu[xs6-1:xe6+1, ys6:ye6+1] = 1       # Obstacle 6
flagv[xs6:xe6+1, ys6-1:ye6+1] = 1
flagp[xs6:xe6+1, ys6:ye6+1] = 1

flagu[xs7-1:xe7+1, ys7:ye7+1] = 1       # Obstacle 7
flagv[xs7:xe7+1, ys7-1:ye7+1] = 1
flagp[xs7:xe7+1, ys7:ye7+1] = 1

flagu[xs8-1:xe8+1, ys8:ye8+1] = 1       # Obstacle 8
flagv[xs8:xe8+1, ys8-1:ye8+1] = 1
flagp[xs8:xe8+1, ys8:ye8+1] = 1

flagu[xs9-1:xe9+1, ys9:ye9+1] = 1       # Obstacle 9
flagv[xs9:xe9+1, ys9-1:ye9+1] = 1
flagp[xs9:xe9+1, ys9:ye9+1] = 1

# Initial conditions: set reasonable initial velocity value instead of initializing everything to zero
u[:, :] = 0.5                      # Internal points: fixed velocity [m/s]
u[xs1-1:xe1+1, ys1:ye1+1] = 0      # Obstacle 1
u[xs2-1:xe2+1, ys2:ye2+1] = 0      # Obstacle 2
u[xs3-1:xe3+1, ys3:ye3+1] = 0      # Obstacle 3
u[xs4-1:xe4+1, ys4:ye4+1] = 0      # Obstacle 4
u[xs5-1:xe5+1, ys5:ye5+1] = 0      # Obstacle 5
u[xs6-1:xe6+1, ys6:ye6+1] = 0      # Obstacle 6
u[xs7-1:xe7+1, ys7:ye7+1] = 0      # Obstacle 7
u[xs8-1:xe8+1, ys8:ye8+1] = 0      # Obstacle 8
u[xs9-1:xe9+1, ys9:ye9+1] = 0      # Obstacle 9
ut = u

# Immobilized cells initialized over obstacles
cB[xs1-th:xs1, ys1:ye1] = cB0    # Obstacle 1
cB[xe1+1:xe1+th+1, ys1:ye1] = cB0
cB[xs2-th:xs2+1, ys2:ye2] = cB0    # Obstacle 2
cB[xe2+1:xe2+th+1, ys2:ye2] = cB0
cB[xs3-th:xs3, ys3:ye3] = cB0    # Obstacle 3
cB[xe3+1:xe3+th+1, ys3:ye3] = cB0
cB[xs4-th:xs4, ys4:ye4] = cB0    # Obstacle 4
cB[xe4+1:xe4+th+1, ys4:ye4] = cB0
cB[xs5-th:xs5, ys5:ye5] = cB0    # Obstacle 5
cB[xe5+1:xe5+th+1, ys5:ye5] = cB0
cB[xs6-th:xs6, ys6:ye6] = cB0    # Obstacle 6
cB[xe6+1:xe6+th+1, ys6:ye6] = cB0
cB[xs7-th:xs7, ys7:ye7] = cB0    # Obstacle 7
cB[xe7+1:xe7+th+1, ys7:ye7] = cB0
cB[xs8-th:xs8, ys8:ye8] = cB0    # Obstacle 8
cB[xe8+1:xe8+th+1, ys8:ye8] = cB0
cB[xs9-th:xs9, ys9:ye9] = cB0    # Obstacle 9
cB[xe9+1:xe9+th+1, ys9:ye9] = cB0

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
    # Obstacle 1
    u[xs1-1:xe1+1, ye1] = 2*uwall - u[xs1-1:xe1+1, ye1+1]    # north face
    u[xs1-1:xe1+1, ys1] = 2*uwall - u[xs1-1:xe1+1, ys1-1]    # south face
    v[xs1, ys1-1:ye1+1] = 2*uwall - v[xs1-1, ys1-1:ye1+1]    # west  face
    v[xe1, ys1-1:ye1+1] = 2*uwall - v[xe1+1, ys1-1:ye1+1]    # east  face
    # Obstacle 2
    u[xs2-1:xe2+1, ye2] = 2*uwall - u[xs2-1:xe2+1, ye2+1]    # north face
    u[xs2-1:xe2+1, ys2] = 2*uwall - u[xs2-1:xe2+1, ys2-1]    # south face
    v[xs2, ys2-1:ye2+1] = 2*uwall - v[xs2-1, ys2-1:ye2+1]    # west  face
    v[xe2, ys2-1:ye2+1] = 2*uwall - v[xe2+1, ys2-1:ye2+1]    # east  face
    # Obstacle 3
    u[xs3-1:xe3+1, ye3] = 2*uwall - u[xs3-1:xe3+1, ye3+1]    # north face
    u[xs3-1:xe3+1, ys3] = 2*uwall - u[xs3-1:xe3+1, ys3-1]    # south face
    v[xs3, ys3-1:ye3+1] = 2*uwall - v[xs3-1, ys3-1:ye3+1]    # west  face
    v[xe3, ys3-1:ye3+1] = 2*uwall - v[xe3+1, ys3-1:ye3+1]    # east  face
    # Obstacle 4
    u[xs4-1:xe4+1, ye4] = 2*uwall - u[xs4-1:xe4+1, ye4+1]    # north face
    u[xs4-1:xe4+1, ys4] = 2*uwall - u[xs4-1:xe4+1, ys4-1]    # south face
    v[xs4, ys4-1:ye4+1] = 2*uwall - v[xs4-1, ys4-1:ye4+1]    # west  face
    v[xe4, ys4-1:ye4+1] = 2*uwall - v[xe4+1, ys4-1:ye4+1]    # east  face
    # Obstacle 5
    u[xs5-1:xe5+1, ye5] = 2*uwall - u[xs5-1:xe5+1, ye5+1]    # north face
    u[xs5-1:xe5+1, ys5] = 2*uwall - u[xs5-1:xe5+1, ys5-1]    # south face
    v[xs5, ys5-1:ye5+1] = 2*uwall - v[xs5-1, ys5-1:ye5+1]    # west  face
    v[xe5, ys5-1:ye5+1] = 2*uwall - v[xe5+1, ys5-1:ye5+1]    # east  face
    # Obstacle 6
    u[xs6-1:xe6+1, ye6] = 2*uwall - u[xs6-1:xe6+1, ye6+1]    # north face
    u[xs6-1:xe6+1, ys6] = 2*uwall - u[xs6-1:xe6+1, ys6-1]    # south face
    v[xs6, ys6-1:ye6+1] = 2*uwall - v[xs6-1, ys6-1:ye6+1]    # west  face
    v[xe6, ys6-1:ye6+1] = 2*uwall - v[xe6+1, ys6-1:ye6+1]    # east  face
    # Obstacle 7
    u[xs7-1:xe7+1, ye7] = 2*uwall - u[xs7-1:xe7+1, ye7+1]    # north face
    u[xs7-1:xe7+1, ys7] = 2*uwall - u[xs7-1:xe7+1, ys7-1]    # south face
    v[xs7, ys7-1:ye7+1] = 2*uwall - v[xs7-1, ys7-1:ye7+1]    # west  face
    v[xe7, ys7-1:ye7+1] = 2*uwall - v[xe7+1, ys7-1:ye7+1]    # east  face
    # Obstacle 8
    u[xs8-1:xe8+1, ye8] = 2*uwall - u[xs8-1:xe8+1, ye8+1]    # north face
    u[xs8-1:xe8+1, ys8] = 2*uwall - u[xs8-1:xe8+1, ys8-1]    # south face
    v[xs8, ys8-1:ye8+1] = 2*uwall - v[xs8-1, ys8-1:ye8+1]    # west  face
    v[xe8, ys8-1:ye8+1] = 2*uwall - v[xe8+1, ys8-1:ye8+1]    # east  face
    # Obstacle 9
    u[xs9-1:xe9+1, ye9] = 2*uwall - u[xs9-1:xe9+1, ye9+1]    # north face
    u[xs9-1:xe9+1, ys9] = 2*uwall - u[xs9-1:xe9+1, ys9-1]    # south face
    v[xs9, ys9-1:ye9+1] = 2*uwall - v[xs9-1, ys9-1:ye9+1]    # west  face
    v[xe9, ys9-1:ye9+1] = 2*uwall - v[xe9+1, ys9-1:ye9+1]    # east  face

    # Over-writing inlet conditions
    u[0, nin_start-1:nin_end+1] = u_in   # fixed inlet velocity

    # Over-writing outlet conditions
    u[-1, nout_start-1:nout_end+1] = u[-2, nout_start-1:nout_end+1]   # zero-gradient outlet velocity
    v[-1, nout_start-1:nout_end+1] = v[-2, nout_start-1:nout_end+1]   # zero-gradient outlet velocity

    # Advection-diffusion equation (predictor)
    ut, vt = AdvectionDiffusion2D(ut, vt, u, v, nx, ny, hx, hy, dt, nu, flagu, flagv, method)

    # Update boundary conditions for temporary velocities
    ut[0, nin_start-1:nin_end+1] = u_in       # fixed inlet velocity
    ut[-1, nout_start-1:nout_end+1] = u[-1, nout_start-1:nout_end+1]   # zero-gradient outlet velocity
    vt[-1, nout_start-1:nout_end+1] = v[-1, nout_start-1:nout_end+1]   # zero-gradient outlet velocity

    # Pressure equation (Poisson)
    p, iter = Pressure_Poisson(p, ut, vt, gamma, nx, ny, hx, hy, dt, beta, max_iterations, max_error, flagp)

    # Correction on the velocity
    u, v = correction_velocity(u, v, ut, vt, p, nx, ny, hx, hy, dt, flagu, flagv)

    # Print on the screen
    if it <= 50:
        if it % 1 == 0:
            print('Step:', it, '- Time:', t,  '- Poisson iterations:', iter)
    else :
        if it % 500 == 1:
            print('Step:', it, '- Time:', t,  '- Poisson iterations:', iter)
    #----------------------------------------------------------------------------------#
    # 2. Transport of species and reaction                                             #
    #----------------------------------------------------------------------------------#
    # Impermeable walls
    cO2 = SpeciesBCs(cO2, nx,ny, xs1,xe1,ys1,ye1, xs2,xe2,ys2,ye2, xs3,xe3,ys3,ye3, xs4,xe4,ys4,ye4, xs5,xe5,ys5,ye5, xs6,xe6,ys6,ye6, xs7,xe7,ys7,ye7, xs8,xe8,ys8,ye8, xs9,xe9,ys9,ye9)

    # Inlet sections
    cO2[0, nin_start-1:nin_end+1] = 2*cO2in - cO2[1, nin_start-1:nin_end+1]

    # Advection-Diffusion equation
    cO2star = AdvDiffSpecies(cO2, u, v, dt, hx, hy, Gamma, nx, ny, flagp, method)
    cBstar = cB

    # Reaction step (Linearization + Segregation)
    cO2, cB = ReactionStep(cO2star, cBstar, dt, k, nx, ny)

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

# Defining the obstacles coordinates
X1 = [xs1, xe1, ys1, ye1]
X2 = [xs2, xe2, ys2, ye2]
X3 = [xs3, xe3, ys3, ye3]
X4 = [xs4, xe4, ys4, ye4]
X5 = [xs5, xe5, ys5, ye5]
X6 = [xs6, xe6, ys6, ye6]
X7 = [xs7, xe7, ys7, ye7]
X8 = [xs8, xe8, ys8, ye8]
X9 = [xs9, xe9, ys9, ye9]

# Interpolation of cells concentration needed for graphical purposes
ccB[xs1-th:xs1, ys1:ye1] = cB[xs1-th+1:xs1+1, ys1+1:ye1+1]  # Obstacle 1
ccB[xe1+1:xe1+th+1, ys1:ye1] = cB[xe1+1:xe1+th+1, ys1+1:ye1+1]
ccB[xs2-th:xs2, ys2:ye2] = cB[xs2-th+1:xs2+1, ys2+1:ye2+1]  # Obstacle 2
ccB[xe2+1:xe2+th+1, ys2:ye2] = cB[xe2+1:xe2+th+1, ys2+1:ye2+1]
ccB[xs3-th:xs3, ys3:ye3] = cB[xs3-th+1:xs3+1, ys3+1:ye3+1]  # Obstacle 3
ccB[xe3+1:xe3+th+1, ys3:ye3] = cB[xe3+1:xe3+th+1, ys3+1:ye3+1]
ccB[xs4-th:xs4, ys4:ye4] = cB[xs4-th+1:xs4+1, ys4+1:ye4+1]  # Obstacle 4
ccB[xe4+1:xe4+th+1, ys4:ye4] = cB[xe4+1:xe4+th+1, ys4+1:ye4+1]
ccB[xs5-th:xs5, ys5:ye5] = cB[xs5-th+1:xs5+1, ys5+1:ye5+1]  # Obstacle 5
ccB[xe5+1:xe5+th+1, ys5:ye5] = cB[xe5+1:xe5+th+1, ys5+1:ye5+1]
ccB[xs6-th:xs6, ys6:ye6] = cB[xs6-th+1:xs6+1, ys6+1:ye6+1]  # Obstacle 6
ccB[xe6+1:xe6+th+1, ys6:ye6] = cB[xe6+1:xe6+th+1, ys6+1:ye6+1]
ccB[xs7-th:xs7, ys7:ye7] = cB[xs7-th+1:xs7+1, ys7+1:ye7+1]  # Obstacle 7
ccB[xe7+1:xe7+th+1, ys7:ye7] = cB[xe7+1:xe7+th+1, ys7+1:ye7+1]
ccB[xs8-th:xs8, ys8:ye8] = cB[xs8-th+1:xs8+1, ys8+1:ye8+1]  # Obstacle 8
ccB[xe8+1:xe8+th+1, ys8:ye8] = cB[xe8+1:xe8+th+1, ys8+1:ye8+1]
ccB[xs9-th:xs9, ys9:ye9] = cB[xs9-th+1:xs9+1, ys9+1:ye9+1]  # Obstacle 9
ccB[xe9+1:xe9+th+1, ys9:ye9] = cB[xe9+1:xe9+th+1, ys9+1:ye9+1]

# Addition of obstacles for graphical purposes
uu, vv, pp, ccO2, ccB = Graphical_obstacles(uu, vv, pp, ccO2, ccB, X1, X2, X3, X4, X5, X6, X7, X8, X9)

# Creating a grid
xx,yy = np.meshgrid(x,y)

# Plotting the results
# Surface map: pressure
PlotFunctions(xx, yy, pp, x, y, X1, X2, X3, X4, X5, X6, X7, X8, X9, Lx, Ly, 'Delta Pressure [Pa]', 'x [m]', 'y [m]')

# Surface map: u-velocity
PlotFunctions(xx, yy, uu, x, y, X1, X2, X3, X4, X5, X6, X7, X8, X9, Lx, Ly, 'u - velocity [m/s]', 'x [m]', 'y [m]')

# Surface map: v-velocity
PlotFunctions(xx, yy, vv, x, y, X1, X2, X3, X4, X5, X6, X7, X8, X9, Lx, Ly, 'v - velocity [m/s]', 'x [m]', 'y [m]')

# Surface map: cO2
PlotFunctions(xx, yy, ccO2, x, y, X1, X2, X3, X4, X5, X6, X7, X8, X9, Lx, Ly, 'O2 concentration [kmol/m3]', 'x [m]', 'y [m]')

# Surface map: cB
PlotFunctions(xx, yy, ccB, x, y, X1, X2, X3, X4, X5, X6, X7, X8, X9, Lx, Ly, 'Cells concentration [kmol/m3]', 'x [m]', 'y [m]')

# Streamlines
fig, ax = plt.subplots()
# plt.quiver(xx[::10], yy[::10], np.transpose(uu)[::10], np.transpose(vv)[::10])
plt.streamplot(xx, yy, np.transpose(uu), np.transpose(vv), linewidth = 1)
ax.add_patch(Rectangle((x[xs1-1], y[ys1-1]), x[xe1-xs1+1], y[ye1-ys1+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs2-1], y[ys2-1]), x[xe2-xs2+1], y[ye2-ys2+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs3-1], y[ys3-1]), x[xe3-xs3+1], y[ye3-ys3+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs4-1], y[ys4-1]), x[xe4-xs4+1], y[ye4-ys4+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs5-1], y[ys5-1]), x[xe5-xs5+1], y[ye5-ys5+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs6-1], y[ys6-1]), x[xe6-xs6+1], y[ye6-ys6+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs7-1], y[ys7-1]), x[xe7-xs7+1], y[ye7-ys7+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs8-1], y[ys8-1]), x[xe8-xs8+1], y[ye8-ys8+1], facecolor = 'grey'))
ax.add_patch(Rectangle((x[xs9-1], y[ys9-1]), x[xe9-xs9+1], y[ye9-ys9+1], facecolor = 'grey'))
plt.title('Streamlines')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlim(0,Lx)
plt.ylim(0,Ly)

plt.show()