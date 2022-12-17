import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from   matplotlib import cm
from   matplotlib.patches import Rectangle
from   numba import njit
from   functions_4 import *
plt.style.use(['science', 'no-latex'])

## -------------------------------------------------------------------------------------------- ##
##                                     PRE-PROCESSING                                           ##
## -------------------------------------------------------------------------------------------- ##
# Data
P = 101325.0       # Pa
T = 25+273.15      # K
g = 0              # m/s2  (g = 9.80665)
rhoL = 1000.       # kg/m3

Lx = 3.6           # m
Ly = 1.0           # m
nx = 360
ny = 100
nu = 1e-6          # m2/s  (dynamic viscosity of fluid)
Gamma = 1e-4       # m2/s  (diffusion coefficient of O2 in water)
u_in = 2e-3        # m/s   (inlet velocity)
Time = 100.0       # s     (simulation time)
method = 'Upwind'  # Discretization method (CDS or Upwind)

# Boundary conditions (no-slip)
un = 0             # m/s
us = 0             # m/s
ve = 0             # m/s
vw = 0             # m/s

# Inlet concentrations
cO2in = 1.        # mol/m3
cGin = 1.         # mol/m3

# Immobilized cells parameters
cB0 = 0.01          # mol/m3
thick = 0.05        # m (thickness of cells layer)
mu_max = 0.5/3600   # 1/s (max growth rate)
K_G = 0.5           # mol/m3 (Monod const. of glucose)

# Parameters for SOR (Poisson eq)
max_iterations = 1000000
beta = 1.9
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
x_obst1_start = 0.
x_obst1_end = 0.1
y_obst1_start = 0.
y_obst1_end = 0.6
# Obstacle 2
x_obst2_start = 0.5
x_obst2_end = 0.6
y_obst2_start = 0.4
y_obst2_end = 1
# Obstacle 3
x_obst3_start = 1.
x_obst3_end = 1.1
y_obst3_start = 0.
y_obst3_end = 0.6
# Obstacle 4
x_obst4_start = 1.5
x_obst4_end = 1.6
y_obst4_start = 0.4
y_obst4_end = 1.0
# Obstacle 5
x_obst5_start = 2.
x_obst5_end = 2.1
y_obst5_start = 0.
y_obst5_end = 0.6
# Obstacle 6
x_obst6_start = 2.5
x_obst6_end = 2.6
y_obst6_start = 0.4
y_obst6_end = 1.0
# Obstacle 7
x_obst7_start = 3.0
x_obst7_end = 3.1
y_obst7_start = 0.
y_obst7_end = 0.6
# Obstacle 8
x_obst8_start = 3.5
x_obst8_end = 3.6
y_obst8_start = 0.4
y_obst8_end = 1.0

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

# Defining the obstacles coordinates
X1 = (xs1, xe1, ys1, ye1)
X2 = (xs2, xe2, ys2, ye2)
X3 = (xs3, xe3, ys3, ye3)
X4 = (xs4, xe4, ys4, ye4)
X5 = (xs5, xe5, ys5, ye5)
X6 = (xs6, xe6, ys6, ye6)
X7 = (xs7, xe7, ys7, ye7)
X8 = (xs8, xe8, ys8, ye8)

# Inlet section (west side)
nin_start = ye1 + 1   # first cell index (pay attention cause it is in MATLAB notation!)
nin_end = ny + 1      # last cell index (pay attention cause it is in MATLAB notation!)

# Outlet section (east side)
nout_start = 2        # first cell index (pay attention cause it is in MATLAB notation!)
nout_end = ys8 - 1    # last cell index (pay attention cause it is in MATLAB notation!)

# Time step
sigma = 0.5                                        # safety factor for time step (stability)
dt_diff_ns = np.minimum(hx,hy)**2/4/nu             # time step (diffusion stability) [s]
dt_conv_ns = 4*nu/u_in**2                          # time step (convection stability) [s]
dt_ns      = np.minimum(dt_diff_ns, dt_conv_ns);   # time step (stability due to FD) [s]
dt_diff_sp = np.minimum(hx,hy)**2/4/Gamma;         # time step (species diffusion stability) [s]
dt_conv_sp = 4*Gamma/u_in**2;                      # time step (species convection stability) [s]
dt_sp      = np.minimum(dt_conv_sp, dt_diff_sp);   # time step (stability due to species) [s]
dt         = sigma*min(dt_ns, dt_sp);              # time step (stability) [s]
nsteps     = int(Time/dt)                           # number of steps
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
cG  = np.zeros([nx+2, ny+2])

# Temporary velocity fields
ut = np.zeros_like(u)
vt = np.zeros_like(v)

# Fields used only for graphical purposes
uu   = np.zeros([nx+1, ny+1])
vv   = np.zeros([nx+1, ny+1])
pp   = np.zeros([nx+1, ny+1])
ccO2 = np.zeros([nx+1, ny+1])
ccB  = np.zeros([nx+1, ny+1])
ccG  = np.zeros([nx+1, ny+1])
cBmean1 = np.zeros([nsteps])
cBmean2 = np.zeros([nsteps])
cBmean3 = np.zeros([nsteps])
cBmean4 = np.zeros([nsteps])
cBmean5 = np.zeros([nsteps])
cBmean6 = np.zeros([nsteps])
cBmean7 = np.zeros([nsteps])
t_vector = np.zeros([nsteps])

# Coefficients for pressure equation
gamma = np.zeros([nx+2,ny+2]) + hx*hy / (2*hx**2 + 2*hy**2)   # internal points
gamma = gammaCoeff(gamma, hx,hy, nout_start,nout_end, X1, X2, X3, X4, X5, X6, X7, X8)

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

# Initial conditions: set reasonable initial velocity value instead of initializing everything to zero
u[:, :] = u_in                     # Internal points: fixed velocity [m/s]
u[xs1-1:xe1+1, ys1:ye1+1] = 0      # Obstacle 1
u[xs2-1:xe2+1, ys2:ye2+1] = 0      # Obstacle 2
u[xs3-1:xe3+1, ys3:ye3+1] = 0      # Obstacle 3
u[xs4-1:xe4+1, ys4:ye4+1] = 0      # Obstacle 4
u[xs5-1:xe5+1, ys5:ye5+1] = 0      # Obstacle 5
u[xs6-1:xe6+1, ys6:ye6+1] = 0      # Obstacle 6
u[xs7-1:xe7+1, ys7:ye7+1] = 0      # Obstacle 7
u[xs8-1:xe8+1, ys8:ye8+1] = 0      # Obstacle 8
ut = u

# Immobilized cells initialized over obstacles
cB[xe1+1:xe1+th+1, ys1:ye1+1] = cB0    # Obstacle 1
cB[xe2+1:xe2+th+1, ys2:ye2+1] = cB0    # Obstacle 2
cB[xe3+1:xe3+th+1, ys3:ye3+1] = cB0    # Obstacle 3
cB[xe4+1:xe4+th+1, ys4:ye4+1] = cB0    # Obstacle 4
cB[xe5+1:xe5+th+1, ys5:ye5+1] = cB0    # Obstacle 5
cB[xe6+1:xe6+th+1, ys6:ye6+1] = cB0    # Obstacle 6
cB[xe7+1:xe7+th+1, ys7:ye7+1] = cB0    # Obstacle 7

xx,yy = np.meshgrid(x,y)

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
    u, v = VelocityBCs(u, v, uwall, X1, X2, X3, X4, X5, X6, X7, X8)

    # Over-writing inlet conditions
    u[0, nin_start-1:nin_end+1] = u_in   # fixed inlet velocity

    # Over-writing outlet conditions
    u[-1, nout_start-1:nout_end+1] = u[-2, nout_start-1:nout_end+1]   # zero-gradient outlet velocity
    v[-1, nout_start-1:nout_end+1] = v[-2, nout_start-1:nout_end+1]   # zero-gradient outlet velocity

    # Advection-diffusion equation (predictor)
    ut, vt = AdvectionDiffusion2D(ut, vt, u, v, nx, ny, hx, hy, dt, nu, g, flagu, flagv, method)

    # Update boundary conditions for temporary velocities
    ut[0, nin_start-1:nin_end+1] = u_in       # fixed inlet velocity
    ut[-1, nout_start-1:nout_end+1] = u[-1, nout_start-1:nout_end+1]   # zero-gradient outlet velocity
    vt[-1, nout_start-1:nout_end+1] = v[-1, nout_start-1:nout_end+1]   # zero-gradient outlet velocity

    # Pressure equation (Poisson)
    p, iter = Pressure_Poisson(p, ut, vt, gamma, nx, ny, hx, hy, dt, beta, max_iterations, max_error, flagp)

    # Correction on the velocity
    u, v = correction_velocity(u, v, ut, vt, p, nx, ny, hx, hy, dt, flagu, flagv)

    # Print on the screen
    if it <= 20:
        if it % 1 == 0:
            print('Step:', it, '- Time:', round(t,3),  '- Poisson iterations:', iter)
    else :
        if it % 500 == 0:
            print('Step:', it, '- Time:', round(t,3),  '- Poisson iterations:', iter)
            
    #----------------------------------------------------------------------------------#
    # 2. Transport of species and reaction                                             #
    #----------------------------------------------------------------------------------#
    # Impermeable walls
    cO2 = SpeciesBCs(cO2, nx,ny, X1, X2, X3, X4, X5, X6, X7, X8)
    cG  = SpeciesBCs(cG, nx,ny, X1, X2, X3, X4, X5, X6, X7, X8)

    # Inlet sections
    cO2[0, nin_start-1:nin_end+1] = 2*cO2in - cO2[1, nin_start-1:nin_end+1]
    cG[0, nin_start-1:nin_end+1] = 2*cGin - cG[1, nin_start-1:nin_end+1]

    # Advection-Diffusion equation
    cO2star = AdvDiffSpecies(cO2, u, v, dt, hx, hy, Gamma, nx, ny, flagp, method)
    cGstar = AdvDiffSpecies(cG, u, v, dt, hx, hy, Gamma, nx, ny, flagp, method)
    cBstar = cB

    # Reaction step (Linearization + Segregation)
    cO2, cB, cG = ReactionStep(cO2star, cBstar, cGstar, dt, mu_max, K_G, nx, ny)

    # Collecting cells mean values for graphical purposes
    cBmean1[it-1] = CellsMeanConcentration(cB, cB0, th, X1)
    cBmean2[it-1] = CellsMeanConcentration(cB, cB0, th, X2)
    cBmean3[it-1] = CellsMeanConcentration(cB, cB0, th, X3)
    cBmean4[it-1] = CellsMeanConcentration(cB, cB0, th, X4)
    cBmean5[it-1] = CellsMeanConcentration(cB, cB0, th, X5)
    cBmean6[it-1] = CellsMeanConcentration(cB, cB0, th, X6)
    cBmean7[it-1] = CellsMeanConcentration(cB, cB0, th, X7)

    # Collecting time values for graphical purposes
    t_vector[it-1] = t

    # Advance in time
    t = t + dt

# Shear stress over the entire domain
dR = x_obst3_start - x_obst1_end
Tau_wall = ShearStress(v, dR, nu, rhoL, nx, ny)


## -------------------------------------------------------------------------------------------- ##
##                                    FINAL POST-PROCESSING                                     ##
## -------------------------------------------------------------------------------------------- ##
# Field reconstruction
uu   = node_interp(u, 'u', nx, ny, flagu)
vv   = node_interp(v, 'v', nx, ny, flagv)
pp   = node_interp(p, 'p', nx, ny, flagp)
tau  = node_interp(Tau_wall, 'v', nx, ny, flagv)
ccO2 = node_interp(cO2, 'p', nx, ny, flagp)
ccG  = node_interp(cG, 'p', nx, ny, flagp)

# Interpolation of cells concentration needed for graphical purposes
ccB[xe1+1:xe1+th+1, ys1:ye1] = cB[xe1+1:xe1+th+1, ys1+1:ye1+1]  # Obstacle 1
ccB[xe2+1:xe2+th+1, ys2:ye2] = cB[xe2+1:xe2+th+1, ys2+1:ye2+1]  # Obstacle 2
ccB[xe3+1:xe3+th+1, ys3:ye3] = cB[xe3+1:xe3+th+1, ys3+1:ye3+1]  # Obstacle 3
ccB[xe4+1:xe4+th+1, ys4:ye4] = cB[xe4+1:xe4+th+1, ys4+1:ye4+1]  # Obstacle 4
ccB[xe5+1:xe5+th+1, ys5:ye5] = cB[xe5+1:xe5+th+1, ys5+1:ye5+1]  # Obstacle 5
ccB[xe6+1:xe6+th+1, ys6:ye6] = cB[xe6+1:xe6+th+1, ys6+1:ye6+1]  # Obstacle 6
ccB[xe7+1:xe7+th+1, ys7:ye7] = cB[xe7+1:xe7+th+1, ys7+1:ye7+1]  # Obstacle 7

# Addition of obstacles for graphical purposes
uu, vv, pp, tau, ccO2, ccB, ccG = Graphical_obstacles(uu, vv, pp, tau, ccO2, ccB, ccG, X1, X2, X3, X4, X5, X6, X7, X8)

# Creating a grid
xx,yy = np.meshgrid(x,y)

# Plotting the results
# Surface map: pressure
PlotFunctions(xx, yy, pp, x, y, X1, X2, X3, X4, X5, X6, X7, X8, Lx, Ly, 'Delta Pressure [Pa]', 'x [m]', 'y [m]')

# Surface map: u-velocity
PlotFunctions(xx, yy, uu, x, y, X1, X2, X3, X4, X5, X6, X7, X8, Lx, Ly, 'u - velocity [m/s]', 'x [m]', 'y [m]')

# Surface map: v-velocity
PlotFunctions(xx, yy, vv, x, y, X1, X2, X3, X4, X5, X6, X7, X8, Lx, Ly, 'v - velocity [m/s]', 'x [m]', 'y [m]')

# Surface map: cO2
PlotFunctions(xx, yy, ccO2, x, y, X1, X2, X3, X4, X5, X6, X7, X8, Lx, Ly, 'O2 concentration [mol/m3]', 'x [m]', 'y [m]')

# Surface map: cG
PlotFunctions(xx, yy, ccG, x, y, X1, X2, X3, X4, X5, X6, X7, X8, Lx, Ly, 'Glucose concentration [mol/m3]', 'x [m]', 'y [m]')

# Surface map: cB
PlotFunctions(xx, yy, ccB, x, y, X1, X2, X3, X4, X5, X6, X7, X8, Lx, Ly, 'Cells concentration [mol/m3]', 'x [m]', 'y [m]')

# Surface map: Tau
PlotFunctions(xx, yy, tau, x, y, X1, X2, X3, X4, X5, X6, X7, X8, Lx, Ly, 'Shear stress [N/m2]', 'x [m]', 'y [m]')

# Streamlines
Streamlines(x, y, xx, yy, uu, vv, X1, X2, X3, X4, X5, X6, X7, X8, Lx, Ly, 'Streamlines', 'x [m]', 'y [m]')

# Mean values of cB
MeanCellsPlot(t_vector, cBmean1, cBmean2, cBmean3, cBmean4, cBmean5, cBmean6, cBmean7, Time, 0.2)

plt.show()