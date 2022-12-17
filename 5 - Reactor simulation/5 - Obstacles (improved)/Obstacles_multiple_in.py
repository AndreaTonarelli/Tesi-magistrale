import numpy as np
import matplotlib.pyplot as plt
from   numba import njit
from   functions_9 import *
from   Obstacles_def import *
plt.style.use(['science', 'no-latex'])

## -------------------------------------------------------------------------------------------- ##
##                                     PRE-PROCESSING                                           ##
## -------------------------------------------------------------------------------------------- ##
# Data
P = 101325.0       # Pa
T = 25+273.15      # K
g = 0.             # m/s2  (g = 9.80665)
rhoL = 1000.       # kg/m3

Lx = 4.0           # m
Ly = 1.2           # m
nx = 500
ny = 160
nu = 1e-6          # m2/s  (dynamic viscosity of fluid)
Gamma = 1e-4       # m2/s  (diffusion coefficient of O2 in water)
u_in = 1e-3        # m/s   (inlet velocity)
v_in = 5e-4        # m/s   (inlet velocity)
Time = 800.0      # s     (simulation time)
method = 'Upwind'  # Discretization method (CDS or Upwind)

# Boundary conditions (no-slip)
un = 0             # m/s
us = 0             # m/s
ve = 0             # m/s
vw = 0             # m/s

# Inlet concentrations
cO2in = 1.         # mol/m3
cGin = 1.          # mol/m3

# Immobilized cells parameters
cB0 = 0.01          # mol/m3 (initialconcentration of cells)
thick = 0.05        # m (thickness of cells layer)
mu_max = 0.5/3600   # 1/s (max growth rate)
K_G = 0.5           # mol/m3 (Monod const. of glucose)

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

# Find cells corresponding to thickness
th = int(np.ceil(thick/hx))    # it finds the index not the positions!

# Defining the obstacles coordinates
X1 = Obstacles(x, y)[0]; xs1 = X1[0]; xe1 = X1[1]; ys1 = X1[2]; ye1 = X1[3]
X2 = Obstacles(x, y)[1]; xs2 = X2[0]; xe2 = X2[1]; ys2 = X2[2]; ye2 = X2[3]
X3 = Obstacles(x, y)[2]; xs3 = X3[0]; xe3 = X3[1]; ys3 = X3[2]; ye3 = X3[3]
X4 = Obstacles(x, y)[3]; xs4 = X4[0]; xe4 = X4[1]; ys4 = X4[2]; ye4 = X4[3]
X5 = Obstacles(x, y)[4]; xs5 = X5[0]; xe5 = X5[1]; ys5 = X5[2]; ye5 = X5[3]
X6 = Obstacles(x, y)[5]; xs6 = X6[0]; xe6 = X6[1]; ys6 = X6[2]; ye6 = X6[3]
X7 = Obstacles(x, y)[6]; xs7 = X7[0]; xe7 = X7[1]; ys7 = X7[2]; ye7 = X7[3]
X8 = Obstacles(x, y)[7]; xs8 = X8[0]; xe8 = X8[1]; ys8 = X8[2]; ye8 = X8[3]
X9 = Obstacles(x, y)[8]; xs9 = X9[0]; xe9 = X9[1]; ys9 = X9[2]; ye9 = X9[3]
X10 = Obstacles(x, y)[9]; xs10 = X10[0]; xe10 = X10[1]; ys10 = X10[2]; ye10 = X10[3]

# Defining obstacles near corners
xs11 = X1[4]; xe11 = X1[5]; ys11 = X1[6]; ye11 = X1[7]; xs12 = X1[8]; xe12 = X1[9]; ys12 = X1[10]; ye12 = X1[11]
xs21 = X2[4]; xe21 = X2[5]; ys21 = X2[6]; ye21 = X2[7]; xs22 = X2[8]; xe22 = X2[9]; ys22 = X2[10]; ye22 = X2[11]
xs31 = X3[4]; xe31 = X3[5]; ys31 = X3[6]; ye31 = X3[7]; xs32 = X3[8]; xe32 = X3[9]; ys32 = X3[10]; ye32 = X3[11]
xs41 = X4[4]; xe41 = X4[5]; ys41 = X4[6]; ye41 = X4[7]; xs42 = X4[8]; xe42 = X4[9]; ys42 = X4[10]; ye42 = X4[11]
xs51 = X5[4]; xe51 = X5[5]; ys51 = X5[6]; ye51 = X5[7]; xs52 = X5[8]; xe52 = X5[9]; ys52 = X5[10]; ye52 = X5[11]
xs61 = X6[4]; xe61 = X6[5]; ys61 = X6[6]; ye61 = X6[7]; xs62 = X6[8]; xe62 = X6[9]; ys62 = X6[10]; ye62 = X6[11]
xs71 = X7[4]; xe71 = X7[5]; ys71 = X7[6]; ye71 = X7[7]; xs72 = X7[8]; xe72 = X7[9]; ys72 = X7[10]; ye72 = X7[11]
xs81 = X8[4]; xe81 = X8[5]; ys81 = X8[6]; ye81 = X8[7]; xs82 = X8[8]; xe82 = X8[9]; ys82 = X8[10]; ye82 = X8[11]
xs91 = X9[4]; xe91 = X9[5]; ys91 = X9[6]; ye91 = X9[7]; xs92 = X9[8]; xe92 = X9[9]; ys92 = X9[10]; ye92 = X9[11]

# Inlet section (west side)
nin_start1 = ye1 + 1   # first cell index (pay attention cause it is in MATLAB notation!)
nin_end1 = ny + 1      # last cell index (pay attention cause it is in MATLAB notation!)
# Inlet section lateral 1 (south side)
nin_start2 = xe2 + 1   # first cell index (pay attention cause it is in MATLAB notation!)
nin_end2 = xs3 - 1      # last cell index (pay attention cause it is in MATLAB notation!)
# Inlet section lateral 2 (north side)
nin_start3 = xe3 + 1   # first cell index (pay attention cause it is in MATLAB notation!)
nin_end3 = xs4 - 1      # last cell index (pay attention cause it is in MATLAB notation!)
# Inlet section lateral 3 (south side)
nin_start4 = xe4 + 1   # first cell index (pay attention cause it is in MATLAB notation!)
nin_end4 = xs5 - 1      # last cell index (pay attention cause it is in MATLAB notation!)
# Inlet section lateral 4 (north side)
nin_start5 = xe5 + 1   # first cell index (pay attention cause it is in MATLAB notation!)
nin_end5 = xs6 - 1      # last cell index (pay attention cause it is in MATLAB notation!)
# Inlet section lateral 5 (south side)
nin_start6 = xe6 + 1   # first cell index (pay attention cause it is in MATLAB notation!)
nin_end6 = xs7 - 1      # last cell index (pay attention cause it is in MATLAB notation!)
# Inlet section lateral 6 (north side)
nin_start7 = xe7 + 1   # first cell index (pay attention cause it is in MATLAB notation!)
nin_end7 = xs8 - 1      # last cell index (pay attention cause it is in MATLAB notation!)
# Inlet section lateral 7 (south side)
nin_start8 = xe8 + 1   # first cell index (pay attention cause it is in MATLAB notation!)
nin_end8 = xs9 - 1      # last cell index (pay attention cause it is in MATLAB notation!)
# Inlet section lateral 8 (north side)
nin_start9 = xe9 + 1   # first cell index (pay attention cause it is in MATLAB notation!)
nin_end9 = xs10 - 1      # last cell index (pay attention cause it is in MATLAB notation!)

# Outlet section (east side)
nout_start = 2        # first cell index (pay attention cause it is in MATLAB notation!)
nout_end = ys8 - 1    # last cell index (pay attention cause it is in MATLAB notation!)

# Time step
sigma = 0.75                                       # safety factor for time step (stability)
dt_diff_ns = np.minimum(hx,hy)**2/4/nu             # time step (diffusion stability) [s]
dt_conv_ns = 4*nu/u_in**2                          # time step (convection stability) [s]
dt_ns      = np.minimum(dt_diff_ns, dt_conv_ns);   # time step (stability due to FD) [s]
dt_diff_sp = np.minimum(hx,hy)**2/4/Gamma;         # time step (species diffusion stability) [s]
dt_conv_sp = 4*Gamma/u_in**2;                      # time step (species convection stability) [s]
dt_sp      = np.minimum(dt_conv_sp, dt_diff_sp);   # time step (stability due to species) [s]
dt         = sigma*min(dt_ns, dt_sp);              # time step (stability) [s]
nsteps     = int(Time/dt)                          # number of steps
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
cBmean8 = np.zeros([nsteps])
cBmean9 = np.zeros([nsteps])
cBmean10 = np.zeros([nsteps])
t_vector = np.zeros([nsteps])

# Coefficients for pressure equation
gamma = np.zeros([nx+2,ny+2]) + hx*hy / (2*hx**2 + 2*hy**2)   # internal points
gamma = gammaCoeff(gamma, hx,hy, nout_start,nout_end, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10)

# Flags to recognize where is necessary to solve the equations
# This flag is 1 in the cells that contain and obstacle and 0 in the others
flagu = np.zeros([nx+1, ny+2])          # u-cells corresponding to the obstacle
flagv = np.zeros([nx+2, ny+1])          # v-cells corresponding to the obstacle
flagp = np.zeros([nx+2, ny+2])          # p-cells corresponding to the obstacle

# Set flag to 1 in obstacle cells
flagu, flagv, flagp = flag(flagu,flagv,flagp, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10)

# Initial conditions: set reasonable initial velocity value instead of initializing everything to zero
u[:, :] = u_in/10                   # Internal points: fixed velocity [m/s]
u = u_initialize(u, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10) # set u = 0 over obstacles
ut = u

# Immobilized cells initialized over obstacles
'''cB[xe1+1:xe1+th+1, ys1:ye1+1] = cB0    # Obstacle 1
cB[xe2+1:xe2+th+1, ys2:ye2+1] = cB0    # Obstacle 2
cB[xe3+1:xe3+th+1, ys3:ye3+1] = cB0    # Obstacle 3
cB[xe4+1:xe4+th+1, ys4:ye4+1] = cB0    # Obstacle 4
cB[xe5+1:xe5+th+1, ys5:ye5+1] = cB0    # Obstacle 5
cB[xe6+1:xe6+th+1, ys6:ye6+1] = cB0    # Obstacle 6
cB[xe7+1:xe7+th+1, ys7:ye7+1] = cB0    # Obstacle 7'''
cB[xs2-th:xs2, ys2:ye2] = cB0    # Obstacle 2
cB[xs3-th:xs3, ys3:ye3] = cB0    # Obstacle 3
cB[xs4-th:xs4, ys4:ye4] = cB0    # Obstacle 4
cB[xs5-th:xs5, ys5:ye5] = cB0    # Obstacle 5
cB[xs6-th:xs6, ys6:ye6] = cB0    # Obstacle 6
cB[xs7-th:xs7, ys7:ye7] = cB0    # Obstacle 7
cB[xs8-th:xs8, ys8:ye8] = cB0    # Obstacle 8
cB[xs9-th:xs9, ys9:ye9] = cB0    # Obstacle 9
cB[xs10-th:xs10, ys10:ye10] = cB0    # Obstacle 10

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
    u, v = VelocityBCs(u, v, uwall, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10)

    # Over-writing inlet conditions
    u[0, nin_start1-1:nin_end1+1] = u_in     # fixed inlet velocity (west)
    v[nin_start2-1:nin_end2+1, 0] = v_in     # fixed inlet velocity (south)
    v[nin_start3-1:nin_end3+1, -1] = -v_in   # fixed inlet velocity (north)
    v[nin_start4-1:nin_end4+1, 0] = v_in     # fixed inlet velocity (south)
    v[nin_start5-1:nin_end5+1, -1] = -v_in   # fixed inlet velocity (north)
    v[nin_start6-1:nin_end6+1, 0] = v_in     # fixed inlet velocity (south)
    v[nin_start7-1:nin_end7+1, -1] = -v_in   # fixed inlet velocity (north)
    v[nin_start8-1:nin_end8+1, 0] = v_in     # fixed inlet velocity (south)
    v[nin_start9-1:nin_end9+1, -1] = -v_in   # fixed inlet velocity (north)

    # Over-writing outlet conditions
    u[-1, nout_start-1:nout_end+1] = u[-2, nout_start-1:nout_end+1]   # zero-gradient outlet velocity
    v[-1, nout_start-1:nout_end+1] = v[-2, nout_start-1:nout_end+1]   # zero-gradient outlet velocity

    # Advection-diffusion equation (predictor)
    ut, vt = AdvectionDiffusion2D(ut, vt, u, v, nx, ny, hx, hy, dt, nu, g, flagu, flagv, method)

    # Update boundary conditions for temporary velocities
    ut[0, nin_start1-1:nin_end1+1] = u_in     # fixed inlet velocity (west)
    vt[nin_start2-1:nin_end2+1, 0] = v_in     # fixed inlet velocity (south)
    vt[nin_start3-1:nin_end3+1, -1] = -v_in   # fixed inlet velocity (north)
    vt[nin_start4-1:nin_end4+1, 0] = v_in     # fixed inlet velocity (south)
    vt[nin_start5-1:nin_end5+1, -1] = -v_in   # fixed inlet velocity (north)
    vt[nin_start6-1:nin_end6+1, 0] = v_in     # fixed inlet velocity (south)
    vt[nin_start7-1:nin_end7+1, -1] = -v_in   # fixed inlet velocity (north)
    vt[nin_start8-1:nin_end8+1, 0] = v_in     # fixed inlet velocity (south)
    vt[nin_start9-1:nin_end9+1, -1] = -v_in   # fixed inlet velocity (north)
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
    cO2 = SpeciesBCs(cO2, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10)
    cG  = SpeciesBCs(cG, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10)

    # Inlet sections
    cO2[0, nin_start1-1:nin_end1+1] = 2*cO2in - cO2[1, nin_start1-1:nin_end1+1]
    cG[0, nin_start1-1:nin_end1+1] = 2*cGin - cG[1, nin_start1-1:nin_end1+1]
    cO2[nin_start2-1:nin_end2+1, 0] = 2*cO2in - cO2[nin_start2-1:nin_end2+1, 1]
    cG[nin_start2-1:nin_end2+1, 0] = 2*cGin - cG[nin_start2-1:nin_end2+1, 1]
    cO2[nin_start3-1:nin_end3+1, -1] = 2*cO2in - cO2[nin_start3-1:nin_end3+1, -2]
    cG[nin_start3-1:nin_end3+1, -1] = 2*cGin - cG[nin_start3-1:nin_end3+1, -2]
    cO2[nin_start4-1:nin_end4+1, 0] = 2*cO2in - cO2[nin_start4-1:nin_end4+1, 1]
    cG[nin_start4-1:nin_end4+1, 0] = 2*cGin - cG[nin_start4-1:nin_end4+1, 1]
    cO2[nin_start5-1:nin_end5+1, -1] = 2*cO2in - cO2[nin_start5-1:nin_end5+1, -2]
    cG[nin_start5-1:nin_end5+1, -1] = 2*cGin - cG[nin_start5-1:nin_end5+1, -2]
    cO2[nin_start6-1:nin_end6+1, 0] = 2*cO2in - cO2[nin_start6-1:nin_end6+1, 1]
    cG[nin_start6-1:nin_end6+1, 0] = 2*cGin - cG[nin_start6-1:nin_end6+1, 1]
    cO2[nin_start7-1:nin_end7+1, -1] = 2*cO2in - cO2[nin_start7-1:nin_end7+1, -2]
    cG[nin_start7-1:nin_end7+1, -1] = 2*cGin - cG[nin_start7-1:nin_end7+1, -2]
    cO2[nin_start8-1:nin_end8+1, 0] = 2*cO2in - cO2[nin_start8-1:nin_end8+1, 1]
    cG[nin_start8-1:nin_end8+1, 0] = 2*cGin - cG[nin_start8-1:nin_end8+1, 1]
    cO2[nin_start9-1:nin_end9+1, -1] = 2*cO2in - cO2[nin_start9-1:nin_end9+1, -2]
    cG[nin_start9-1:nin_end9+1, -1] = 2*cGin - cG[nin_start9-1:nin_end9+1, -2]

    # Advection-Diffusion equation
    cO2star = AdvDiffSpecies(cO2, u, v, dt, hx, hy, Gamma, nx, ny, flagp, method)
    cGstar = AdvDiffSpecies(cG, u, v, dt, hx, hy, Gamma, nx, ny, flagp, method)
    cBstar = cB

    # Reaction step (Linearization + Segregation)
    cO2, cB, cG = ReactionStep(cO2star, cBstar, cGstar, dt, mu_max, K_G, nx, ny)

    # Collecting cells mean values for graphical purposes
    cBmean2[it-1] = CellsMeanConcentration(cB, th, X2)   # (pay attention if cells are above or under the plate!!!)
    cBmean3[it-1] = CellsMeanConcentration(cB, th, X3)
    cBmean4[it-1] = CellsMeanConcentration(cB, th, X4)
    cBmean5[it-1] = CellsMeanConcentration(cB, th, X5)
    cBmean6[it-1] = CellsMeanConcentration(cB, th, X6)
    cBmean7[it-1] = CellsMeanConcentration(cB, th, X7)
    cBmean8[it-1] = CellsMeanConcentration(cB, th, X8)
    cBmean9[it-1] = CellsMeanConcentration(cB, th, X9)
    cBmean10[it-1] = CellsMeanConcentration(cB, th, X10)
    

    # Collecting time values for graphical purposes
    t_vector[it-1] = t

    # Advance in time
    t = t + dt

# Shear stress over the entire domain
dR = Obstacles(x, y)[10]
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
'''ccB[xe1+1:xe1+th+1, ys1:ye1] = cB[xe1+1:xe1+th+1, ys1+1:ye1+1]  # Obstacle 1
ccB[xe2+1:xe2+th+1, ys2:ye2] = cB[xe2+1:xe2+th+1, ys2+1:ye2+1]  # Obstacle 2
ccB[xe3+1:xe3+th+1, ys3:ye3] = cB[xe3+1:xe3+th+1, ys3+1:ye3+1]  # Obstacle 3
ccB[xe4+1:xe4+th+1, ys4:ye4] = cB[xe4+1:xe4+th+1, ys4+1:ye4+1]  # Obstacle 4
ccB[xe5+1:xe5+th+1, ys5:ye5] = cB[xe5+1:xe5+th+1, ys5+1:ye5+1]  # Obstacle 5
ccB[xe6+1:xe6+th+1, ys6:ye6] = cB[xe6+1:xe6+th+1, ys6+1:ye6+1]  # Obstacle 6
ccB[xe7+1:xe7+th+1, ys7:ye7] = cB[xe7+1:xe7+th+1, ys7+1:ye7+1]  # Obstacle 7'''
ccB[xs2-th:xs2, ys2:ye2] = cB[xs2-th+1:xs2+1, ys2+1:ye2+1]  # Obstacle 2
ccB[xs3-th:xs3, ys3:ye3] = cB[xs3-th+1:xs3+1, ys3+1:ye3+1]  # Obstacle 3
ccB[xs4-th:xs4, ys4:ye4] = cB[xs4-th+1:xs4+1, ys4+1:ye4+1]  # Obstacle 4
ccB[xs5-th:xs5, ys5:ye5] = cB[xs5-th+1:xs5+1, ys5+1:ye5+1]  # Obstacle 5
ccB[xs6-th:xs6, ys6:ye6] = cB[xs6-th+1:xs6+1, ys6+1:ye6+1]  # Obstacle 6
ccB[xs7-th:xs7, ys7:ye7] = cB[xs7-th+1:xs7+1, ys7+1:ye7+1]  # Obstacle 7
ccB[xs8-th:xs8, ys8:ye8] = cB[xs8-th+1:xs8+1, ys8+1:ye8+1]  # Obstacle 8
ccB[xs9-th:xs9, ys9:ye9] = cB[xs9-th+1:xs9+1, ys9+1:ye9+1]  # Obstacle 7
ccB[xs10-th:xs10, ys10:ye10] = cB[xs10-th+1:xs10+1, ys10+1:ye10+1]  # Obstacle 8

# Addition of obstacles for graphical purposes
uu, vv, pp, tau, ccO2, ccB, ccG = Graphical_obstacles(uu, vv, pp, tau, ccO2, ccB, ccG, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10)

# Creating a grid
xx,yy = np.meshgrid(x,y)

# Plotting the results
# Surface map: pressure
PlotFunctions(xx, yy, pp, x, y, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, Lx, Ly, 'Delta Pressure [Pa]', 'x [m]', 'y [m]')

# Surface map: u-velocity
PlotFunctions(xx, yy, uu, x, y, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, Lx, Ly, 'u - velocity [m/s]', 'x [m]', 'y [m]')

# Surface map: v-velocity
PlotFunctions(xx, yy, vv, x, y, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, Lx, Ly, 'v - velocity [m/s]', 'x [m]', 'y [m]')

# Surface map: cO2
PlotFunctions(xx, yy, ccO2, x, y, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, Lx, Ly, 'O2 concentration [mol/m3]', 'x [m]', 'y [m]')

# Surface map: cG
PlotFunctions(xx, yy, ccG, x, y, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, Lx, Ly, 'Glucose concentration [mol/m3]', 'x [m]', 'y [m]')

# Surface map: cB
PlotFunctions(xx, yy, ccB, x, y, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, Lx, Ly, 'Cells concentration [mol/m3]', 'x [m]', 'y [m]')

# Surface map: Tau
PlotFunctions(xx, yy, tau, x, y, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, Lx, Ly, 'Shear stress [N/m2]', 'x [m]', 'y [m]')

# Streamlines
Streamlines(x, y, xx, yy, uu, vv, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, Lx, Ly, 'Streamlines', 'x [m]', 'y [m]')

# Mean values of cB
MeanCellsPlot(t_vector, cBmean1, cBmean2, cBmean3, cBmean4, cBmean5, cBmean6, cBmean7, cBmean8, cBmean9, cBmean10, Time, np.max(cBmean2)*1.5)

plt.show()