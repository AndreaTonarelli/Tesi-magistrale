import numpy as np
import matplotlib.pyplot as plt
from   numba import njit
from   functions_10 import *
from   Obstacles_def_1in_u import *
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
nx = 800
ny = 300
nu = 1e-6          # m2/s  (dynamic viscosity of fluid)
Gamma = 1e-5       # m2/s  (diffusion coefficient of O2 in water)
u_in = 1e-3        # m/s   (inlet velocity)
Time = 1800.0      # s     (simulation time)
method = 'Upwind'  # Discretization method (CDS or Upwind)

# Boundary conditions (no-slip)
un = 0             # m/s
us = 0             # m/s
ve = 0             # m/s
vw = 0             # m/s

# Inlet concentrations
cO2in = 10.        # mol/m3
cGlcin = 100.      # mol/m3
cGlnin = 100.      # mol/m3

# Immobilized cells parameters
Xv0 = 1e8             # cells (initial number of viable cells)
thick = 0.05          # m (thickness of cells layer)
mu_max = 0.1/3600     # 1/s (max growth rate) -> 0.029 1/h
k_d = 0.02/3600       # 1/s (max death rate) -> 0.016 1/h
K_Glc = 0.084         # mol/m3 (Monod const. of glucose)
K_Gln = 0.047         # mol/m3 (Monod const. of glutamine)
KI_Amm = 6.51         # mol/m3 (Monod const. of ammonia)
KI_Lac = 43.0         # mol/m3 (Monod const. of lactose)
KD_Amm = 6.51         # mol/m3 (Inhibition const. of ammonia)
KD_Lac = 45.8         # mol/m3 (Inhibition const. of lactose)
Y_Glc = 1.69e11       # cells/mol_Glc
Y_Gln = 9.74e11       # cells/mol_Gln
Y_Lac = 1.23          # mol_Lac/mol_Glc
Y_Amm = 0.67          # mol_Amm/mol_Gln
parameters = List([mu_max, k_d, K_Glc, K_Gln, KI_Amm, KI_Lac, KD_Amm, KD_Lac, Y_Glc, Y_Gln, Y_Lac, Y_Amm])

# Parameters for SOR/SUR (Poisson eq)
max_iterations = 1000000
beta = 0.5
max_error = 1e-6

## -------------------------------------------------------------------------------------------- ##
##                                     DATA PROCESSING                                          ##
## -------------------------------------------------------------------------------------------- ##
if nx % 2 != 0 or ny % 2 != 0:
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
XX = Obstacles(x, y)[0]
X1 = XX[0]; xs1 = X1[0]; xe1 = X1[1]; ys1 = X1[2]; ye1 = X1[3]
X2 = XX[1]; xs2 = X2[0]; xe2 = X2[1]; ys2 = X2[2]; ye2 = X2[3]
X3 = XX[2]; xs3 = X3[0]; xe3 = X3[1]; ys3 = X3[2]; ye3 = X3[3]
X4 = XX[3]; xs4 = X4[0]; xe4 = X4[1]; ys4 = X4[2]; ye4 = X4[3]
X5 = XX[4]; xs5 = X5[0]; xe5 = X5[1]; ys5 = X5[2]; ye5 = X5[3]
X6 = XX[5]; xs6 = X6[0]; xe6 = X6[1]; ys6 = X6[2]; ye6 = X6[3]
X7 = XX[6]; xs7 = X7[0]; xe7 = X7[1]; ys7 = X7[2]; ye7 = X7[3]
X8 = XX[7]; xs8 = X8[0]; xe8 = X8[1]; ys8 = X8[2]; ye8 = X8[3]
X9 = XX[8]; xs9 = X9[0]; xe9 = X9[1]; ys9 = X9[2]; ye9 = X9[3]
X10 = XX[9]; xs10 = X10[0]; xe10 = X10[1]; ys10 = X10[2]; ye10 = X10[3]

# Inlet section (west side)
nin_start1 = ye1 + 1   # first cell index (pay attention cause it is in MATLAB notation!)
nin_end1 = ny + 1      # last cell index (pay attention cause it is in MATLAB notation!)

# Outlet section (east side)
nout_start = 2        # first cell index (pay attention cause it is in MATLAB notation!)
nout_end = ys8 - 1    # last cell index (pay attention cause it is in MATLAB notation!)

# Time step
sigma = 0.75                                        # safety factor for time step (stability)
dt_diff_ns = np.minimum(hx,hy)**2/4/nu             # time step (diffusion stability) [s]
dt_conv_ns = 4*nu/u_in**2                          # time step (convection stability) [s]
dt_ns      = np.minimum(dt_diff_ns, dt_conv_ns)    # time step (stability due to FD) [s]
dt_diff_sp = np.minimum(hx,hy)**2/4/Gamma          # time step (species diffusion stability) [s]
dt_conv_sp = 4*Gamma/u_in**2                       # time step (species convection stability) [s]
dt_sp      = np.minimum(dt_conv_sp, dt_diff_sp)    # time step (stability due to species) [s]
dt         = sigma*min(dt_ns, dt_sp)               # time step (stability) [s]
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
u    = np.zeros([nx+1, ny+2]); v    = np.zeros([nx+2, ny+1]); p    = np.zeros([nx+2, ny+2])
cO2  = np.zeros([nx+2, ny+2]); Xv   = np.zeros([nx+2, ny+2]); cGlc = np.zeros([nx+2, ny+2])
cGln = np.zeros([nx+2, ny+2]); cAmm = np.zeros([nx+2, ny+2]); cLac = np.zeros([nx+2, ny+2])

# Temporary velocity fields
ut = np.zeros_like(u)
vt = np.zeros_like(v)

# Fields used only for graphical purposes
uu    = np.zeros([nx+1, ny+1]); vv    = np.zeros([nx+1, ny+1]); pp    = np.zeros([nx+1, ny+1])
ccO2  = np.zeros([nx+1, ny+1]); ccXv  = np.zeros([nx+1, ny+1]); ccGlc = np.zeros([nx+1, ny+1])
ccGln = np.zeros([nx+1, ny+1]); ccAmm = np.zeros([nx+1, ny+1]); ccLac = np.zeros([nx+1, ny+1])
cBmean1 = np.zeros([nsteps]); cBmean2 = np.zeros([nsteps]); cBmean3 = np.zeros([nsteps])
cBmean4 = np.zeros([nsteps]); cBmean5 = np.zeros([nsteps]); cBmean6 = np.zeros([nsteps])
cBmean7 = np.zeros([nsteps]); cBmean8 = np.zeros([nsteps]); cBmean9 = np.zeros([nsteps])
cBmean10 = np.zeros([nsteps])
t_vector = np.zeros([nsteps])

# Coefficients for pressure equation
gamma = np.zeros([nx+2,ny+2]) + hx*hy / (2*hx**2 + 2*hy**2)   # internal points
gamma = gammaCoeff(gamma, hx,hy, nout_start,nout_end, XX)

# Flags to recognize where is necessary to solve the equations
# This flag is 1 in the cells that contain and obstacle and 0 in the others
flagu = np.zeros([nx+1, ny+2])          # u-cells corresponding to the obstacle
flagv = np.zeros([nx+2, ny+1])          # v-cells corresponding to the obstacle
flagp = np.zeros([nx+2, ny+2])          # p-cells corresponding to the obstacle

# Set flag to 1 in obstacle cells
flagu, flagv, flagp = flag(flagu,flagv,flagp, XX)

# Initial conditions: set reasonable initial velocity value instead of initializing everything to zero
u[:, :] = u_in/2                   # Internal points: fixed velocity [m/s]
u = u_initialize(u, XX)            # set u = 0 over obstacles
ut = u

# Immobilized cells initialized over obstacles
Xv[xs2-th:xs2, ys2:ye2] = Xv0      # Obstacle 2
Xv[xs3-th:xs3, ys3:ye3] = Xv0      # Obstacle 3
Xv[xs4-th:xs4, ys4:ye4] = Xv0      # Obstacle 4
Xv[xs5-th:xs5, ys5:ye5] = Xv0      # Obstacle 5
Xv[xs6-th:xs6, ys6:ye6] = Xv0      # Obstacle 6
Xv[xs7-th:xs7, ys7:ye7] = Xv0      # Obstacle 7
Xv[xs8-th:xs8, ys8:ye8] = Xv0      # Obstacle 8
Xv[xs9-th:xs9, ys9:ye9] = Xv0    # Obstacle 9
Xv[xs10-th:xs10, ys10:ye10] = Xv0    # Obstacle 10

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
    u, v = VelocityBCs(u, v, uwall, XX)

    # Over-writing inlet conditions
    u[0, nin_start1-1:nin_end1+1] = u_in     # fixed inlet velocity (west)

    # Over-writing outlet conditions
    u[-1, nout_start-1:nout_end+1] = u[-2, nout_start-1:nout_end+1]   # zero-gradient outlet velocity
    v[-1, nout_start-1:nout_end+1] = v[-2, nout_start-1:nout_end+1]   # zero-gradient outlet velocity

    # Advection-diffusion equation (predictor)
    ut, vt = AdvectionDiffusion2D(ut, vt, u, v, nx, ny, hx, hy, dt, nu, g, flagu, flagv, method)

    # Update boundary conditions for temporary velocities
    ut[0, nin_start1-1:nin_end1+1] = u_in     # fixed inlet velocity (west)
    ut[-1, nout_start-1:nout_end+1] = u[-1, nout_start-1:nout_end+1]   # zero-gradient outlet velocity
    vt[-1, nout_start-1:nout_end+1] = v[-1, nout_start-1:nout_end+1]   # zero-gradient outlet velocity

    # Pressure equation (Poisson)
    p, iter = Pressure_Poisson(p, ut, vt, gamma, nx, ny, hx, hy, dt, beta, max_iterations, max_error, flagp)

    # Correction on the velocity
    u, v = correction_velocity(u, v, ut, vt, p, nx, ny, hx, hy, dt, flagu, flagv)

    # Print on the screen
    if it <= 20:
        if it % 1 == 0:
            print('Step:', it, '- Time:', round(t,1),  '- Poisson iterations:', iter)
    else :
        if it % 500 == 0:
            print('Step:', it, '- Time:', round(t,1),  '- Poisson iterations:', iter)
            
    #----------------------------------------------------------------------------------#
    # 2. Transport of species and reaction                                             #
    #----------------------------------------------------------------------------------#
    # Impermeable walls
    cO2  = SpeciesBCs(cO2, XX)
    cGlc = SpeciesBCs(cGlc, XX)
    cGln = SpeciesBCs(cGln, XX)
    cLac = SpeciesBCs(cLac, XX)
    cAmm = SpeciesBCs(cAmm, XX)

    # Inlet sections
    cO2   = InletConcentration(cO2, cO2in, nin_start1, nin_end1, 'west')
    cGlc  = InletConcentration(cGlc, cGlcin, nin_start1, nin_end1, 'west')
    cGln  = InletConcentration(cGln, cGlnin, nin_start1, nin_end1, 'west')

    # Advection-Diffusion equation
    cO2star = AdvDiffSpecies(cO2, u, v, dt, hx, hy, Gamma, nx, ny, flagp, method)
    cGlcstar = AdvDiffSpecies(cGlc, u, v, dt, hx, hy, Gamma, nx, ny, flagp, method)
    cGlnstar = AdvDiffSpecies(cGln, u, v, dt, hx, hy, Gamma, nx, ny, flagp, method)
    cLacstar = AdvDiffSpecies(cLac, u, v, dt, hx, hy, Gamma, nx, ny, flagp, method)
    cAmmstar = AdvDiffSpecies(cAmm, u, v, dt, hx, hy, Gamma, nx, ny, flagp, method)
    Xvstar = Xv

    # Reaction step (Linearization + Segregation)
    cO2, Xv, cGlc, cGln, cLac, cAmm = ReactionStep(cO2star, Xvstar, cGlcstar, cGlnstar, cLacstar, cAmmstar, dt, parameters, nx, ny)

    # Collecting cells mean values for graphical purposes
    cBmean2[it-1] = CellsMeanConcentration(Xv, th, X2)   # (pay attention if cells are above or under the plate!!!)
    cBmean3[it-1] = CellsMeanConcentration(Xv, th, X3)
    cBmean4[it-1] = CellsMeanConcentration(Xv, th, X4)
    cBmean5[it-1] = CellsMeanConcentration(Xv, th, X5)
    cBmean6[it-1] = CellsMeanConcentration(Xv, th, X6)
    cBmean7[it-1] = CellsMeanConcentration(Xv, th, X7)
    cBmean8[it-1] = CellsMeanConcentration(Xv, th, X8)
    cBmean9[it-1] = CellsMeanConcentration(Xv, th, X9)
    cBmean10[it-1] = CellsMeanConcentration(Xv, th, X10)
    
    # Collecting time values for graphical purposes
    t_vector[it-1] = t

    # Advance in time
    t = t + dt

# Shear stress over the entire domain
dR = Obstacles(x, y)[-1]
Tau_wall = ShearStress(v, dR, nu, rhoL, nx, ny)


## -------------------------------------------------------------------------------------------- ##
##                                    FINAL POST-PROCESSING                                     ##
## -------------------------------------------------------------------------------------------- ##
# Field reconstruction
uu    = node_interp(u, 'u', nx, ny, flagu)
vv    = node_interp(v, 'v', nx, ny, flagv)
pp    = node_interp(p, 'p', nx, ny, flagp) * rhoL
tau   = node_interp(Tau_wall, 'v', nx, ny, flagv)
ccO2  = node_interp(cO2, 'p', nx, ny, flagp)
ccGlc = node_interp(cGlc, 'p', nx, ny, flagp)
ccGln = node_interp(cGln, 'p', nx, ny, flagp)
ccLac = node_interp(cLac, 'p', nx, ny, flagp)
ccAmm = node_interp(cAmm, 'p', nx, ny, flagp)

# Interpolation of cells concentration needed for graphical purposes
ccXv = Interpolation_of_cells(ccXv, Xv, X2, th, 'left')  # Obstacle 2
ccXv = Interpolation_of_cells(ccXv, Xv, X3, th, 'left')  # Obstacle 3
ccXv = Interpolation_of_cells(ccXv, Xv, X4, th, 'left')  # Obstacle 4
ccXv = Interpolation_of_cells(ccXv, Xv, X5, th, 'left')  # Obstacle 5
ccXv = Interpolation_of_cells(ccXv, Xv, X6, th, 'left')  # Obstacle 6
ccXv = Interpolation_of_cells(ccXv, Xv, X7, th, 'left')  # Obstacle 7
ccXv = Interpolation_of_cells(ccXv, Xv, X8, th, 'left')  # Obstacle 8
ccXv = Interpolation_of_cells(ccXv, Xv, X9, th, 'left')  # Obstacle 9
ccXv = Interpolation_of_cells(ccXv, Xv, X10, th, 'left')  # Obstacle 10

# Addition of obstacles for graphical purposes
uu, vv, pp, tau, ccO2, ccXv, ccGlc, ccGln, ccLac, ccAmm = Graphical_obstacles(uu, vv, pp, tau, ccO2, ccXv, ccGlc, ccGln, ccLac, ccAmm, XX)

# Creating a grid
xx,yy = np.meshgrid(x,y)

# Plotting the results
# Surface map: pressure
PlotFunctions(xx, yy, pp, x, y, XX, Lx, Ly, 'Delta Pressure [Pa]', 'x [m]', 'y [m]')

# Surface map: u-velocity
PlotFunctions(xx, yy, uu, x, y, XX, Lx, Ly, 'u - velocity [m/s]', 'x [m]', 'y [m]')

# Surface map: v-velocity
PlotFunctions(xx, yy, vv, x, y, XX, Lx, Ly, 'v - velocity [m/s]', 'x [m]', 'y [m]')

# Surface map: cO2
PlotFunctions(xx, yy, ccO2, x, y, XX, Lx, Ly, 'O2 concentration [mol/m3]', 'x [m]', 'y [m]')

# Surface map: cGlc
PlotFunctions(xx, yy, ccGlc, x, y, XX, Lx, Ly, 'Glucose concentration [mol/m3]', 'x [m]', 'y [m]')

# Surface map: cGln
PlotFunctions(xx, yy, ccGln, x, y, XX, Lx, Ly, 'Glutamine concentration [mol/m3]', 'x [m]', 'y [m]')

# Surface map: cLac
PlotFunctions(xx, yy, ccLac, x, y, XX, Lx, Ly, 'Lactose concentration [mol/m3]', 'x [m]', 'y [m]')

# Surface map: cAmm
PlotFunctions(xx, yy, ccAmm, x, y, XX, Lx, Ly, 'Ammonia concentration [mol/m3]', 'x [m]', 'y [m]')

# Surface map: Xv
PlotFunctions(xx, yy, ccXv, x, y, XX, Lx, Ly, 'Viable cells number [cells]', 'x [m]', 'y [m]')

# Surface map: Tau
PlotFunctions(xx, yy, tau, x, y, XX, Lx, Ly, 'Shear stress [N/m2]', 'x [m]', 'y [m]')

# Streamlines
Streamlines(x, y, xx, yy, uu, vv, XX, Lx, Ly, 'Streamlines', 'x [m]', 'y [m]')

# Mean values of cB
MeanCellsPlot(t_vector, [cBmean1, cBmean2, cBmean3, cBmean4, cBmean5, cBmean6, cBmean7, cBmean8, cBmean9, cBmean10], Time, np.max(cBmean2)*1.5)

plt.show()