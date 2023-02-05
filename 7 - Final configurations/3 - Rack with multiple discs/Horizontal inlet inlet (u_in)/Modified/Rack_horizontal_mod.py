import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from   numba import njit
from   numba.typed import List
from   Rack_horizontal_mod_functions import *
from   Rack_horizontal_mod_obstacles_def import *
plt.style.use(['science', 'no-latex'])
import time
start = time.time()

## -------------------------------------------------------------------------------------------- ##
##                                     PRE-PROCESSING                                           ##
## -------------------------------------------------------------------------------------------- ##
# Data
P = 101325.0       # Pa
T = 25+273.15      # K
g = 9.80665        # m/s2  (g = 9.80665)
rhoL = 1000.       # kg/m3

Lx = 6.0           # m
Ly = 2.0           # m
nx = 1200
ny = 400
nu = 1e-6          # m2/s  (dynamic viscosity of fluid)
Gamma = 1e-6       # m2/s  (diffusion coefficient of O2 in water)
v_in = 1e-3        # m/s   (inlet velocity)
Time = 86400.      # s     (simulation time)
method = 'Upwind'  # Discretization method (CDS or Upwind)
OnTheFly = 'No'    # On-the-fly plotting (if 'Yes')

# Boundary conditions (no-slip)
un = 0             # m/s
us = 0             # m/s
ve = 0             # m/s
vw = 0             # m/s

# Inlet concentrations
cO2in = 100.       # mol/m3
cGlcin = 100.       # mol/m3
cGlnin = 100.        # mol/m3

# Immobilized cells parameters
Xv0 = 1e4            # cells/m3 (initial number of viable cells)
cB1_0 = 1.            # mol/m3 (initial concentration of B1)
thick = 0.05          # m (thickness of cells layer)
mu_max = 0.029/3600   # 1/s (max growth rate)
k_d = 0.016/3600      # 1/s (max death rate)
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
Q_B1 = 4e-15/3600     # m3/cell/s
q_O2 = 2e-13/3600     # mol_O2/cells/s
parameters = List([mu_max, k_d, K_Glc, K_Gln, KI_Amm, KI_Lac, KD_Amm, KD_Lac, Y_Glc, Y_Gln, Y_Lac, Y_Amm, Q_B1, q_O2])

# Parameters for SOR/SUR (Poisson eq)
max_iterations = 1000000
beta = 1.
max_error = 3e-10

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
XX = Obstacles(x, y, Lx, Ly)[0]
X2_1 = XX[1]; X3_1 = XX[2]; X4_1 = XX[3]; X5_1 = XX[4]; X6_1 = XX[5]; X7_1 = XX[6]; X8_1 = XX[7]; X9_1 = XX[8]
X10_1 = XX[9]; X11_1 = XX[10]; X12_1 = XX[11]; X13_1 = XX[12]; X14_1 = XX[13]; X15_1 = XX[14]; X16_1 = XX[15]; X17_1 = XX[16]
X2_2 = XX[17]; X3_2 = XX[18]; X4_2 = XX[19]; X5_2 = XX[20]; X6_2 = XX[21]; X7_2 = XX[22]; X8_2 = XX[23]; X9_2 = XX[24]
X10_2 = XX[25]; X11_2 = XX[26]; X12_2 = XX[27]; X13_2 = XX[28]; X14_2 = XX[29]; X15_2 = XX[30]; X16_2 = XX[31]; X17_2 = XX[32]
X32 = XX[-2]; xs32 = X32[0]; xe32 = X32[1]; ys32 = X32[2]; ye32 = X32[3]
X33 = XX[-1]; xs32 = X33[0]; xe33 = X33[1]; ys33 = X33[2]; ye33 = X33[3]
xs_entrance1, xe_entrance1, xs_entrance2, xe_entrance2, xs_entrance3, xe_entrance3, xs_entrance4, xe_entrance4,\
xs_entrance5, xe_entrance5, xs_entrance6, xe_entrance6, xs_entrance7, xe_entrance7, xs_entrance8, xe_entrance8,\
xs_entrance9, xe_entrance9, xs_entrance10, xe_entrance10, xs_entrance11, xe_entrance11, xs_entrance12, xe_entrance12,\
xs_entrance13, xe_entrance13, xs_entrance14, xe_entrance14, xs_entrance15, xe_entrance15, xs_entrance16, xe_entrance16 = Obstacles(x, y, Lx, Ly)[2]

# Inlet sections (north/south side)
nin_start1 = xs_entrance1; nin_end1 = xe_entrance1 - 1  # (pay attention to Python INDEX!!!)
nin_start2 = xs_entrance2; nin_end2 = xe_entrance2 - 1  # (pay attention to Python INDEX!!!)
nin_start3 = xs_entrance3; nin_end3 = xe_entrance3 - 1  # (pay attention to Python INDEX!!!)
nin_start4 = xs_entrance4; nin_end4 = xe_entrance4 - 1  # (pay attention to Python INDEX!!!)
nin_start5 = xs_entrance5; nin_end5 = xe_entrance5 - 1  # (pay attention to Python INDEX!!!)
nin_start6 = xs_entrance6; nin_end6 = xe_entrance6 - 1  # (pay attention to Python INDEX!!!)
nin_start7 = xs_entrance7; nin_end7 = xe_entrance7 - 1  # (pay attention to Python INDEX!!!)
nin_start8 = xs_entrance8; nin_end8 = xe_entrance8 - 1  # (pay attention to Python INDEX!!!)
nin_start9 = xs_entrance9; nin_end9 = xe_entrance9 - 1  # (pay attention to Python INDEX!!!)
nin_start10 = xs_entrance10; nin_end10 = xe_entrance10 - 1  # (pay attention to Python INDEX!!!)
nin_start11 = xs_entrance11; nin_end11 = xe_entrance11 - 1  # (pay attention to Python INDEX!!!)
nin_start12 = xs_entrance12; nin_end12 = xe_entrance12 - 1  # (pay attention to Python INDEX!!!)
nin_start13 = xs_entrance13; nin_end13 = xe_entrance13 - 1  # (pay attention to Python INDEX!!!)
nin_start14 = xs_entrance14; nin_end14 = xe_entrance14 - 1  # (pay attention to Python INDEX!!!)
nin_start15 = xs_entrance15; nin_end15 = xe_entrance15 - 1  # (pay attention to Python INDEX!!!)
nin_start16 = xs_entrance16; nin_end16 = xe_entrance16 - 1  # (pay attention to Python INDEX!!!)

# Outlet section (east side)
nout_start = ye32 + 1    # first cell index (0 is the first one!!!)
nout_end = ys33 - 1      # last cell index

# Inlet/Outlet section areas and outlet velocity
Ain1 = hy*(nin_end1-nin_start1+1); Ain2 = hy*(nin_end2-nin_start2+1); Ain3 = hy*(nin_end3-nin_start3+1); Ain4 = hy*(nin_end4-nin_start4+1)   # inlet section area [m]
Ain5 = hy*(nin_end5-nin_start5+1); Ain6 = hy*(nin_end6-nin_start6+1); Ain7 = hy*(nin_end7-nin_start7+1); Ain8 = hy*(nin_end8-nin_start8+1)
Ain9 = hy*(nin_end9-nin_start9+1); Ain10 = hy*(nin_end10-nin_start10+1); Ain11 = hy*(nin_end11-nin_start11+1); Ain12 = hy*(nin_end12-nin_start12+1)
Ain13 = hy*(nin_end13-nin_start13+1); Ain14 = hy*(nin_end14-nin_start14+1); Ain15 = hy*(nin_end15-nin_start15+1); Ain16 = hy*(nin_end16-nin_start16+1)
Aout = hy*(nout_end-nout_start+1)       # outlet section area [m]
u_out = (Ain1+Ain2+Ain3+Ain4+Ain5+Ain6+Ain7+Ain8+Ain9+Ain10+Ain11+Ain12+Ain13+Ain14+Ain15+Ain16)*v_in/Aout

# Time step
sigma = 0.5                                        # safety factor for time step (stability)
dt_diff_ns = np.minimum(hx,hy)**2/4/nu             # time step (diffusion stability) [s]
dt_conv_ns = 4*nu/v_in**2                          # time step (convection stability) [s]
dt_ns      = np.minimum(dt_diff_ns, dt_conv_ns)    # time step (stability due to FD) [s]
dt_diff_sp = np.minimum(hx,hy)**2/4/Gamma          # time step (species diffusion stability) [s]
dt_conv_sp = 4*Gamma/v_in**2                       # time step (species convection stability) [s]
dt_sp      = np.minimum(dt_conv_sp, dt_diff_sp)    # time step (stability due to species) [s]
dt         = sigma*min(dt_ns, dt_sp)               # time step (stability) [s]
nsteps     = int(Time/dt)*1000                     # number of steps
Re         = v_in*Ly/nu                            # Reynolds' number

print('Time step:', round(dt,1), 's' )
print(' - Diffusion (NS):', round(dt_diff_ns,1), 's')
print(' - Convection (NS):', round(dt_conv_ns,1), 's')
print(' - Diffusion (Species):', round(dt_diff_sp,1), 's')
print(' - Convection (Species):', round(dt_conv_sp,1), 's')
print('Reynolds number:', round(Re,1), '\n')

## -------------------------------------------------------------------------------------------- ##
##                                     MEMORY ALLOCATION                                        ##
## -------------------------------------------------------------------------------------------- ##
# Main fields (velocities and pressure)
u    = np.zeros([nx+1, ny+2]); v    = np.zeros([nx+2, ny+1]); p    = np.zeros([nx+2, ny+2])
cO2  = np.zeros([nx+2, ny+2]); Xv   = np.zeros([nx+2, ny+2]); cGlc = np.zeros([nx+2, ny+2])
cGln = np.zeros([nx+2, ny+2]); cAmm = np.zeros([nx+2, ny+2]); cLac = np.zeros([nx+2, ny+2]); cB1 = np.zeros([nx+2, ny+2])

# Temporary velocity fields
ut = np.zeros_like(u)
vt = np.zeros_like(v)

# Fields used only for graphical purposes
uu    = np.zeros([nx+1, ny+1]); vv    = np.zeros([nx+1, ny+1]); pp    = np.zeros([nx+1, ny+1])
ccO2  = np.zeros([nx+1, ny+1]); ccXv  = np.zeros([nx+1, ny+1]); ccGlc = np.zeros([nx+1, ny+1])
ccGln = np.zeros([nx+1, ny+1]); ccAmm = np.zeros([nx+1, ny+1]); ccLac = np.zeros([nx+1, ny+1]); ccB1 = np.zeros([nx+1, ny+1])
cBmean2 = cBmean3 = cBmean4 = cBmean5 = cBmean6 = cBmean7 = cBmean8 = cBmean9 = cBmean10 = np.empty(0)
cBmean11 = cBmean12 = cBmean13 = cBmean14 = cBmean15 = cBmean16 = cBmean17 = cB1out = t_vector = np.empty(0)
cO2_max = cGlc_max = cGln_max = cAmm_max = cLac_max = cB1_max = u_max = v_max = np.empty(0)

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
v[nin_start1-1:nin_end1+1, 0] = v_in;   v[nin_start2-1:nin_end2+1, -1] = -v_in     # fixed inlet velocity (south/north)
v[nin_start3-1:nin_end3+1, 0] = v_in;   v[nin_start4-1:nin_end4+1, -1] = -v_in     # fixed inlet velocity (south/north)
v[nin_start5-1:nin_end5+1, 0] = v_in;   v[nin_start6-1:nin_end6+1, -1] = -v_in     # fixed inlet velocity (south/north)
v[nin_start7-1:nin_end7+1, 0] = v_in;   v[nin_start8-1:nin_end8+1, -1] = -v_in     # fixed inlet velocity (south/north)
v[nin_start9-1:nin_end9+1, 0] = v_in;   v[nin_start10-1:nin_end10+1, -1] = -v_in   # fixed inlet velocity (south/north)
v[nin_start11-1:nin_end11+1, 0] = v_in; v[nin_start12-1:nin_end12+1, -1] = -v_in   # fixed inlet velocity (south/north)
v[nin_start13-1:nin_end13+1, 0] = v_in; v[nin_start14-1:nin_end14+1, -1] = -v_in   # fixed inlet velocity (south/north)
v[nin_start15-1:nin_end15+1, 0] = v_in; v[nin_start16-1:nin_end16+1, -1] = -v_in   # fixed inlet velocity (south/north)
u[-1, nout_start-1:nout_end+1] = u_out   # fixed outlet velocity (east)
''' u[1:-1, 1:-1] = v_in/2                   # Internal points: fixed velocity [m/s]
u = u_initialize(u, XX)                  # set u = 0 over obstacles '''
ut = u; vt = v

# Immobilized cells initialized over obstacles
cells_position = 'left'
Xv = ImmobilizedCells(Xv, Xv0, X2_1, X2_2, th, cells_position)  # Obstacle 2
Xv = ImmobilizedCells(Xv, Xv0, X3_1, X3_2, th, cells_position)  # Obstacle 3
Xv = ImmobilizedCells(Xv, Xv0, X4_1, X4_2, th, cells_position)  # Obstacle 4
Xv = ImmobilizedCells(Xv, Xv0, X5_1, X5_2, th, cells_position)  # Obstacle 5
Xv = ImmobilizedCells(Xv, Xv0, X6_1, X6_2, th, cells_position)  # Obstacle 6
Xv = ImmobilizedCells(Xv, Xv0, X7_1, X7_2, th, cells_position)  # Obstacle 7
Xv = ImmobilizedCells(Xv, Xv0, X8_1, X8_2, th, cells_position)  # Obstacle 8
Xv = ImmobilizedCells(Xv, Xv0, X9_1, X9_2, th, cells_position)  # Obstacle 9
Xv = ImmobilizedCells(Xv, Xv0, X10_1, X10_2, th, cells_position)  # Obstacle 10
Xv = ImmobilizedCells(Xv, Xv0, X11_1, X11_2, th, cells_position)  # Obstacle 11
Xv = ImmobilizedCells(Xv, Xv0, X12_1, X12_2, th, cells_position)  # Obstacle 12
Xv = ImmobilizedCells(Xv, Xv0, X13_1, X13_2, th, cells_position)  # Obstacle 13
Xv = ImmobilizedCells(Xv, Xv0, X14_1, X14_2, th, cells_position)  # Obstacle 14
Xv = ImmobilizedCells(Xv, Xv0, X15_1, X15_2, th, cells_position)  # Obstacle 15
Xv = ImmobilizedCells(Xv, Xv0, X16_1, X16_2, th, cells_position)  # Obstacle 16
Xv = ImmobilizedCells(Xv, Xv0, X17_1, X17_2, th, cells_position)  # Obstacle 17
cB1 = ImmobilizedCells(cB1, cB1_0, X2_1, X2_2, th, cells_position)  # Obstacle 2
cB1 = ImmobilizedCells(cB1, cB1_0, X3_1, X3_2, th, cells_position)  # Obstacle 3
cB1 = ImmobilizedCells(cB1, cB1_0, X4_1, X4_2, th, cells_position)  # Obstacle 4
cB1 = ImmobilizedCells(cB1, cB1_0, X5_1, X5_2, th, cells_position)  # Obstacle 5
cB1 = ImmobilizedCells(cB1, cB1_0, X6_1, X6_2, th, cells_position)  # Obstacle 6
cB1 = ImmobilizedCells(cB1, cB1_0, X7_1, X7_2, th, cells_position)  # Obstacle 7
cB1 = ImmobilizedCells(cB1, cB1_0, X8_1, X8_2, th, cells_position)  # Obstacle 8
cB1 = ImmobilizedCells(cB1, cB1_0, X9_1, X9_2, th, cells_position)  # Obstacle 9
cB1 = ImmobilizedCells(cB1, cB1_0, X10_1, X10_2, th, cells_position)  # Obstacle 10
cB1 = ImmobilizedCells(cB1, cB1_0, X11_1, X11_2, th, cells_position)  # Obstacle 11
cB1 = ImmobilizedCells(cB1, cB1_0, X12_1, X12_2, th, cells_position)  # Obstacle 12
cB1 = ImmobilizedCells(cB1, cB1_0, X13_1, X13_2, th, cells_position)  # Obstacle 13
cB1 = ImmobilizedCells(cB1, cB1_0, X14_1, X14_2, th, cells_position)  # Obstacle 14
cB1 = ImmobilizedCells(cB1, cB1_0, X15_1, X15_2, th, cells_position)  # Obstacle 15
cB1 = ImmobilizedCells(cB1, cB1_0, X16_1, X16_2, th, cells_position)  # Obstacle 16
cB1 = ImmobilizedCells(cB1, cB1_0, X17_1, X17_2, th, cells_position)  # Obstacle 17


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
    v[nin_start1-1:nin_end1+1, 0] = v_in;   v[nin_start2-1:nin_end2+1, -1] = -v_in     # fixed inlet velocity (south/north)
    v[nin_start3-1:nin_end3+1, 0] = v_in;   v[nin_start4-1:nin_end4+1, -1] = -v_in     # fixed inlet velocity (south/north)
    v[nin_start5-1:nin_end5+1, 0] = v_in;   v[nin_start6-1:nin_end6+1, -1] = -v_in     # fixed inlet velocity (south/north)
    v[nin_start7-1:nin_end7+1, 0] = v_in;   v[nin_start8-1:nin_end8+1, -1] = -v_in     # fixed inlet velocity (south/north)
    v[nin_start9-1:nin_end9+1, 0] = v_in;   v[nin_start10-1:nin_end10+1, -1] = -v_in   # fixed inlet velocity (south/north)
    v[nin_start11-1:nin_end11+1, 0] = v_in; v[nin_start12-1:nin_end12+1, -1] = -v_in   # fixed inlet velocity (south/north)
    v[nin_start13-1:nin_end13+1, 0] = v_in; v[nin_start14-1:nin_end14+1, -1] = -v_in   # fixed inlet velocity (south/north)
    v[nin_start15-1:nin_end15+1, 0] = v_in; v[nin_start16-1:nin_end16+1, -1] = -v_in   # fixed inlet velocity (south/north)

    # Over-writing outlet conditions
    u[-1, nout_start-1:nout_end+1] = u_out   # fixed outlet velocity (east)
    u[-1, nout_start-1:nout_end+1] = u[-2, nout_start-1:nout_end+1]   # zero-gradient outlet velocity
    v[-1, nout_start-1:nout_end+1] = v[-2, nout_start-1:nout_end+1]   # zero-gradient outlet velocity

    # Uptade the dt for stability conditions
    dt_conv_ns = 1/(np.max(u)/hx + np.max(v)/hy)       # time step (convection stability) [s]
    dt_ns      = np.minimum(dt_diff_ns, dt_conv_ns)    # time step (stability due to FD) [s]
    dt         = sigma*min(dt_ns, dt_sp)               # time step (stability) [s]

    # Advection-diffusion equation (predictor)
    ut, vt = AdvectionDiffusion2D(ut, vt, u, v, nx, ny, hx, hy, dt, nu, g, flagu, flagv, method)

    # Update boundary conditions for temporary velocities
    vt[nin_start1-1:nin_end1+1, 0] = v_in;   vt[nin_start2-1:nin_end2+1, -1] = -v_in     # fixed inlet velocity (south/north)
    vt[nin_start3-1:nin_end3+1, 0] = v_in;   vt[nin_start4-1:nin_end4+1, -1] = -v_in     # fixed inlet velocity (south/north)
    vt[nin_start5-1:nin_end5+1, 0] = v_in;   vt[nin_start6-1:nin_end6+1, -1] = -v_in     # fixed inlet velocity (south/north)
    vt[nin_start7-1:nin_end7+1, 0] = v_in;   vt[nin_start8-1:nin_end8+1, -1] = -v_in     # fixed inlet velocity (south/north)
    vt[nin_start9-1:nin_end9+1, 0] = v_in;   vt[nin_start10-1:nin_end10+1, -1] = -v_in   # fixed inlet velocity (south/north)
    vt[nin_start11-1:nin_end11+1, 0] = v_in; vt[nin_start12-1:nin_end12+1, -1] = -v_in   # fixed inlet velocity (south/north)
    vt[nin_start13-1:nin_end13+1, 0] = v_in; vt[nin_start14-1:nin_end14+1, -1] = -v_in   # fixed inlet velocity (south/north)
    vt[nin_start15-1:nin_end15+1, 0] = v_in; vt[nin_start16-1:nin_end16+1, -1] = -v_in   # fixed inlet velocity (south/north)
    ut[-1, nout_start-1:nout_end+1] = u_out   # fixed outlet velocity (east)
    ut[-1, nout_start-1:nout_end+1] = u[-1, nout_start-1:nout_end+1]   # zero-gradient outlet velocity
    vt[-1, nout_start-1:nout_end+1] = v[-1, nout_start-1:nout_end+1]   # zero-gradient outlet velocity

    # Pressure equation (Poisson)
    p, iter = Pressure_Poisson(p, ut, vt, gamma, nx, ny, hx, hy, dt, beta, max_iterations, max_error, flagp)

    # Correction on the velocity
    u, v = correction_velocity(u, v, ut, vt, p, nx, ny, hx, hy, dt, flagu, flagv)

    # Correct the velocity on the outlet patch
    u[-1, nout_start-1:nout_end+1] = ut[-1, nout_start-1:nout_end+1] - dt/hx*(p[-1,nout_start-1:nout_end+1] - p[-2,nout_start-1:nout_end+1])

    # It is better to correct the outlet velocity in order to force conservation of mass
    Qin1 = abs(np.mean(v[nin_start1-1:nin_end1+1, 0])*Ain1);     Qin2 = abs(np.mean(v[nin_start2-1:nin_end2+1, -1])*Ain2)    # inlet flow rate [m2/s]
    Qin3 = abs(np.mean(v[nin_start3-1:nin_end3+1, 0])*Ain3);     Qin4 = abs(np.mean(v[nin_start4-1:nin_end4+1, -1])*Ain4)
    Qin5 = abs(np.mean(v[nin_start5-1:nin_end5+1, 0])*Ain5);     Qin6 = abs(np.mean(v[nin_start6-1:nin_end6+1, -1])*Ain6)
    Qin7 = abs(np.mean(v[nin_start7-1:nin_end7+1, 0])*Ain7);     Qin8 = abs(np.mean(v[nin_start8-1:nin_end8+1, -1])*Ain8)
    Qin9 = abs(np.mean(v[nin_start9-1:nin_end9+1, 0])*Ain9);     Qin10 = abs(np.mean(v[nin_start10-1:nin_end10+1, -1])*Ain10)
    Qin11 = abs(np.mean(v[nin_start11-1:nin_end11+1, 0])*Ain11); Qin12 = abs(np.mean(v[nin_start12-1:nin_end12+1, -1])*Ain12)
    Qin13 = abs(np.mean(v[nin_start13-1:nin_end13+1, 0])*Ain13); Qin14 = abs(np.mean(v[nin_start14-1:nin_end14+1, -1])*Ain14)
    Qin15 = abs(np.mean(v[nin_start15-1:nin_end15+1, 0])*Ain15); Qin16 = abs(np.mean(v[nin_start16-1:nin_end16+1, -1])*Ain16)
    Qin = Qin1+Qin2+Qin3+Qin4+Qin5+Qin6+Qin7+Qin8+Qin9+Qin10+Qin11+Qin12+Qin13+Qin14+Qin15+Qin16
    Qout = np.mean(u[-1, nout_start-1:nout_end+1])*Aout    # outlet flow rate [m2/s]    
    if (abs(Qout)>1.e-6):
        u[-1, nout_start-1:nout_end+1] = u[-1, nout_start-1:nout_end+1] * abs(Qin/Qout)

    # Print on the screen
    if it <= 20:
        if it % 1 == 0:
            print('Step:', it, '- dt:', round(dt,1), '- Time:', round(t,1),  '- Poisson iterations:', iter)
    else :
        if it % 500 == 0:
            print('Step:', it, '- dt:', round(dt,1), '- Time:', round(t,1),  '- Poisson iterations:', iter)
            
    #----------------------------------------------------------------------------------#
    # 2. Transport of species and reaction                                             #
    #----------------------------------------------------------------------------------#
    # Impermeable walls
    cO2  = SpeciesBCs(cO2, XX)
    cGlc = SpeciesBCs(cGlc, XX)
    cGln = SpeciesBCs(cGln, XX)
    cLac = SpeciesBCs(cLac, XX)
    cAmm = SpeciesBCs(cAmm, XX)
    cB1  = SpeciesBCs(cB1, XX)

    # Inlet sections
    cO2   = InletConcentration(cO2, cO2in, nin_start1, nin_end1, 'south')
    cGlc  = InletConcentration(cGlc, cGlcin, nin_start1, nin_end1, 'south')
    cGln  = InletConcentration(cGln, cGlnin, nin_start1, nin_end1, 'south')
    cO2   = InletConcentration(cO2, cO2in, nin_start2, nin_end2, 'north')
    cGlc  = InletConcentration(cGlc, cGlcin, nin_start2, nin_end2, 'north')
    cGln  = InletConcentration(cGln, cGlnin, nin_start2, nin_end2, 'north')
    cO2   = InletConcentration(cO2, cO2in, nin_start3, nin_end3, 'south')
    cGlc  = InletConcentration(cGlc, cGlcin, nin_start3, nin_end3, 'south')
    cGln  = InletConcentration(cGln, cGlnin, nin_start3, nin_end3, 'south')
    cO2   = InletConcentration(cO2, cO2in, nin_start4, nin_end4, 'north')
    cGlc  = InletConcentration(cGlc, cGlcin, nin_start4, nin_end4, 'north')
    cGln  = InletConcentration(cGln, cGlnin, nin_start4, nin_end4, 'north')
    cO2   = InletConcentration(cO2, cO2in, nin_start5, nin_end5, 'south')
    cGlc  = InletConcentration(cGlc, cGlcin, nin_start5, nin_end5, 'south')
    cGln  = InletConcentration(cGln, cGlnin, nin_start5, nin_end5, 'south')
    cO2   = InletConcentration(cO2, cO2in, nin_start6, nin_end6, 'north')
    cGlc  = InletConcentration(cGlc, cGlcin, nin_start6, nin_end6, 'north')
    cGln  = InletConcentration(cGln, cGlnin, nin_start6, nin_end6, 'north')
    cO2   = InletConcentration(cO2, cO2in, nin_start7, nin_end7, 'south')
    cGlc  = InletConcentration(cGlc, cGlcin, nin_start7, nin_end7, 'south')
    cGln  = InletConcentration(cGln, cGlnin, nin_start7, nin_end7, 'south')
    cO2   = InletConcentration(cO2, cO2in, nin_start8, nin_end8, 'north')
    cGlc  = InletConcentration(cGlc, cGlcin, nin_start8, nin_end8, 'north')
    cGln  = InletConcentration(cGln, cGlnin, nin_start8, nin_end8, 'north')
    cO2   = InletConcentration(cO2, cO2in, nin_start9, nin_end9, 'south')
    cGlc  = InletConcentration(cGlc, cGlcin, nin_start9, nin_end9, 'south')
    cGln  = InletConcentration(cGln, cGlnin, nin_start9, nin_end9, 'south')
    cO2   = InletConcentration(cO2, cO2in, nin_start10, nin_end10, 'north')
    cGlc  = InletConcentration(cGlc, cGlcin, nin_start10, nin_end10, 'north')
    cGln  = InletConcentration(cGln, cGlnin, nin_start10, nin_end10, 'north')
    cO2   = InletConcentration(cO2, cO2in, nin_start11, nin_end11, 'south')
    cGlc  = InletConcentration(cGlc, cGlcin, nin_start11, nin_end11, 'south')
    cGln  = InletConcentration(cGln, cGlnin, nin_start11, nin_end11, 'south')
    cO2   = InletConcentration(cO2, cO2in, nin_start12, nin_end12, 'north')
    cGlc  = InletConcentration(cGlc, cGlcin, nin_start12, nin_end12, 'north')
    cGln  = InletConcentration(cGln, cGlnin, nin_start12, nin_end12, 'north')
    cO2   = InletConcentration(cO2, cO2in, nin_start13, nin_end13, 'south')
    cGlc  = InletConcentration(cGlc, cGlcin, nin_start13, nin_end13, 'south')
    cGln  = InletConcentration(cGln, cGlnin, nin_start13, nin_end13, 'south')
    cO2   = InletConcentration(cO2, cO2in, nin_start14, nin_end14, 'north')
    cGlc  = InletConcentration(cGlc, cGlcin, nin_start14, nin_end14, 'north')
    cGln  = InletConcentration(cGln, cGlnin, nin_start14, nin_end14, 'north')
    cO2   = InletConcentration(cO2, cO2in, nin_start15, nin_end15, 'south')
    cGlc  = InletConcentration(cGlc, cGlcin, nin_start15, nin_end15, 'south')
    cGln  = InletConcentration(cGln, cGlnin, nin_start15, nin_end15, 'south')
    cO2   = InletConcentration(cO2, cO2in, nin_start16, nin_end16, 'north')
    cGlc  = InletConcentration(cGlc, cGlcin, nin_start16, nin_end16, 'north')
    cGln  = InletConcentration(cGln, cGlnin, nin_start16, nin_end16, 'north')

    # Advection-Diffusion equation
    cO2star = AdvDiffSpecies(cO2, u, v, dt, hx, hy, Gamma, nx, ny, flagp, method)
    cGlcstar = AdvDiffSpecies(cGlc, u, v, dt, hx, hy, Gamma, nx, ny, flagp, method)
    cGlnstar = AdvDiffSpecies(cGln, u, v, dt, hx, hy, Gamma, nx, ny, flagp, method)
    cLacstar = AdvDiffSpecies(cLac, u, v, dt, hx, hy, Gamma, nx, ny, flagp, method)
    cAmmstar = AdvDiffSpecies(cAmm, u, v, dt, hx, hy, Gamma, nx, ny, flagp, method)
    cB1star  = AdvDiffSpecies(cB1, u, v, dt, hx, hy, Gamma, nx, ny, flagp, method)
    Xvstar = Xv

    # Reaction step (Linearization + Segregation)
    cO2, Xv, cGlc, cGln, cLac, cAmm, cB1 = ReactionStep(cO2star, Xvstar, cGlcstar, cGlnstar, cLacstar, cAmmstar, cB1star, dt, parameters, nx, ny)

    # Collecting cells mean values for graphical purposes
    cBmean2 = np.append(cBmean2, CellsMeanConcentration(Xv, th, X2_1, X2_2, cells_position)) # (pay attention if cells are above or under the plate!!!)
    cBmean3 = np.append(cBmean3, CellsMeanConcentration(Xv, th, X3_1, X3_2, cells_position))
    cBmean4 = np.append(cBmean4, CellsMeanConcentration(Xv, th, X4_1, X4_2, cells_position))
    cBmean5 = np.append(cBmean5, CellsMeanConcentration(Xv, th, X5_1, X5_2, cells_position))
    cBmean6 = np.append(cBmean6, CellsMeanConcentration(Xv, th, X6_1, X6_2, cells_position))
    cBmean7 = np.append(cBmean7, CellsMeanConcentration(Xv, th, X7_1, X7_2, cells_position))
    cBmean8 = np.append(cBmean8, CellsMeanConcentration(Xv, th, X8_1, X8_2, cells_position))
    cBmean9 = np.append(cBmean9, CellsMeanConcentration(Xv, th, X9_1, X9_2, cells_position))
    cBmean10 = np.append(cBmean10, CellsMeanConcentration(Xv, th, X10_1, X10_2, cells_position))
    cBmean11 = np.append(cBmean11, CellsMeanConcentration(Xv, th, X11_1, X11_2, cells_position))
    cBmean12 = np.append(cBmean12, CellsMeanConcentration(Xv, th, X12_1, X12_2, cells_position))
    cBmean13 = np.append(cBmean13, CellsMeanConcentration(Xv, th, X13_1, X13_2, cells_position))
    cBmean14 = np.append(cBmean14, CellsMeanConcentration(Xv, th, X14_1, X14_2, cells_position))
    cBmean15 = np.append(cBmean15, CellsMeanConcentration(Xv, th, X15_1, X15_2, cells_position))
    cBmean16 = np.append(cBmean16, CellsMeanConcentration(Xv, th, X16_1, X16_2, cells_position))
    cBmean17 = np.append(cBmean17, CellsMeanConcentration(Xv, th, X17_1, X17_2, cells_position))

    # Collecting the outlet concentration of B1
    cB1out = np.append(cB1out, B1OutletConcentration(cB1, nout_start, nout_end))

    # Collecting the maximum value of species concentration and velocities
    cGlc_max = np.append(cGlc_max, np.max(cGlc));   cGln_max = np.append(cGln_max, np.max(cGln))
    cAmm_max = np.append(cAmm_max, np.max(cAmm));   cLac_max = np.append(cLac_max, np.max(cLac))
    cO2_max = np.append(cO2_max, np.max(cO2));      cB1_max = np.append(cB1_max, np.max(cB1))
    u_max = np.append(u_max, np.max(u));            v_max = np.append(v_max, np.max(v))
    
    # Collecting time values for graphical purposes
    t_vector = np.append(t_vector, t)

    # If t = Time exits from the loop
    if t >= Time:
        break

    # Advance in time
    t = t + dt

    # On-the-fly plotting
    if OnTheFly == 'Yes' and it % 250 == 1:
        plt.ion()
        xx,yy = np.meshgrid(x,y)
        ccGlc = node_interp(cGlc, 'p', nx, ny, flagp)
        PlotFunctions(xx, yy, ccGlc, x, y, XX, Lx, Ly, f'Glucose concentration at time {t/60} min [mol/m3]', 'x [m]', 'y [m]')
        plt.pause(0.5)
        
# Export outlet concentration of B1 in an Excel file
filename = f'Rack_horizontal_{nx}x{ny}_{Time/3600}h.dat'
dictionary = { 't': t_vector, 'B1_out': cB1out , 'Glc_max': cGlc_max, 'Gln_max': cGln_max , 'Amm_max': cAmm_max, 'Lac_max': cLac_max , 'O2_max': cO2_max, 'B1_max': cB1_max}
B1df = pd.DataFrame(dictionary)
B1df.to_csv(filename, sep = ';')

# Control the simulation time
end = time.time()
print(f'time ealpsed: {end-start} sec')

# Shear stress over the entire domain
dR = Obstacles(x, y, Lx, Ly)[1]
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
ccB1  = node_interp(cB1, 'p', nx, ny, flagp)

# Interpolation of cells concentration needed for graphical purposes
ccXv = Interpolation_of_cells(ccXv, Xv, X2_1, X2_2, th, cells_position)  # Obstacle 2
ccXv = Interpolation_of_cells(ccXv, Xv, X3_1, X3_2, th, cells_position)  # Obstacle 3
ccXv = Interpolation_of_cells(ccXv, Xv, X4_1, X4_2, th, cells_position)  # Obstacle 4
ccXv = Interpolation_of_cells(ccXv, Xv, X5_1, X5_2, th, cells_position)  # Obstacle 5
ccXv = Interpolation_of_cells(ccXv, Xv, X6_1, X6_2, th, cells_position)  # Obstacle 6
ccXv = Interpolation_of_cells(ccXv, Xv, X7_1, X7_2, th, cells_position)  # Obstacle 7
ccXv = Interpolation_of_cells(ccXv, Xv, X8_1, X8_2, th, cells_position)  # Obstacle 8
ccXv = Interpolation_of_cells(ccXv, Xv, X9_1, X9_2, th, cells_position)  # Obstacle 9
ccXv = Interpolation_of_cells(ccXv, Xv, X10_1, X10_2, th, cells_position)  # Obstacle 10
ccXv = Interpolation_of_cells(ccXv, Xv, X11_1, X11_2, th, cells_position)  # Obstacle 11
ccXv = Interpolation_of_cells(ccXv, Xv, X12_1, X12_2, th, cells_position)  # Obstacle 12
ccXv = Interpolation_of_cells(ccXv, Xv, X13_1, X13_2, th, cells_position)  # Obstacle 13
ccXv = Interpolation_of_cells(ccXv, Xv, X14_1, X14_2, th, cells_position)  # Obstacle 14
ccXv = Interpolation_of_cells(ccXv, Xv, X15_1, X15_2, th, cells_position)  # Obstacle 15
ccXv = Interpolation_of_cells(ccXv, Xv, X16_1, X16_2, th, cells_position)  # Obstacle 16
ccXv = Interpolation_of_cells(ccXv, Xv, X17_1, X17_2, th, cells_position)  # Obstacle 17

# Addition of obstacles for graphical purposes
uu, vv, pp, tau, ccO2, ccXv, ccGlc, ccGln, ccLac, ccAmm, ccB1 = Graphical_obstacles(uu, vv, pp, tau, ccO2, ccXv, ccGlc, ccGln, ccLac, ccAmm, ccB1, XX)

# Creating a grid
xx,yy = np.meshgrid(x,y)

# Plotting the results
plt.ioff()
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

# Surface map: cB1
PlotFunctions(xx, yy, ccB1, x, y, XX, Lx, Ly, 'Antibody fusion protein concentration [mol/m3]', 'x [m]', 'y [m]')

# Surface map: Xv
PlotFunctions(xx, yy, ccXv, x, y, XX, Lx, Ly, 'Viable cells number [cells]', 'x [m]', 'y [m]')

# Surface map: Tau
PlotFunctions(xx, yy, tau, x, y, XX, Lx, Ly, 'Shear stress [N/m2]', 'x [m]', 'y [m]')

# Streamlines
Streamlines(x, y, xx, yy, uu, vv, XX, Lx, Ly, 'Streamlines', 'x [m]', 'y [m]')

# Mean values of cB
MeanCellsPlot(t_vector, [cBmean2, cBmean3, cBmean4, cBmean5, cBmean6, cBmean7, cBmean8, cBmean9, cBmean10, cBmean11, cBmean12, cBmean13, cBmean14, cBmean15, cBmean16, cBmean17], Time, Xv0)

# Outlet concentration of B1 in time
PlotinTime(t_vector, cB1out, Time, 'Total B1 protein concentration at the outlet section of the reactor', 'Antibody fusion protein concentration [mol/m3]')

# Maximum concentration of Glucose in time
PlotinTime(t_vector, cGlc_max, Time, 'Maximum concentration of glucose in the reactor', 'Glucose concentration [mol/m3]')

# Maximum concentration of Glutamine in time
PlotinTime(t_vector, cGln_max, Time, 'Maximum concentration of glutamine in the reactor', 'Glutamine concentration [mol/m3]')

# Maximum concentration of Ammonia in time
PlotinTime(t_vector, cAmm_max, Time, 'Maximum concentration of ammonia in the reactor', 'Ammonia concentration [mol/m3]')

# Maximum concentration of Lactose in time
PlotinTime(t_vector, cLac_max, Time, 'Maximum concentration of lactose in the reactor', 'Lactose concentration [mol/m3]')

# Maximum concentration of Oxygen in time
PlotinTime(t_vector, cO2_max, Time, 'Maximum concentration of oxygen in the reactor', 'Oxygen concentration [mol/m3]')

# Maximum concentration of B1 protein in time
PlotinTime(t_vector, cB1_max, Time, 'Maximum concentration of B1 protein in the reactor', 'Antibody fusion protein concentration [mol/m3]')

plt.show()