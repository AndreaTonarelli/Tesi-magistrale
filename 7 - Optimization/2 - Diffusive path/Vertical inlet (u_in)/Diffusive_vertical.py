import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from   numba import njit
from   numba.typed import List
from   Diffusive_vertical_functions import *
from   Diffusive_vertical_obstacles_def import *
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
u_in = 1e-3        # m/s   (inlet velocity)
Time = 7200.      # s     (simulation time)
method = 'Upwind'  # Discretization method (CDS or Upwind)

# Boundary conditions (no-slip)
un = 0             # m/s
us = 0             # m/s
ve = 0             # m/s
vw = 0             # m/s

# Inlet concentrations
cO2in = 100.       # mol/m3
cGlcin = 100.      # mol/m3
cGlnin = 100.      # mol/m3

# Immobilized cells parameters
Xv0 = 1e4             # cells/m3 (initial number of viable cells)
cB1_0 = 1e0           # mol/m3 (initial concentration of B1)
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
XX = Obstacles(x, y, Lx, Ly)[0]
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
X11 = XX[10]; xs11 = X11[0]; xe11 = X11[1]; ys11 = X11[2]; ye11 = X11[3]
X12 = XX[11]; xs12 = X12[0]; xe12 = X12[1]; ys12 = X12[2]; ye12 = X12[3]
X13 = XX[12]; xs13 = X13[0]; xe13 = X13[1]; ys13 = X13[2]; ye13 = X13[3]
X14 = XX[13]; xs14 = X14[0]; xe14 = X14[1]; ys14 = X14[2]; ye14 = X14[3]
X15 = XX[14]; xs15 = X15[0]; xe15 = X15[1]; ys15 = X15[2]; ye15 = X15[3]

# Inlet section (north side)
nin_start1 = ye1 + 1    # first cell index (pay attention to Python INDEX!!!)
nin_end1 = ny           # last cell index
nin_start2 = 1          # first cell index (pay attention to Python INDEX!!!)
nin_end2 = ys1 - 1      # last cell index

# Outlet section (east side)
nout_start = ye11 + 1   # first cell index (0 is the first one!!!)
nout_end = ys12 - 1     # last cell index

# Inlet/Outlet section areas and outlet velocity
Ain1 = hy*(nin_end1-nin_start1+1)       # inlet section area [m]
Ain2 = hy*(nin_end2-nin_start2+1)
Ain = Ain1 + Ain2                       # total inlet section area [m]
Aout = hy*(nout_end-nout_start+1)       # outlet section area [m]
u_out = (Ain1*u_in + Ain2*u_in)/Aout

# Time step
sigma = 0.5                                        # safety factor for time step (stability)
dt_diff_ns = np.minimum(hx,hy)**2/4/nu             # time step (diffusion stability) [s]
dt_conv_ns = 4*nu/u_in**2                          # time step (convection stability) [s]
dt_ns      = np.minimum(dt_diff_ns, dt_conv_ns)    # time step (stability due to FD) [s]
dt_diff_sp = np.minimum(hx,hy)**2/4/Gamma          # time step (species diffusion stability) [s]
dt_conv_sp = 4*Gamma/u_in**2                       # time step (species convection stability) [s]
dt_sp      = np.minimum(dt_conv_sp, dt_diff_sp)    # time step (stability due to species) [s]
dt         = sigma*min(dt_ns, dt_sp)               # time step (stability) [s]
nsteps     = int(Time/dt)*1000                     # number of steps
Re         = u_in*Ly/nu                            # Reynolds' number

print('Time step:', round(dt,1), 's' )
print(' - Diffusion (NS):', round(dt_diff_ns), 's')
print(' - Convection (NS):', round(dt_conv_ns), 's')
print(' - Diffusion (Species):', round(dt_diff_sp), 's')
print(' - Convection (Species):', round(dt_conv_sp), 's')
print('Reynolds number:', round(Re), '\n')

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
cBmean11 = cBmean12 = cBmean13 = cBmean14 = cBmean15 = cB1out = t_vector = np.empty(0)
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
u[0, nin_start1-1:nin_end1+1] = u_in     # fixed inlet velocity (west)
u[0, nin_start2-1:nin_end2+1] = u_in
u[-1, nout_start-1:nout_end+1] = u_out   # fixed outlet velocity (east)
'''u[:, :] = u_in/10                   # Internal points: fixed velocity [m/s]
u = u_initialize(u, XX)             # set u = 0 over obstacles'''
ut = u

# Immobilized cells initialized over obstacles
cells_position = 'left'
Xv = ImmobilizedCells(Xv, Xv0, X2, th, cells_position)  # Obstacle 2
Xv = ImmobilizedCells(Xv, Xv0, X3, th, cells_position)  # Obstacle 3
Xv = ImmobilizedCells(Xv, Xv0, X4, th, cells_position)  # Obstacle 4
Xv = ImmobilizedCells(Xv, Xv0, X5, th, cells_position)  # Obstacle 5
Xv = ImmobilizedCells(Xv, Xv0, X6, th, cells_position)  # Obstacle 6
Xv = ImmobilizedCells(Xv, Xv0, X7, th, cells_position)  # Obstacle 7
Xv = ImmobilizedCells(Xv, Xv0, X8, th, cells_position)  # Obstacle 8
Xv = ImmobilizedCells(Xv, Xv0, X9, th, cells_position)  # Obstacle 9
Xv = ImmobilizedCells(Xv, Xv0, X10, th, cells_position)  # Obstacle 10
Xv = ImmobilizedCells(Xv, Xv0, X11, th, cells_position)  # Obstacle 11
Xv = ImmobilizedCells(Xv, Xv0, X12, th, cells_position)  # Obstacle 12
Xv = ImmobilizedCells(Xv, Xv0, X13, th, cells_position)  # Obstacle 13
Xv = ImmobilizedCells(Xv, Xv0, X14, th, cells_position)  # Obstacle 14
Xv = ImmobilizedCells(Xv, Xv0, X15, th, cells_position)  # Obstacle 15
cB1 = ImmobilizedCells(cB1, cB1_0, X2, th, cells_position)  # Obstacle 2
cB1 = ImmobilizedCells(cB1, cB1_0, X3, th, cells_position)  # Obstacle 3
cB1 = ImmobilizedCells(cB1, cB1_0, X4, th, cells_position)  # Obstacle 4
cB1 = ImmobilizedCells(cB1, cB1_0, X5, th, cells_position)  # Obstacle 5
cB1 = ImmobilizedCells(cB1, cB1_0, X6, th, cells_position)  # Obstacle 6
cB1 = ImmobilizedCells(cB1, cB1_0, X7, th, cells_position)  # Obstacle 7
cB1 = ImmobilizedCells(cB1, cB1_0, X8, th, cells_position)  # Obstacle 8
cB1 = ImmobilizedCells(cB1, cB1_0, X9, th, cells_position)  # Obstacle 9
cB1 = ImmobilizedCells(cB1, cB1_0, X10, th, cells_position)  # Obstacle 10
cB1 = ImmobilizedCells(cB1, cB1_0, X11, th, cells_position)  # Obstacle 11
cB1 = ImmobilizedCells(cB1, cB1_0, X12, th, cells_position)  # Obstacle 12
cB1 = ImmobilizedCells(cB1, cB1_0, X13, th, cells_position)  # Obstacle 13
cB1 = ImmobilizedCells(cB1, cB1_0, X14, th, cells_position)  # Obstacle 14
cB1 = ImmobilizedCells(cB1, cB1_0, X15, th, cells_position)  # Obstacle 15

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
    u[0, nin_start2-1:nin_end2+1] = u_in

    # Over-writing outlet conditions
    u[-1, nout_start-1:nout_end+1] = u[-2, nout_start-1:nout_end+1]   # zero-gradient outlet velocity
    v[-1, nout_start-1:nout_end+1] = v[-2, nout_start-1:nout_end+1]   # zero-gradient outlet velocity

    # Uptade the dt for stability conditions
    dt_conv_ns = 1/(np.max(u)/hx + np.max(v)/hy)       # time step (convection stability) [s]
    dt_ns      = np.minimum(dt_diff_ns, dt_conv_ns)    # time step (stability due to FD) [s]
    dt         = sigma*min(dt_ns, dt_sp)               # time step (stability) [s]

    # Advection-diffusion equation (predictor)
    ut, vt = AdvectionDiffusion2D(ut, vt, u, v, nx, ny, hx, hy, dt, nu, g, flagu, flagv, method)

    # Update boundary conditions for temporary velocities
    ut[0, nin_start1-1:nin_end1+1] = u_in     # fixed inlet velocity (west)
    ut[0, nin_start2-1:nin_end2+1] = u_in
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
    Qin1  = abs(np.mean(u[0, nin_start1-1:nin_end1+1])*Ain1)    # inlet flow rate [m2/s]
    Qin2  = abs(np.mean(u[0, nin_start2-1:nin_end2+1])*Ain2)
    Qin = Qin1 + Qin2
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

    if iter > 100:
        print('At step ', it, ' Pressure eq. has overcome 100 iterations, there is a possible problem!')
            
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
    cO2   = InletConcentration(cO2, cO2in, nin_start1, nin_end1, 'west')
    cGlc  = InletConcentration(cGlc, cGlcin, nin_start1, nin_end1, 'west')
    cGln  = InletConcentration(cGln, cGlnin, nin_start1, nin_end1, 'west')
    cO2   = InletConcentration(cO2, cO2in, nin_start2, nin_end2, 'west')
    cGlc  = InletConcentration(cGlc, cGlcin, nin_start2, nin_end2, 'west')
    cGln  = InletConcentration(cGln, cGlnin, nin_start2, nin_end2, 'west')

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
    cBmean2 = np.append(cBmean2, CellsMeanConcentration(Xv, th, X2, cells_position)) # (pay attention if cells are above or under the plate!!!)
    cBmean3 = np.append(cBmean3, CellsMeanConcentration(Xv, th, X3, cells_position))
    cBmean4 = np.append(cBmean4, CellsMeanConcentration(Xv, th, X4, cells_position))
    cBmean5 = np.append(cBmean5, CellsMeanConcentration(Xv, th, X5, cells_position))
    cBmean6 = np.append(cBmean6, CellsMeanConcentration(Xv, th, X6, cells_position))
    cBmean7 = np.append(cBmean7, CellsMeanConcentration(Xv, th, X7, cells_position))
    cBmean8 = np.append(cBmean8, CellsMeanConcentration(Xv, th, X8, cells_position))
    cBmean9 = np.append(cBmean9, CellsMeanConcentration(Xv, th, X9, cells_position))
    cBmean10 = np.append(cBmean10, CellsMeanConcentration(Xv, th, X10, cells_position))
    cBmean11 = np.append(cBmean11, CellsMeanConcentration(Xv, th, X11, cells_position))
    cBmean12 = np.append(cBmean12, CellsMeanConcentration(Xv, th, X12, cells_position))
    cBmean13 = np.append(cBmean13, CellsMeanConcentration(Xv, th, X13, cells_position))
    cBmean14 = np.append(cBmean14, CellsMeanConcentration(Xv, th, X14, cells_position))
    cBmean15 = np.append(cBmean15, CellsMeanConcentration(Xv, th, X15, cells_position))

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
    
    # Control the simulation time
    
# Export outlet concentration of B1 in an Excel file
filename = f'Diffusive_vertical_{nx}x{ny}_{Time}sec.dat'
dictionary = { 't': t_vector, 'B1_out': cB1out , 'Glc_max': cGlc_max, 'Gln_max': cGln_max , 'Amm_max': cAmm_max, 'Lac_max': cLac_max , 'O2_max': cO2_max, 'B1_max': cB1_max}
B1df = pd.DataFrame(dictionary)
B1df.to_csv(filename, sep = ';')
    
end = time.time()
print(f'Elapsed time: {end-start} sec')

# Shear stress over the entire domain
dR = Obstacles(x, y, Lx, Ly)[-1]
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
ccXv = Interpolation_of_cells(ccXv, Xv, X2, th, cells_position)  # Obstacle 2
ccXv = Interpolation_of_cells(ccXv, Xv, X3, th, cells_position)  # Obstacle 3
ccXv = Interpolation_of_cells(ccXv, Xv, X4, th, cells_position)  # Obstacle 4
ccXv = Interpolation_of_cells(ccXv, Xv, X5, th, cells_position)  # Obstacle 5
ccXv = Interpolation_of_cells(ccXv, Xv, X6, th, cells_position)  # Obstacle 6
ccXv = Interpolation_of_cells(ccXv, Xv, X7, th, cells_position)  # Obstacle 7
ccXv = Interpolation_of_cells(ccXv, Xv, X8, th, cells_position)  # Obstacle 8
ccXv = Interpolation_of_cells(ccXv, Xv, X9, th, cells_position)  # Obstacle 9
ccXv = Interpolation_of_cells(ccXv, Xv, X10, th, cells_position)  # Obstacle 10
ccXv = Interpolation_of_cells(ccXv, Xv, X11, th, cells_position)  # Obstacle 11
ccXv = Interpolation_of_cells(ccXv, Xv, X12, th, cells_position)  # Obstacle 12
ccXv = Interpolation_of_cells(ccXv, Xv, X13, th, cells_position)  # Obstacle 13
ccXv = Interpolation_of_cells(ccXv, Xv, X14, th, cells_position)  # Obstacle 14
ccXv = Interpolation_of_cells(ccXv, Xv, X15, th, cells_position)  # Obstacle 15

# Addition of obstacles for graphical purposes
uu, vv, pp, tau, ccO2, ccXv, ccGlc, ccGln, ccLac, ccAmm, ccB1 = Graphical_obstacles(uu, vv, pp, tau, ccO2, ccXv, ccGlc, ccGln, ccLac, ccAmm, ccB1, XX)

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

# Surface map: cB1
PlotFunctions(xx, yy, ccB1, x, y, XX, Lx, Ly, 'Antibody fusion protein concentration [mol/m3]', 'x [m]', 'y [m]')

# Surface map: Xv
PlotFunctions(xx, yy, ccXv, x, y, XX, Lx, Ly, 'Viable cells number [cells/m3]', 'x [m]', 'y [m]')

# Surface map: Tau
PlotFunctions(xx, yy, tau, x, y, XX, Lx, Ly, 'Shear stress [N/m2]', 'x [m]', 'y [m]')

# Streamlines
Streamlines(x, y, xx, yy, uu, vv, XX, Lx, Ly, 'Streamlines', 'x [m]', 'y [m]')

# Mean values of cB
MeanCellsPlot(t_vector, [cBmean2, cBmean3, cBmean4, cBmean5, cBmean6, cBmean7, cBmean8, cBmean9, cBmean10, cBmean11, cBmean12, cBmean13, cBmean14, cBmean15], Time, Xv0)

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