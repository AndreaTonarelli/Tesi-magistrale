import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from   matplotlib import cm
from   matplotlib.patches import Rectangle
from   numba import njit


# Gamma coefficient for Poisson equation
def gammaCoeff(gamma, hx,hy, nout_start,nout_end, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10):

    # Defining the obstacles coordinates
    xs1 = X1[0]; xe1 = X1[1]; ys1 = X1[2]; ye1 = X1[3]
    xs2 = X2[0]; xe2 = X2[1]; ys2 = X2[2]; ye2 = X2[3]
    xs3 = X3[0]; xe3 = X3[1]; ys3 = X3[2]; ye3 = X3[3]
    xs4 = X4[0]; xe4 = X4[1]; ys4 = X4[2]; ye4 = X4[3]
    xs5 = X5[0]; xe5 = X5[1]; ys5 = X5[2]; ye5 = X5[3]
    xs6 = X6[0]; xe6 = X6[1]; ys6 = X6[2]; ye6 = X6[3]
    xs7 = X7[0]; xe7 = X7[1]; ys7 = X7[2]; ye7 = X7[3]
    xs8 = X8[0]; xe8 = X8[1]; ys8 = X8[2]; ye8 = X8[3]
    xs9 = X9[0]; xe9 = X9[1]; ys9 = X9[2]; ye9 = X9[3]
    xs10 = X10[0]; xe10 = X10[1]; ys10 = X10[2]; ye10 = X10[3]

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

    # Defining gamma on the entire domain edges and corners
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
    # Obstacle 10
    gamma[xs10-1, ys10:ye10+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xe10+1, ys10:ye10+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xs10:xe10+1, ys10-1] = hx*hy / (  hx**2 + 2*hy**2)
    gamma[xs10:xe10+1, ye10+1] = hx*hy / (  hx**2 + 2*hy**2)
    # near Obstacle 1
    gamma[xs11-1, ys11:ye11+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xe11+1, ys11:ye11+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xs11:xe11+1, ys11-1] = hx*hy / (  hx**2 + 2*hy**2)
    gamma[xs11:xe11+1, ye11+1] = hx*hy / (  hx**2 + 2*hy**2)
    gamma[xs12-1, ys12:ye12+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xe12+1, ys12:ye12+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xs12:xe12+1, ys12-1] = hx*hy / (  hx**2 + 2*hy**2)
    gamma[xs12:xe12+1, ye12+1] = hx*hy / (  hx**2 + 2*hy**2)
    # near Obstacle 2
    gamma[xs21-1, ys21:ye21+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xe21+1, ys21:ye21+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xs21:xe21+1, ys21-1] = hx*hy / (  hx**2 + 2*hy**2)
    gamma[xs21:xe21+1, ye21+1] = hx*hy / (  hx**2 + 2*hy**2)
    gamma[xs22-1, ys22:ye22+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xe22+1, ys22:ye22+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xs22:xe22+1, ys22-1] = hx*hy / (  hx**2 + 2*hy**2)
    gamma[xs22:xe22+1, ye22+1] = hx*hy / (  hx**2 + 2*hy**2)
    # near Obstacle 3
    gamma[xs31-1, ys31:ye31+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xe31+1, ys31:ye31+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xs31:xe31+1, ys31-1] = hx*hy / (  hx**2 + 2*hy**2)
    gamma[xs31:xe31+1, ye31+1] = hx*hy / (  hx**2 + 2*hy**2)
    gamma[xs32-1, ys32:ye32+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xe32+1, ys32:ye32+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xs32:xe32+1, ys32-1] = hx*hy / (  hx**2 + 2*hy**2)
    gamma[xs32:xe32+1, ye32+1] = hx*hy / (  hx**2 + 2*hy**2)
    # near Obstacle 4
    gamma[xs41-1, ys41:ye41+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xe41+1, ys41:ye41+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xs41:xe41+1, ys41-1] = hx*hy / (  hx**2 + 2*hy**2)
    gamma[xs41:xe41+1, ye41+1] = hx*hy / (  hx**2 + 2*hy**2)
    gamma[xs42-1, ys42:ye42+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xe42+1, ys42:ye42+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xs42:xe42+1, ys42-1] = hx*hy / (  hx**2 + 2*hy**2)
    gamma[xs42:xe42+1, ye42+1] = hx*hy / (  hx**2 + 2*hy**2)
    # near Obstacle 5
    gamma[xs51-1, ys51:ye51+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xe51+1, ys51:ye51+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xs51:xe51+1, ys51-1] = hx*hy / (  hx**2 + 2*hy**2)
    gamma[xs51:xe51+1, ye51+1] = hx*hy / (  hx**2 + 2*hy**2)
    gamma[xs52-1, ys52:ye52+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xe52+1, ys52:ye52+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xs52:xe52+1, ys52-1] = hx*hy / (  hx**2 + 2*hy**2)
    gamma[xs52:xe52+1, ye52+1] = hx*hy / (  hx**2 + 2*hy**2)
    # near Obstacle 6
    gamma[xs61-1, ys61:ye61+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xe61+1, ys61:ye61+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xs61:xe61+1, ys61-1] = hx*hy / (  hx**2 + 2*hy**2)
    gamma[xs61:xe61+1, ye61+1] = hx*hy / (  hx**2 + 2*hy**2)
    gamma[xs62-1, ys62:ye62+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xe62+1, ys62:ye62+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xs62:xe62+1, ys62-1] = hx*hy / (  hx**2 + 2*hy**2)
    gamma[xs62:xe62+1, ye62+1] = hx*hy / (  hx**2 + 2*hy**2)
    # near Obstacle 7
    gamma[xs71-1, ys71:ye71+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xe71+1, ys71:ye71+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xs71:xe71+1, ys71-1] = hx*hy / (  hx**2 + 2*hy**2)
    gamma[xs71:xe71+1, ye71+1] = hx*hy / (  hx**2 + 2*hy**2)
    gamma[xs72-1, ys72:ye72+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xe72+1, ys72:ye72+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xs72:xe72+1, ys72-1] = hx*hy / (  hx**2 + 2*hy**2)
    gamma[xs72:xe72+1, ye72+1] = hx*hy / (  hx**2 + 2*hy**2)
    # near Obstacle 8
    gamma[xs81-1, ys81:ye81+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xe81+1, ys81:ye81+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xs81:xe81+1, ys81-1] = hx*hy / (  hx**2 + 2*hy**2)
    gamma[xs81:xe81+1, ye81+1] = hx*hy / (  hx**2 + 2*hy**2)
    gamma[xs82-1, ys82:ye82+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xe82+1, ys82:ye82+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xs82:xe82+1, ys82-1] = hx*hy / (  hx**2 + 2*hy**2)
    gamma[xs82:xe82+1, ye82+1] = hx*hy / (  hx**2 + 2*hy**2)
    # near Obstacle 9
    gamma[xs91-1, ys91:ye91+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xe91+1, ys91:ye91+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xs91:xe91+1, ys91-1] = hx*hy / (  hx**2 + 2*hy**2)
    gamma[xs91:xe91+1, ye91+1] = hx*hy / (  hx**2 + 2*hy**2)
    gamma[xs92-1, ys92:ye92+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xe92+1, ys92:ye92+1] = hx*hy / (2*hx**2 +   hy**2)
    gamma[xs92:xe92+1, ys92-1] = hx*hy / (  hx**2 + 2*hy**2)
    gamma[xs92:xe92+1, ye92+1] = hx*hy / (  hx**2 + 2*hy**2)

    # Correction of gamma close to the obstacles edges
    gamma[xs1, ye2+1] = hx*hy / (hx**2 + hy**2)          # Obstacle 1
    gamma[xe1+1, ye11+1] = hx*hy / (hx**2 + hy**2)
    gamma[xe11+1, ye12+1] = hx*hy / (hx**2 + hy**2)
    gamma[xe12+1, ys1] = hx*hy / (hx**2 + hy**2)

    gamma[xs2-1, ye2] = hx*hy / (hx**2 + hy**2)          # Obstacle 2
    gamma[xe2+1, ys21-1] = hx*hy / (hx**2 + hy**2)
    gamma[xe21+1, ys22-1] = hx*hy / (hx**2 + hy**2)
    gamma[xe22+1, ys2] = hx*hy / (hx**2 + hy**2)

    gamma[xs3-1, ys3] = hx*hy / (hx**2 + hy**2)          # Obstacle 3
    gamma[xe3+1, ye31+1] = hx*hy / (hx**2 + hy**2)
    gamma[xe31+1, ye32+1] = hx*hy / (hx**2 + hy**2)
    gamma[xe32+1, ys3] = hx*hy / (hx**2 + hy**2)

    gamma[xs4-1, ye4] = hx*hy / (hx**2 + hy**2)          # Obstacle 4
    gamma[xe4+1, ys41-1] = hx*hy / (hx**2 + hy**2)
    gamma[xe41+1, ys42-1] = hx*hy / (hx**2 + hy**2)
    gamma[xe42+1, ye4] = hx*hy / (hx**2 + hy**2)

    gamma[xs5-1, ys5] = hx*hy / (hx**2 + hy**2)          # Obstacle 5
    gamma[xe5+1, ye51+1] = hx*hy / (hx**2 + hy**2)
    gamma[xe51+1, ye52+1] = hx*hy / (hx**2 + hy**2)
    gamma[xe52+1, ys5] = hx*hy / (hx**2 + hy**2)

    gamma[xs6-1, ye6] = hx*hy / (hx**2 + hy**2)          # Obstacle 6
    gamma[xe6+1, ys61-1] = hx*hy / (hx**2 + hy**2)
    gamma[xe61+1, ys62-1] = hx*hy / (hx**2 + hy**2)
    gamma[xe62+1, ye6] = hx*hy / (hx**2 + hy**2)

    gamma[xs7-1, ys7] = hx*hy / (hx**2 + hy**2)          # Obstacle 7
    gamma[xe7+1, ye71+1] = hx*hy / (hx**2 + hy**2)
    gamma[xe71+1, ye72+1] = hx*hy / (hx**2 + hy**2)
    gamma[xe72+1, ys7] = hx*hy / (hx**2 + hy**2)

    gamma[xs8-1, ye8] = hx*hy / (hx**2 + hy**2)          # Obstacle 8
    gamma[xe8+1, ys81-1] = hx*hy / (hx**2 + hy**2)
    gamma[xe81+1, ys82-1] = hx*hy / (hx**2 + hy**2)
    gamma[xe82+1, ye8] = hx*hy / (hx**2 + hy**2)

    gamma[xs9-1, ys9] = hx*hy / (hx**2 + hy**2)          # Obstacle 9
    gamma[xe9+1, ye91+1] = hx*hy / (hx**2 + hy**2)
    gamma[xe91+1, ye92+1] = hx*hy / (hx**2 + hy**2)
    gamma[xe92+1, ys9] = hx*hy / (hx**2 + hy**2)

    gamma[xs10-1, ye10] = hx*hy / (hx**2 + hy**2)          # Obstacle 10

    return gamma



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
def AdvectionDiffusion2D(ut, vt, u, v, nx, ny, hx, hy, dt, nu, g, flagu, flagv, method):

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

                ut[i, j] = u[i, j] + dt*(-A + D) - dt*g

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



# Boundary conditions for velocities
def VelocityBCs(u, v, uwall, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10):

    # Defining the obstacles coordinates
    xs1 = X1[0]; xe1 = X1[1]; ys1 = X1[2]; ye1 = X1[3]
    xs2 = X2[0]; xe2 = X2[1]; ys2 = X2[2]; ye2 = X2[3]
    xs3 = X3[0]; xe3 = X3[1]; ys3 = X3[2]; ye3 = X3[3]
    xs4 = X4[0]; xe4 = X4[1]; ys4 = X4[2]; ye4 = X4[3]
    xs5 = X5[0]; xe5 = X5[1]; ys5 = X5[2]; ye5 = X5[3]
    xs6 = X6[0]; xe6 = X6[1]; ys6 = X6[2]; ye6 = X6[3]
    xs7 = X7[0]; xe7 = X7[1]; ys7 = X7[2]; ye7 = X7[3]
    xs8 = X8[0]; xe8 = X8[1]; ys8 = X8[2]; ye8 = X8[3]
    xs9 = X9[0]; xe9 = X9[1]; ys9 = X9[2]; ye9 = X9[3]
    xs10 = X10[0]; xe10 = X10[1]; ys10 = X10[2]; ye10 = X10[3]

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
    # Obstacle 10
    u[xs10-1:xe10+1, ye10] = 2*uwall - u[xs10-1:xe10+1, ye10+1]    # north face
    u[xs10-1:xe10+1, ys10] = 2*uwall - u[xs10-1:xe10+1, ys10-1]    # south face
    v[xs10, ys10-1:ye10+1] = 2*uwall - v[xs10-1, ys10-1:ye10+1]    # west  face
    v[xe10, ys10-1:ye10+1] = 2*uwall - v[xe10+1, ys10-1:ye10+1]    # east  face

    # near Obstacle 1
    u[xs11-1:xe11+1, ye11] = 2*uwall - u[xs11-1:xe11+1, ye11+1]    # north face
    u[xs11-1:xe11+1, ys11] = 2*uwall - u[xs11-1:xe11+1, ys11-1]    # south face
    v[xs11, ys11-1:ye11+1] = 2*uwall - v[xs11-1, ys11-1:ye11+1]    # west  face
    v[xe11, ys11-1:ye11+1] = 2*uwall - v[xe11+1, ys11-1:ye11+1]    # east  face
    u[xs12-1:xe12+1, ye12] = 2*uwall - u[xs12-1:xe12+1, ye12+1]    # north face
    u[xs12-1:xe12+1, ys12] = 2*uwall - u[xs12-1:xe12+1, ys12-1]    # south face
    v[xs12, ys12-1:ye12+1] = 2*uwall - v[xs12-1, ys12-1:ye12+1]    # west  face
    v[xe12, ys12-1:ye12+1] = 2*uwall - v[xe12+1, ys12-1:ye12+1]    # east  face
    # near Obstacle 2
    u[xs21-1:xe21+1, ye21] = 2*uwall - u[xs21-1:xe21+1, ye21+1]    # north face
    u[xs21-1:xe21+1, ys21] = 2*uwall - u[xs21-1:xe21+1, ys21-1]    # south face
    v[xs21, ys21-1:ye21+1] = 2*uwall - v[xs21-1, ys21-1:ye21+1]    # west  face
    v[xe21, ys21-1:ye21+1] = 2*uwall - v[xe21+1, ys21-1:ye21+1]    # east  face
    u[xs22-1:xe22+1, ye22] = 2*uwall - u[xs22-1:xe22+1, ye22+1]    # north face
    u[xs22-1:xe22+1, ys22] = 2*uwall - u[xs22-1:xe22+1, ys22-1]    # south face
    v[xs22, ys22-1:ye22+1] = 2*uwall - v[xs22-1, ys22-1:ye22+1]    # west  face
    v[xe22, ys22-1:ye22+1] = 2*uwall - v[xe22+1, ys22-1:ye22+1]    # east  face
    # near Obstacle 3
    u[xs31-1:xe31+1, ye31] = 2*uwall - u[xs31-1:xe31+1, ye31+1]    # north face
    u[xs31-1:xe31+1, ys31] = 2*uwall - u[xs31-1:xe31+1, ys31-1]    # south face
    v[xs31, ys31-1:ye31+1] = 2*uwall - v[xs31-1, ys31-1:ye31+1]    # west  face
    v[xe31, ys31-1:ye31+1] = 2*uwall - v[xe31+1, ys31-1:ye31+1]    # east  face
    u[xs32-1:xe32+1, ye32] = 2*uwall - u[xs32-1:xe32+1, ye32+1]    # north face
    u[xs32-1:xe32+1, ys32] = 2*uwall - u[xs32-1:xe32+1, ys32-1]    # south face
    v[xs32, ys32-1:ye32+1] = 2*uwall - v[xs32-1, ys32-1:ye32+1]    # west  face
    v[xe32, ys32-1:ye32+1] = 2*uwall - v[xe32+1, ys32-1:ye32+1]    # east  face
    # near Obstacle 4
    u[xs41-1:xe41+1, ye41] = 2*uwall - u[xs41-1:xe41+1, ye41+1]    # north face
    u[xs41-1:xe41+1, ys41] = 2*uwall - u[xs41-1:xe41+1, ys41-1]    # south face
    v[xs41, ys41-1:ye41+1] = 2*uwall - v[xs41-1, ys41-1:ye41+1]    # west  face
    v[xe41, ys41-1:ye41+1] = 2*uwall - v[xe41+1, ys41-1:ye41+1]    # east  face
    u[xs42-1:xe42+1, ye42] = 2*uwall - u[xs42-1:xe42+1, ye42+1]    # north face
    u[xs42-1:xe42+1, ys42] = 2*uwall - u[xs42-1:xe42+1, ys42-1]    # south face
    v[xs42, ys42-1:ye42+1] = 2*uwall - v[xs42-1, ys42-1:ye42+1]    # west  face
    v[xe42, ys42-1:ye42+1] = 2*uwall - v[xe42+1, ys42-1:ye42+1]    # east  face
    # near Obstacle 5
    u[xs51-1:xe51+1, ye51] = 2*uwall - u[xs51-1:xe51+1, ye51+1]    # north face
    u[xs51-1:xe51+1, ys51] = 2*uwall - u[xs51-1:xe51+1, ys51-1]    # south face
    v[xs51, ys51-1:ye51+1] = 2*uwall - v[xs51-1, ys51-1:ye51+1]    # west  face
    v[xe51, ys51-1:ye51+1] = 2*uwall - v[xe51+1, ys51-1:ye51+1]    # east  face
    u[xs52-1:xe52+1, ye52] = 2*uwall - u[xs52-1:xe52+1, ye52+1]    # north face
    u[xs52-1:xe52+1, ys52] = 2*uwall - u[xs52-1:xe52+1, ys52-1]    # south face
    v[xs52, ys52-1:ye52+1] = 2*uwall - v[xs52-1, ys52-1:ye52+1]    # west  face
    v[xe52, ys52-1:ye52+1] = 2*uwall - v[xe52+1, ys52-1:ye52+1]    # east  face
    # near Obstacle 6
    u[xs61-1:xe61+1, ye61] = 2*uwall - u[xs61-1:xe61+1, ye61+1]    # north face
    u[xs61-1:xe61+1, ys61] = 2*uwall - u[xs61-1:xe61+1, ys61-1]    # south face
    v[xs61, ys61-1:ye61+1] = 2*uwall - v[xs61-1, ys61-1:ye61+1]    # west  face
    v[xe61, ys61-1:ye61+1] = 2*uwall - v[xe61+1, ys61-1:ye61+1]    # east  face
    u[xs62-1:xe62+1, ye62] = 2*uwall - u[xs62-1:xe62+1, ye62+1]    # north face
    u[xs62-1:xe62+1, ys62] = 2*uwall - u[xs62-1:xe62+1, ys62-1]    # south face
    v[xs62, ys62-1:ye62+1] = 2*uwall - v[xs62-1, ys62-1:ye62+1]    # west  face
    v[xe62, ys62-1:ye62+1] = 2*uwall - v[xe62+1, ys62-1:ye62+1]    # east  face
    # near Obstacle 7
    u[xs71-1:xe71+1, ye71] = 2*uwall - u[xs71-1:xe71+1, ye71+1]    # north face
    u[xs71-1:xe71+1, ys71] = 2*uwall - u[xs71-1:xe71+1, ys71-1]    # south face
    v[xs71, ys71-1:ye71+1] = 2*uwall - v[xs71-1, ys71-1:ye71+1]    # west  face
    v[xe71, ys71-1:ye71+1] = 2*uwall - v[xe71+1, ys71-1:ye71+1]    # east  face
    u[xs72-1:xe72+1, ye72] = 2*uwall - u[xs72-1:xe72+1, ye72+1]    # north face
    u[xs72-1:xe72+1, ys72] = 2*uwall - u[xs72-1:xe72+1, ys72-1]    # south face
    v[xs72, ys72-1:ye72+1] = 2*uwall - v[xs72-1, ys72-1:ye72+1]    # west  face
    v[xe72, ys72-1:ye72+1] = 2*uwall - v[xe72+1, ys72-1:ye72+1]    # east  face
    # near Obstacle 8
    u[xs81-1:xe81+1, ye81] = 2*uwall - u[xs81-1:xe81+1, ye81+1]    # north face
    u[xs81-1:xe81+1, ys81] = 2*uwall - u[xs81-1:xe81+1, ys81-1]    # south face
    v[xs81, ys81-1:ye81+1] = 2*uwall - v[xs81-1, ys81-1:ye81+1]    # west  face
    v[xe81, ys81-1:ye81+1] = 2*uwall - v[xe81+1, ys81-1:ye81+1]    # east  face
    u[xs82-1:xe82+1, ye82] = 2*uwall - u[xs82-1:xe82+1, ye82+1]    # north face
    u[xs82-1:xe82+1, ys82] = 2*uwall - u[xs82-1:xe82+1, ys82-1]    # south face
    v[xs82, ys82-1:ye82+1] = 2*uwall - v[xs82-1, ys82-1:ye82+1]    # west  face
    v[xe82, ys82-1:ye82+1] = 2*uwall - v[xe82+1, ys82-1:ye82+1]    # east  face
    # near Obstacle 9
    u[xs91-1:xe91+1, ye91] = 2*uwall - u[xs91-1:xe91+1, ye91+1]    # north face
    u[xs91-1:xe91+1, ys91] = 2*uwall - u[xs91-1:xe91+1, ys91-1]    # south face
    v[xs91, ys91-1:ye91+1] = 2*uwall - v[xs91-1, ys91-1:ye91+1]    # west  face
    v[xe91, ys91-1:ye91+1] = 2*uwall - v[xe91+1, ys91-1:ye91+1]    # east  face
    u[xs92-1:xe92+1, ye92] = 2*uwall - u[xs92-1:xe92+1, ye92+1]    # north face
    u[xs92-1:xe92+1, ys92] = 2*uwall - u[xs92-1:xe92+1, ys92-1]    # south face
    v[xs92, ys92-1:ye92+1] = 2*uwall - v[xs92-1, ys92-1:ye92+1]    # west  face
    v[xe92, ys92-1:ye92+1] = 2*uwall - v[xe92+1, ys92-1:ye92+1]    # east  face

    return u, v



# Advection-Diffusion equation for species
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

                phi[i, j] = phio[i, j] + dt/(hx*hy) * (-A)

    return phi



# Boundary conditions for species
def SpeciesBCs(phi, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10):

    # Defining the obstacles coordinates
    xs1 = X1[0]; xe1 = X1[1]; ys1 = X1[2]; ye1 = X1[3]
    xs2 = X2[0]; xe2 = X2[1]; ys2 = X2[2]; ye2 = X2[3]
    xs3 = X3[0]; xe3 = X3[1]; ys3 = X3[2]; ye3 = X3[3]
    xs4 = X4[0]; xe4 = X4[1]; ys4 = X4[2]; ye4 = X4[3]
    xs5 = X5[0]; xe5 = X5[1]; ys5 = X5[2]; ye5 = X5[3]
    xs6 = X6[0]; xe6 = X6[1]; ys6 = X6[2]; ye6 = X6[3]
    xs7 = X7[0]; xe7 = X7[1]; ys7 = X7[2]; ye7 = X7[3]
    xs8 = X8[0]; xe8 = X8[1]; ys8 = X8[2]; ye8 = X8[3]
    xs9 = X9[0]; xe9 = X9[1]; ys9 = X9[2]; ye9 = X9[3]
    xs10 = X10[0]; xe10 = X10[1]; ys10 = X10[2]; ye10 = X10[3]

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

    # Entire domain BCs (Dirichlet)
    phi[1:-1, 0] = phi[1:-1, 1];           # South wall
    phi[1:-1, -1] = phi[1:-1, -2];         # North wall
    phi[0, 1:-1] = phi[1, 1:-1];           # West wall
    phi[-1, 1:-1] = phi[-2, 1:-1];         # East wall

    # 1st Obstacle BCS (Dirichlet)
    phi[xs1:xe1+1, ys1] = phi[xs1:xe1+1, ys1-1];     # South wall
    phi[xs1:xe1+1, ye1] = phi[xs1:xe1+1, ye1+1];     # North wall
    phi[xs1, ys1:ye1+1] = phi[xs1-1, ys1:ye1+1];     # West wall
    phi[xe1, ys1:ye1+1] = phi[xe1+1, ys1:ye1+1];     # East wall
    # 2nd Obstacle BCS (Dirichlet)
    phi[xs2:xe2+1, ys2] = phi[xs2:xe2+1, ys2-1];     # South wall
    phi[xs2:xe2+1, ye2] = phi[xs2:xe2+1, ye2+1];     # North wall
    phi[xs2, ys2:ye2+1] = phi[xs2-1, ys2:ye2+1];     # West wall
    phi[xe2, ys2:ye2+1] = phi[xe2+1, ys2:ye2+1];     # East wall
    # 3rd Obstacle BCS (Dirichlet)
    phi[xs3:xe3+1, ys3] = phi[xs3:xe3+1, ys3-1];     # South wall
    phi[xs3:xe3+1, ye3] = phi[xs3:xe3+1, ye3+1];     # North wall
    phi[xs3, ys3:ye3+1] = phi[xs3-1, ys3:ye3+1];     # West wall
    phi[xe3, ys3:ye3+1] = phi[xe3+1, ys3:ye3+1];     # East wall
    # 4th Obstacle BCS (Dirichlet)
    phi[xs4:xe4+1, ys4] = phi[xs4:xe4+1, ys4-1];     # South wall
    phi[xs4:xe4+1, ye4] = phi[xs4:xe4+1, ye4+1];     # North wall
    phi[xs4, ys4:ye4+1] = phi[xs4-1, ys4:ye4+1];     # West wall
    phi[xe4, ys4:ye4+1] = phi[xe4+1, ys4:ye4+1];     # East wall
    # 5th Obstacle BCS (Dirichlet)
    phi[xs5:xe5+1, ys5] = phi[xs5:xe5+1, ys5-1];     # South wall
    phi[xs5:xe5+1, ye5] = phi[xs5:xe5+1, ye5+1];     # North wall
    phi[xs5, ys5:ye5+1] = phi[xs5-1, ys5:ye5+1];     # West wall
    phi[xe5, ys5:ye5+1] = phi[xe5+1, ys5:ye5+1];     # East wall
    # 6th Obstacle BCS (Dirichlet)
    phi[xs6:xe6+1, ys6] = phi[xs6:xe6+1, ys6-1];     # South wall
    phi[xs6:xe6+1, ye6] = phi[xs6:xe6+1, ye6+1];     # North wall
    phi[xs6, ys6:ye6+1] = phi[xs6-1, ys6:ye6+1];     # West wall
    phi[xe6, ys6:ye6+1] = phi[xe6+1, ys6:ye6+1];     # East wall
    # 7th Obstacle BCS (Dirichlet)
    phi[xs7:xe7+1, ys7] = phi[xs7:xe7+1, ys7-1];     # South wall
    phi[xs7:xe7+1, ye7] = phi[xs7:xe7+1, ye7+1];     # North wall
    phi[xs7, ys7:ye7+1] = phi[xs7-1, ys7:ye7+1];     # West wall
    phi[xe7, ys7:ye7+1] = phi[xe7+1, ys7:ye7+1];     # East wall
    # 8th Obstacle BCS (Dirichlet)
    phi[xs8:xe8+1, ys8] = phi[xs8:xe8+1, ys8-1];     # South wall
    phi[xs8:xe8+1, ye8] = phi[xs8:xe8+1, ye8+1];     # North wall
    phi[xs8, ys8:ye8+1] = phi[xs8-1, ys8:ye8+1];     # West wall
    phi[xe8, ys8:ye8+1] = phi[xe8+1, ys8:ye8+1];     # East wall
    # 9th Obstacle BCS (Dirichlet)
    phi[xs9:xe9+1, ys9] = phi[xs9:xe9+1, ys9-1];     # South wall
    phi[xs9:xe9+1, ye9] = phi[xs9:xe9+1, ye9+1];     # North wall
    phi[xs9, ys9:ye9+1] = phi[xs9-1, ys9:ye9+1];     # West wall
    phi[xe9, ys9:ye9+1] = phi[xe9+1, ys9:ye9+1];     # East wall
    # 10th Obstacle BCS (Dirichlet)
    phi[xs10:xe10+1, ys10] = phi[xs10:xe10+1, ys10-1];     # South wall
    phi[xs10:xe10+1, ye10] = phi[xs10:xe10+1, ye10+1];     # North wall
    phi[xs10, ys10:ye10+1] = phi[xs10-1, ys10:ye10+1];     # West wall
    phi[xe10, ys10:ye10+1] = phi[xe10+1, ys10:ye10+1];     # East wall

    # near 1st Obstacle BCS (Dirichlet)
    phi[xs11:xe11+1, ys11] = phi[xs11:xe11+1, ys11-1];     # South wall
    phi[xs11:xe11+1, ye11] = phi[xs11:xe11+1, ye11+1];     # North wall
    phi[xs11, ys11:ye11+1] = phi[xs11-1, ys11:ye11+1];     # West wall
    phi[xe11, ys11:ye11+1] = phi[xe11+1, ys11:ye11+1];     # East wall
    phi[xs12:xe12+1, ys12] = phi[xs12:xe12+1, ys12-1];     # South wall
    phi[xs12:xe12+1, ye12] = phi[xs12:xe12+1, ye12+1];     # North wall
    phi[xs12, ys12:ye12+1] = phi[xs12-1, ys12:ye12+1];     # West wall
    phi[xe12, ys12:ye12+1] = phi[xe12+1, ys12:ye12+1];     # East wall
    # near 2nd Obstacle BCS (Dirichlet)
    phi[xs21:xe21+1, ys21] = phi[xs21:xe21+1, ys21-1];     # South wall
    phi[xs21:xe21+1, ye21] = phi[xs21:xe21+1, ye21+1];     # North wall
    phi[xs21, ys21:ye21+1] = phi[xs21-1, ys21:ye21+1];     # West wall
    phi[xe21, ys21:ye21+1] = phi[xe21+1, ys21:ye21+1];     # East wall
    phi[xs22:xe22+1, ys22] = phi[xs22:xe22+1, ys22-1];     # South wall
    phi[xs22:xe22+1, ye22] = phi[xs22:xe22+1, ye22+1];     # North wall
    phi[xs22, ys22:ye22+1] = phi[xs22-1, ys22:ye22+1];     # West wall
    phi[xe22, ys22:ye22+1] = phi[xe22+1, ys22:ye22+1];     # East wall
    # near 3rd Obstacle BCS (Dirichlet)
    phi[xs31:xe31+1, ys31] = phi[xs31:xe31+1, ys31-1];     # South wall
    phi[xs31:xe31+1, ye31] = phi[xs31:xe31+1, ye31+1];     # North wall
    phi[xs31, ys31:ye31+1] = phi[xs31-1, ys31:ye31+1];     # West wall
    phi[xe31, ys31:ye31+1] = phi[xe31+1, ys31:ye31+1];     # East wall
    phi[xs32:xe32+1, ys32] = phi[xs32:xe32+1, ys32-1];     # South wall
    phi[xs32:xe32+1, ye32] = phi[xs32:xe32+1, ye32+1];     # North wall
    phi[xs32, ys32:ye32+1] = phi[xs32-1, ys32:ye32+1];     # West wall
    phi[xe32, ys32:ye32+1] = phi[xe32+1, ys32:ye32+1];     # East wall
    # near 4th Obstacle BCS (Dirichlet)
    phi[xs41:xe41+1, ys41] = phi[xs41:xe41+1, ys41-1];     # South wall
    phi[xs41:xe41+1, ye41] = phi[xs41:xe41+1, ye41+1];     # North wall
    phi[xs41, ys41:ye41+1] = phi[xs41-1, ys41:ye41+1];     # West wall
    phi[xe41, ys41:ye41+1] = phi[xe41+1, ys41:ye41+1];     # East wall
    phi[xs42:xe42+1, ys42] = phi[xs42:xe42+1, ys42-1];     # South wall
    phi[xs42:xe42+1, ye42] = phi[xs42:xe42+1, ye42+1];     # North wall
    phi[xs42, ys42:ye42+1] = phi[xs42-1, ys42:ye42+1];     # West wall
    phi[xe42, ys42:ye42+1] = phi[xe42+1, ys42:ye42+1];     # East wall
    # near 5th Obstacle BCS (Dirichlet)
    phi[xs51:xe51+1, ys51] = phi[xs51:xe51+1, ys51-1];     # South wall
    phi[xs51:xe51+1, ye51] = phi[xs51:xe51+1, ye51+1];     # North wall
    phi[xs51, ys51:ye51+1] = phi[xs51-1, ys51:ye51+1];     # West wall
    phi[xe51, ys51:ye51+1] = phi[xe51+1, ys51:ye51+1];     # East wall
    phi[xs52:xe52+1, ys52] = phi[xs52:xe52+1, ys52-1];     # South wall
    phi[xs52:xe52+1, ye52] = phi[xs52:xe52+1, ye52+1];     # North wall
    phi[xs52, ys52:ye52+1] = phi[xs52-1, ys52:ye52+1];     # West wall
    phi[xe52, ys52:ye52+1] = phi[xe52+1, ys52:ye52+1];     # East wall
    # near 6th Obstacle BCS (Dirichlet)
    phi[xs61:xe61+1, ys61] = phi[xs61:xe61+1, ys61-1];     # South wall
    phi[xs61:xe61+1, ye61] = phi[xs61:xe61+1, ye61+1];     # North wall
    phi[xs61, ys61:ye61+1] = phi[xs61-1, ys61:ye61+1];     # West wall
    phi[xe61, ys61:ye61+1] = phi[xe61+1, ys61:ye61+1];     # East wall
    phi[xs62:xe62+1, ys62] = phi[xs62:xe62+1, ys62-1];     # South wall
    phi[xs62:xe62+1, ye62] = phi[xs62:xe62+1, ye62+1];     # North wall
    phi[xs62, ys62:ye62+1] = phi[xs62-1, ys62:ye62+1];     # West wall
    phi[xe62, ys62:ye62+1] = phi[xe62+1, ys62:ye62+1];     # East wall
    # near 7th Obstacle BCS (Dirichlet)
    phi[xs71:xe71+1, ys71] = phi[xs71:xe71+1, ys71-1];     # South wall
    phi[xs71:xe71+1, ye71] = phi[xs71:xe71+1, ye71+1];     # North wall
    phi[xs71, ys71:ye71+1] = phi[xs71-1, ys71:ye71+1];     # West wall
    phi[xe71, ys71:ye71+1] = phi[xe71+1, ys71:ye71+1];     # East wall
    phi[xs72:xe72+1, ys72] = phi[xs72:xe72+1, ys72-1];     # South wall
    phi[xs72:xe72+1, ye72] = phi[xs72:xe72+1, ye72+1];     # North wall
    phi[xs72, ys72:ye72+1] = phi[xs72-1, ys72:ye72+1];     # West wall
    phi[xe72, ys72:ye72+1] = phi[xe72+1, ys72:ye72+1];     # East wall
    # near 8th Obstacle BCS (Dirichlet)
    phi[xs81:xe81+1, ys81] = phi[xs81:xe81+1, ys81-1];     # South wall
    phi[xs81:xe81+1, ye81] = phi[xs81:xe81+1, ye81+1];     # North wall
    phi[xs81, ys81:ye81+1] = phi[xs81-1, ys81:ye81+1];     # West wall
    phi[xe81, ys81:ye81+1] = phi[xe81+1, ys81:ye81+1];     # East wall
    phi[xs82:xe82+1, ys82] = phi[xs82:xe82+1, ys82-1];     # South wall
    phi[xs82:xe82+1, ye82] = phi[xs82:xe82+1, ye82+1];     # North wall
    phi[xs82, ys82:ye82+1] = phi[xs82-1, ys82:ye82+1];     # West wall
    phi[xe82, ys82:ye82+1] = phi[xe82+1, ys82:ye82+1];     # East wall
    # near 9th Obstacle BCS (Dirichlet)
    phi[xs91:xe91+1, ys91] = phi[xs91:xe91+1, ys91-1];     # South wall
    phi[xs91:xe91+1, ye91] = phi[xs91:xe91+1, ye91+1];     # North wall
    phi[xs91, ys91:ye91+1] = phi[xs91-1, ys91:ye91+1];     # West wall
    phi[xe91, ys91:ye91+1] = phi[xe91+1, ys91:ye91+1];     # East wall
    phi[xs92:xe92+1, ys92] = phi[xs92:xe92+1, ys92-1];     # South wall
    phi[xs92:xe92+1, ye92] = phi[xs92:xe92+1, ye92+1];     # North wall
    phi[xs92, ys92:ye92+1] = phi[xs92-1, ys92:ye92+1];     # West wall
    phi[xe92, ys92:ye92+1] = phi[xe92+1, ys92:ye92+1];     # East wall

    return phi



# Reaction step after Advection-Diffusion of species
@njit
def ReactionStep(cO2star, cBstar, cGstar, dt, mu_max, K_G, nx, ny):

    cO2 = cO2star
    cB = cBstar
    cG = cGstar

    for i in range(1, nx+1):
        for j in range(1, ny+1):

            cG[i,j] = ( -(K_G - cGstar[i,j] + mu_max*dt*cBstar[i,j]) + ((K_G - cGstar[i,j] + mu_max*dt*cBstar[i,j])**2 + 4*K_G*cGstar[i,j])**0.5 ) / 2
            cB[i,j] = cBstar[i,j] / (1 - mu_max * dt * cG[i,j] / (K_G + cG[i,j]))

    return cO2, cB, cG



# Needed for graphical interpolation 
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



# Graphical representation of obstacles
def Graphical_obstacles(uu, vv, pp, tau, ccO2, ccB, ccG, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10):

    # Defining the obstacles coordinates
    xs1 = X1[0]; xe1 = X1[1]; ys1 = X1[2]; ye1 = X1[3]
    xs2 = X2[0]; xe2 = X2[1]; ys2 = X2[2]; ye2 = X2[3]
    xs3 = X3[0]; xe3 = X3[1]; ys3 = X3[2]; ye3 = X3[3]
    xs4 = X4[0]; xe4 = X4[1]; ys4 = X4[2]; ye4 = X4[3]
    xs5 = X5[0]; xe5 = X5[1]; ys5 = X5[2]; ye5 = X5[3]
    xs6 = X6[0]; xe6 = X6[1]; ys6 = X6[2]; ye6 = X6[3]
    xs7 = X7[0]; xe7 = X7[1]; ys7 = X7[2]; ye7 = X7[3]
    xs8 = X8[0]; xe8 = X8[1]; ys8 = X8[2]; ye8 = X8[3]
    xs9 = X9[0]; xe9 = X9[1]; ys9 = X9[2]; ye9 = X9[3]
    xs10 = X10[0]; xe10 = X10[1]; ys10 = X10[2]; ye10 = X10[3]

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
    
    @njit
    def graphical(X, n, uu, vv, pp, tau, ccO2, ccB, ccG):
        for i in range(0, n):      # n = number of obstacles + 1
            xs = X[4*i]; xe = X[4*i+1]; ys = X[4*i+2]; ye = X[4*i+3]
            uu[xs-1:xe+1, ys-1:ye+1] = 0         # Obstacle i+1
            vv[xs-1:xe+1, ys-1:ye+1] = 0
            pp[xs-1:xe+1, ys-1:ye+1] = 0
            tau[xs-1:xe+1, ys-1:ye+1] = 0
            ccO2[xs-1:xe+1, ys-1:ye+1] = 0
            ccB[xs-1:xe+1, ys-1:ye+1] = 0
            ccG[xs-1:xe+1, ys-1:ye+1] = 0

        return uu, vv, pp, tau, ccO2, ccB, ccG

    uu, vv, pp, tau, ccO2, ccB, ccG = graphical(X1, 3, uu, vv, pp, tau, ccO2, ccB, ccG)
    uu, vv, pp, tau, ccO2, ccB, ccG = graphical(X2, 3, uu, vv, pp, tau, ccO2, ccB, ccG)
    uu, vv, pp, tau, ccO2, ccB, ccG = graphical(X3, 3, uu, vv, pp, tau, ccO2, ccB, ccG)
    uu, vv, pp, tau, ccO2, ccB, ccG = graphical(X4, 3, uu, vv, pp, tau, ccO2, ccB, ccG)
    uu, vv, pp, tau, ccO2, ccB, ccG = graphical(X5, 3, uu, vv, pp, tau, ccO2, ccB, ccG)
    uu, vv, pp, tau, ccO2, ccB, ccG = graphical(X6, 3, uu, vv, pp, tau, ccO2, ccB, ccG)
    uu, vv, pp, tau, ccO2, ccB, ccG = graphical(X7, 3, uu, vv, pp, tau, ccO2, ccB, ccG)
    uu, vv, pp, tau, ccO2, ccB, ccG = graphical(X8, 3, uu, vv, pp, tau, ccO2, ccB, ccG)
    uu, vv, pp, tau, ccO2, ccB, ccG = graphical(X9, 3, uu, vv, pp, tau, ccO2, ccB, ccG)
    uu, vv, pp, tau, ccO2, ccB, ccG = graphical(X10, 1, uu, vv, pp, tau, ccO2, ccB, ccG)


    '''
    uu[xs1-1:xe1+1, ys1-1:ye1+1] = 0         # Obstacle 1
    vv[xs1-1:xe1+1, ys1-1:ye1+1] = 0
    pp[xs1-1:xe1+1, ys1-1:ye1+1] = 0
    tau[xs1-1:xe1+1, ys1-1:ye1+1] = 0
    ccO2[xs1-1:xe1+1, ys1-1:ye1+1] = 0
    ccB[xs1-1:xe1+1, ys1-1:ye1+1] = 0
    ccG[xs1-1:xe1+1, ys1-1:ye1+1] = 0

    uu[xs2-1:xe2+1, ys2-1:ye2+1] = 0         # Obstacle 2
    vv[xs2-1:xe2+1, ys2-1:ye2+1] = 0
    pp[xs2-1:xe2+1, ys2-1:ye2+1] = 0
    tau[xs2-1:xe2+1, ys2-1:ye2+1] = 0
    ccO2[xs2-1:xe2+1, ys2-1:ye2+1] = 0
    ccB[xs2-1:xe2+1, ys2-1:ye2+1] = 0
    ccG[xs2-1:xe2+1, ys2-1:ye2+1] = 0

    uu[xs3-1:xe3+1, ys3-1:ye3+1] = 0         # Obstacle 3
    vv[xs3-1:xe3+1, ys3-1:ye3+1] = 0
    pp[xs3-1:xe3+1, ys3-1:ye3+1] = 0
    tau[xs3-1:xe3+1, ys3-1:ye3+1] = 0
    ccO2[xs3-1:xe3+1, ys3-1:ye3+1] = 0
    ccB[xs3-1:xe3+1, ys3-1:ye3+1] = 0
    ccG[xs3-1:xe3+1, ys3-1:ye3+1] = 0

    uu[xs4-1:xe4+1, ys4-1:ye4+1] = 0         # Obstacle 4
    vv[xs4-1:xe4+1, ys4-1:ye4+1] = 0
    pp[xs4-1:xe4+1, ys4-1:ye4+1] = 0
    tau[xs4-1:xe4+1, ys4-1:ye4+1] = 0
    ccO2[xs4-1:xe4+1, ys4-1:ye4+1] = 0
    ccB[xs4-1:xe4+1, ys4-1:ye4+1] = 0
    ccG[xs4-1:xe4+1, ys4-1:ye4+1] = 0

    uu[xs5-1:xe5+1, ys5-1:ye5+1] = 0         # Obstacle 5
    vv[xs5-1:xe5+1, ys5-1:ye5+1] = 0
    pp[xs5-1:xe5+1, ys5-1:ye5+1] = 0
    tau[xs5-1:xe5+1, ys5-1:ye5+1] = 0
    ccO2[xs5-1:xe5+1, ys5-1:ye5+1] = 0
    ccB[xs5-1:xe5+1, ys5-1:ye5+1] = 0
    ccG[xs5-1:xe5+1, ys5-1:ye5+1] = 0

    uu[xs6-1:xe6+1, ys6-1:ye6+1] = 0         # Obstacle 6
    vv[xs6-1:xe6+1, ys6-1:ye6+1] = 0
    pp[xs6-1:xe6+1, ys6-1:ye6+1] = 0
    tau[xs6-1:xe6+1, ys6-1:ye6+1] = 0
    ccO2[xs6-1:xe6+1, ys6-1:ye6+1] = 0
    ccB[xs6-1:xe6+1, ys6-1:ye6+1] = 0
    ccG[xs6-1:xe6+1, ys6-1:ye6+1] = 0

    uu[xs7-1:xe7+1, ys7-1:ye7+1] = 0         # Obstacle 7
    vv[xs7-1:xe7+1, ys7-1:ye7+1] = 0
    pp[xs7-1:xe7+1, ys7-1:ye7+1] = 0
    tau[xs7-1:xe7+1, ys7-1:ye7+1] = 0
    ccO2[xs7-1:xe7+1, ys7-1:ye7+1] = 0
    ccB[xs7-1:xe7+1, ys7-1:ye7+1] = 0
    ccG[xs7-1:xe7+1, ys7-1:ye7+1] = 0

    uu[xs8-1:xe8+1, ys8-1:ye8+1] = 0         # Obstacle 8
    vv[xs8-1:xe8+1, ys8-1:ye8+1] = 0
    pp[xs8-1:xe8+1, ys8-1:ye8+1] = 0
    tau[xs8-1:xe8+1, ys8-1:ye8+1] = 0
    ccO2[xs8-1:xe8+1, ys8-1:ye8+1] = 0
    ccB[xs8-1:xe8+1, ys8-1:ye8+1] = 0
    ccG[xs8-1:xe8+1, ys8-1:ye8+1] = 0

    uu[xs9-1:xe9+1, ys9-1:ye9+1] = 0         # Obstacle 9
    vv[xs9-1:xe9+1, ys9-1:ye9+1] = 0
    pp[xs9-1:xe9+1, ys9-1:ye9+1] = 0
    tau[xs9-1:xe9+1, ys9-1:ye9+1] = 0
    ccO2[xs9-1:xe9+1, ys9-1:ye9+1] = 0
    ccB[xs9-1:xe9+1, ys9-1:ye9+1] = 0
    ccG[xs9-1:xe9+1, ys9-1:ye9+1] = 0

    uu[xs10-1:xe10+1, ys10-1:ye10+1] = 0         # Obstacle 10
    vv[xs10-1:xe10+1, ys10-1:ye10+1] = 0
    pp[xs10-1:xe10+1, ys10-1:ye10+1] = 0
    tau[xs10-1:xe10+1, ys10-1:ye10+1] = 0
    ccO2[xs10-1:xe10+1, ys10-1:ye10+1] = 0
    ccB[xs10-1:xe10+1, ys10-1:ye10+1] = 0
    ccG[xs10-1:xe10+1, ys10-1:ye10+1] = 0

    uu[xs11-1:xe11+1, ys11-1:ye11+1] = 0         # near Obstacle 1
    vv[xs11-1:xe11+1, ys11-1:ye11+1] = 0
    pp[xs11-1:xe11+1, ys11-1:ye11+1] = 0
    tau[xs11-1:xe11+1, ys11-1:ye11+1] = 0
    ccO2[xs11-1:xe11+1, ys11-1:ye11+1] = 0
    ccB[xs11-1:xe11+1, ys11-1:ye11+1] = 0
    ccG[xs11-1:xe11+1, ys11-1:ye11+1] = 0
    uu[xs12-1:xe12+1, ys12-1:ye12+1] = 0
    vv[xs12-1:xe12+1, ys12-1:ye12+1] = 0
    pp[xs12-1:xe12+1, ys12-1:ye12+1] = 0
    tau[xs12-1:xe12+1, ys12-1:ye12+1] = 0
    ccO2[xs12-1:xe12+1, ys12-1:ye12+1] = 0
    ccB[xs12-1:xe12+1, ys12-1:ye12+1] = 0
    ccG[xs12-1:xe12+1, ys12-1:ye12+1] = 0

    uu[xs2-1:xe2+1, ys2-1:ye2+1] = 0         # near Obstacle 2
    vv[xs2-1:xe2+1, ys2-1:ye2+1] = 0
    pp[xs2-1:xe2+1, ys2-1:ye2+1] = 0
    tau[xs2-1:xe2+1, ys2-1:ye2+1] = 0
    ccO2[xs2-1:xe2+1, ys2-1:ye2+1] = 0
    ccB[xs2-1:xe2+1, ys2-1:ye2+1] = 0
    ccG[xs2-1:xe2+1, ys2-1:ye2+1] = 0

    uu[xs3-1:xe3+1, ys3-1:ye3+1] = 0         # near Obstacle 3
    vv[xs3-1:xe3+1, ys3-1:ye3+1] = 0
    pp[xs3-1:xe3+1, ys3-1:ye3+1] = 0
    tau[xs3-1:xe3+1, ys3-1:ye3+1] = 0
    ccO2[xs3-1:xe3+1, ys3-1:ye3+1] = 0
    ccB[xs3-1:xe3+1, ys3-1:ye3+1] = 0
    ccG[xs3-1:xe3+1, ys3-1:ye3+1] = 0

    uu[xs4-1:xe4+1, ys4-1:ye4+1] = 0         # near Obstacle 4
    vv[xs4-1:xe4+1, ys4-1:ye4+1] = 0
    pp[xs4-1:xe4+1, ys4-1:ye4+1] = 0
    tau[xs4-1:xe4+1, ys4-1:ye4+1] = 0
    ccO2[xs4-1:xe4+1, ys4-1:ye4+1] = 0
    ccB[xs4-1:xe4+1, ys4-1:ye4+1] = 0
    ccG[xs4-1:xe4+1, ys4-1:ye4+1] = 0

    uu[xs5-1:xe5+1, ys5-1:ye5+1] = 0         # near Obstacle 5
    vv[xs5-1:xe5+1, ys5-1:ye5+1] = 0
    pp[xs5-1:xe5+1, ys5-1:ye5+1] = 0
    tau[xs5-1:xe5+1, ys5-1:ye5+1] = 0
    ccO2[xs5-1:xe5+1, ys5-1:ye5+1] = 0
    ccB[xs5-1:xe5+1, ys5-1:ye5+1] = 0
    ccG[xs5-1:xe5+1, ys5-1:ye5+1] = 0

    uu[xs6-1:xe6+1, ys6-1:ye6+1] = 0         # near Obstacle 6
    vv[xs6-1:xe6+1, ys6-1:ye6+1] = 0
    pp[xs6-1:xe6+1, ys6-1:ye6+1] = 0
    tau[xs6-1:xe6+1, ys6-1:ye6+1] = 0
    ccO2[xs6-1:xe6+1, ys6-1:ye6+1] = 0
    ccB[xs6-1:xe6+1, ys6-1:ye6+1] = 0
    ccG[xs6-1:xe6+1, ys6-1:ye6+1] = 0

    uu[xs7-1:xe7+1, ys7-1:ye7+1] = 0         # near Obstacle 7
    vv[xs7-1:xe7+1, ys7-1:ye7+1] = 0
    pp[xs7-1:xe7+1, ys7-1:ye7+1] = 0
    tau[xs7-1:xe7+1, ys7-1:ye7+1] = 0
    ccO2[xs7-1:xe7+1, ys7-1:ye7+1] = 0
    ccB[xs7-1:xe7+1, ys7-1:ye7+1] = 0
    ccG[xs7-1:xe7+1, ys7-1:ye7+1] = 0

    uu[xs8-1:xe8+1, ys8-1:ye8+1] = 0         # near Obstacle 8
    vv[xs8-1:xe8+1, ys8-1:ye8+1] = 0
    pp[xs8-1:xe8+1, ys8-1:ye8+1] = 0
    tau[xs8-1:xe8+1, ys8-1:ye8+1] = 0
    ccO2[xs8-1:xe8+1, ys8-1:ye8+1] = 0
    ccB[xs8-1:xe8+1, ys8-1:ye8+1] = 0
    ccG[xs8-1:xe8+1, ys8-1:ye8+1] = 0

    uu[xs9-1:xe9+1, ys9-1:ye9+1] = 0         # near Obstacle 9
    vv[xs9-1:xe9+1, ys9-1:ye9+1] = 0
    pp[xs9-1:xe9+1, ys9-1:ye9+1] = 0
    tau[xs9-1:xe9+1, ys9-1:ye9+1] = 0
    ccO2[xs9-1:xe9+1, ys9-1:ye9+1] = 0
    ccB[xs9-1:xe9+1, ys9-1:ye9+1] = 0
    ccG[xs9-1:xe9+1, ys9-1:ye9+1] = 0
    '''

    return uu, vv, pp, tau, ccO2, ccB, ccG



# Correction of velocity (last step of projection method)
@njit
def correction_velocity(u, v, ut, vt, p, nx, ny, hx, hy, dt, flagu, flagv):

    for i in range(1, nx):
        for j in range(1, ny+1):
            if flagu[i,j] == 0:
                u[i, j] = ut[i, j] - (dt/hx) * (p[i+1, j] - p[i, j])

    for i in range(1, nx+1):
        for j in range(1, ny):
            if flagv[i,j] == 0:
                v[i, j] = vt[i, j] - (dt/hy) * (p[i, j+1] - p[i, j])

    return u, v



# Shear stress function
def ShearStress(v, dR, nu, rhoL, nx, ny):

    tau = np.zeros([nx+2, ny+1])
    vo = np.absolute(v)

    for i in range(nx+2):
        for j in range(ny+1):

                tau[i,j] = 0.5 * 0.0792 * (vo[i,j] * dR / nu)**0.25 * rhoL * (vo[i,j])**2

    return tau



# Collecting the cells mean concentration
@njit
def CellsMeanConcentration(cB, th, X):

    xs = X[0]; xe = X[1]; ys = X[2]; ye = X[3]

    cBmean = 0

    for i in range(xs-th, xs):     # (pay attention if cells are above or under the plate!!!)
        for j in range(ys, ye+1):

            cBmean_o = cBmean
            cBmean = cBmean_o + cB[i,j]

    cBmean = cBmean / ((ye - ys) * th)

    return cBmean



# Plotting the results
def PlotFunctions(xx, yy, phi, x, y, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, Lx, Ly, title, x_label, y_label):

    xs1 = X1[0]; xe1 = X1[1]; ys1 = X1[2]; ye1 = X1[3]
    xs2 = X2[0]; xe2 = X2[1]; ys2 = X2[2]; ye2 = X2[3]
    xs3 = X3[0]; xe3 = X3[1]; ys3 = X3[2]; ye3 = X3[3]
    xs4 = X4[0]; xe4 = X4[1]; ys4 = X4[2]; ye4 = X4[3]
    xs5 = X5[0]; xe5 = X5[1]; ys5 = X5[2]; ye5 = X5[3]
    xs6 = X6[0]; xe6 = X6[1]; ys6 = X6[2]; ye6 = X6[3]
    xs7 = X7[0]; xe7 = X7[1]; ys7 = X7[2]; ye7 = X7[3]
    xs8 = X8[0]; xe8 = X8[1]; ys8 = X8[2]; ye8 = X8[3]
    xs9 = X9[0]; xe9 = X9[1]; ys9 = X9[2]; ye9 = X9[3]
    xs10 = X10[0]; xe10 = X10[1]; ys10 = X10[2]; ye10 = X10[3]

    xs11 = X1[4]; xe11 = X1[5]; ys11 = X1[6]; ye11 = X1[7]; xs12 = X1[8]; xe12 = X1[9]; ys12 = X1[10]; ye12 = X1[11]
    xs21 = X2[4]; xe21 = X2[5]; ys21 = X2[6]; ye21 = X2[7]; xs22 = X2[8]; xe22 = X2[9]; ys22 = X2[10]; ye22 = X2[11]
    xs31 = X3[4]; xe31 = X3[5]; ys31 = X3[6]; ye31 = X3[7]; xs32 = X3[8]; xe32 = X3[9]; ys32 = X3[10]; ye32 = X3[11]
    xs41 = X4[4]; xe41 = X4[5]; ys41 = X4[6]; ye41 = X4[7]; xs42 = X4[8]; xe42 = X4[9]; ys42 = X4[10]; ye42 = X4[11]
    xs51 = X5[4]; xe51 = X5[5]; ys51 = X5[6]; ye51 = X5[7]; xs52 = X5[8]; xe52 = X5[9]; ys52 = X5[10]; ye52 = X5[11]
    xs61 = X6[4]; xe61 = X6[5]; ys61 = X6[6]; ye61 = X6[7]; xs62 = X6[8]; xe62 = X6[9]; ys62 = X6[10]; ye62 = X6[11]
    xs71 = X7[4]; xe71 = X7[5]; ys71 = X7[6]; ye71 = X7[7]; xs72 = X7[8]; xe72 = X7[9]; ys72 = X7[10]; ye72 = X7[11]
    xs81 = X8[4]; xe81 = X8[5]; ys81 = X8[6]; ye81 = X8[7]; xs82 = X8[8]; xe82 = X8[9]; ys82 = X8[10]; ye82 = X8[11]
    xs91 = X9[4]; xe91 = X9[5]; ys91 = X9[6]; ye91 = X9[7]; xs92 = X9[8]; xe92 = X9[9]; ys92 = X9[10]; ye92 = X9[11]

    fig, ax = plt.subplots()
    plot = plt.contourf(xx, yy, np.transpose(phi))
    plt.colorbar(plot)
    ax.add_patch(Rectangle((x[xs1-1], y[ys1-1]), x[xe1-xs1+1], y[ye1-ys1+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs2-1], y[ys2-1]), x[xe2-xs2+1], y[ye2-ys2+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs3-1], y[ys3-1]), x[xe3-xs3+1], y[ye3-ys3+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs4-1], y[ys4-1]), x[xe4-xs4+1], y[ye4-ys4+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs5-1], y[ys5-1]), x[xe5-xs5+1], y[ye5-ys5+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs6-1], y[ys6-1]), x[xe6-xs6+1], y[ye6-ys6+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs7-1], y[ys7-1]), x[xe7-xs7+1], y[ye7-ys7+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs8-1], y[ys8-1]), x[xe8-xs8+1], y[ye8-ys8+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs9-1], y[ys9-1]), x[xe9-xs9+1], y[ye9-ys9+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs10-1], y[ys10-1]), x[xe10-xs10+1], y[ye10-ys10+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs11-1], y[ys11-1]), x[xe11-xs11+1], y[ye11-ys11+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs21-1], y[ys21-1]), x[xe21-xs21+1], y[ye21-ys21+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs31-1], y[ys31-1]), x[xe31-xs31+1], y[ye31-ys31+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs41-1], y[ys41-1]), x[xe41-xs41+1], y[ye41-ys41+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs51-1], y[ys51-1]), x[xe51-xs51+1], y[ye51-ys51+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs61-1], y[ys61-1]), x[xe61-xs61+1], y[ye61-ys61+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs71-1], y[ys71-1]), x[xe71-xs71+1], y[ye71-ys71+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs81-1], y[ys81-1]), x[xe81-xs81+1], y[ye81-ys81+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs91-1], y[ys91-1]), x[xe91-xs91+1], y[ye91-ys91+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs12-1], y[ys12-1]), x[xe12-xs12+1], y[ye12-ys12+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs22-1], y[ys22-1]), x[xe22-xs22+1], y[ye22-ys22+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs32-1], y[ys32-1]), x[xe32-xs32+1], y[ye32-ys32+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs42-1], y[ys42-1]), x[xe42-xs42+1], y[ye42-ys42+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs52-1], y[ys52-1]), x[xe52-xs52+1], y[ye52-ys52+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs62-1], y[ys62-1]), x[xe62-xs62+1], y[ye62-ys62+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs72-1], y[ys72-1]), x[xe72-xs72+1], y[ye72-ys72+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs82-1], y[ys82-1]), x[xe82-xs82+1], y[ye82-ys82+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs92-1], y[ys92-1]), x[xe92-xs92+1], y[ye92-ys92+1], facecolor = 'grey'))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(0,Lx)
    plt.ylim(0,Ly)



# Plotting the streamlines
def Streamlines(x, y, xx, yy, uu, vv, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, Lx, Ly, title, x_label, y_label):

    xs1 = X1[0]; xe1 = X1[1]; ys1 = X1[2]; ye1 = X1[3]
    xs2 = X2[0]; xe2 = X2[1]; ys2 = X2[2]; ye2 = X2[3]
    xs3 = X3[0]; xe3 = X3[1]; ys3 = X3[2]; ye3 = X3[3]
    xs4 = X4[0]; xe4 = X4[1]; ys4 = X4[2]; ye4 = X4[3]
    xs5 = X5[0]; xe5 = X5[1]; ys5 = X5[2]; ye5 = X5[3]
    xs6 = X6[0]; xe6 = X6[1]; ys6 = X6[2]; ye6 = X6[3]
    xs7 = X7[0]; xe7 = X7[1]; ys7 = X7[2]; ye7 = X7[3]
    xs8 = X8[0]; xe8 = X8[1]; ys8 = X8[2]; ye8 = X8[3]
    xs9 = X9[0]; xe9 = X9[1]; ys9 = X9[2]; ye9 = X9[3]
    xs10 = X10[0]; xe10 = X10[1]; ys10 = X10[2]; ye10 = X10[3]

    xs11 = X1[4]; xe11 = X1[5]; ys11 = X1[6]; ye11 = X1[7]; xs12 = X1[8]; xe12 = X1[9]; ys12 = X1[10]; ye12 = X1[11]
    xs21 = X2[4]; xe21 = X2[5]; ys21 = X2[6]; ye21 = X2[7]; xs22 = X2[8]; xe22 = X2[9]; ys22 = X2[10]; ye22 = X2[11]
    xs31 = X3[4]; xe31 = X3[5]; ys31 = X3[6]; ye31 = X3[7]; xs32 = X3[8]; xe32 = X3[9]; ys32 = X3[10]; ye32 = X3[11]
    xs41 = X4[4]; xe41 = X4[5]; ys41 = X4[6]; ye41 = X4[7]; xs42 = X4[8]; xe42 = X4[9]; ys42 = X4[10]; ye42 = X4[11]
    xs51 = X5[4]; xe51 = X5[5]; ys51 = X5[6]; ye51 = X5[7]; xs52 = X5[8]; xe52 = X5[9]; ys52 = X5[10]; ye52 = X5[11]
    xs61 = X6[4]; xe61 = X6[5]; ys61 = X6[6]; ye61 = X6[7]; xs62 = X6[8]; xe62 = X6[9]; ys62 = X6[10]; ye62 = X6[11]
    xs71 = X7[4]; xe71 = X7[5]; ys71 = X7[6]; ye71 = X7[7]; xs72 = X7[8]; xe72 = X7[9]; ys72 = X7[10]; ye72 = X7[11]
    xs81 = X8[4]; xe81 = X8[5]; ys81 = X8[6]; ye81 = X8[7]; xs82 = X8[8]; xe82 = X8[9]; ys82 = X8[10]; ye82 = X8[11]
    xs91 = X9[4]; xe91 = X9[5]; ys91 = X9[6]; ye91 = X9[7]; xs92 = X9[8]; xe92 = X9[9]; ys92 = X9[10]; ye92 = X9[11]

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
    ax.add_patch(Rectangle((x[xs10-1], y[ys10-1]), x[xe10-xs10+1], y[ye10-ys10+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs11-1], y[ys11-1]), x[xe11-xs11+1], y[ye11-ys11+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs21-1], y[ys21-1]), x[xe21-xs21+1], y[ye21-ys21+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs31-1], y[ys31-1]), x[xe31-xs31+1], y[ye31-ys31+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs41-1], y[ys41-1]), x[xe41-xs41+1], y[ye41-ys41+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs51-1], y[ys51-1]), x[xe51-xs51+1], y[ye51-ys51+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs61-1], y[ys61-1]), x[xe61-xs61+1], y[ye61-ys61+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs71-1], y[ys71-1]), x[xe71-xs71+1], y[ye71-ys71+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs81-1], y[ys81-1]), x[xe81-xs81+1], y[ye81-ys81+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs91-1], y[ys91-1]), x[xe91-xs91+1], y[ye91-ys91+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs12-1], y[ys12-1]), x[xe12-xs12+1], y[ye12-ys12+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs22-1], y[ys22-1]), x[xe22-xs22+1], y[ye22-ys22+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs32-1], y[ys32-1]), x[xe32-xs32+1], y[ye32-ys32+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs42-1], y[ys42-1]), x[xe42-xs42+1], y[ye42-ys42+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs52-1], y[ys52-1]), x[xe52-xs52+1], y[ye52-ys52+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs62-1], y[ys62-1]), x[xe62-xs62+1], y[ye62-ys62+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs72-1], y[ys72-1]), x[xe72-xs72+1], y[ye72-ys72+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs82-1], y[ys82-1]), x[xe82-xs82+1], y[ye82-ys82+1], facecolor = 'grey'))
    ax.add_patch(Rectangle((x[xs92-1], y[ys92-1]), x[xe92-xs92+1], y[ye92-ys92+1], facecolor = 'grey'))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(0,Lx)
    plt.ylim(0,Ly)



# Plotting the mean values of cB
def MeanCellsPlot(t_vec, cBmean1, cBmean2, cBmean3, cBmean4, cBmean5, cBmean6, cBmean7, cBmean8, cBmean9, cBmean10, Time, cBmax):
    t_vector = t_vec/60
    Time = Time/60
    plt.figure()
    plt.plot(t_vector,cBmean1, label = 'Plate 1')
    plt.plot(t_vector,cBmean2, label = 'Plate 2')
    plt.plot(t_vector,cBmean3, label = 'Plate 3')
    plt.plot(t_vector,cBmean4, label = 'Plate 4')
    plt.plot(t_vector,cBmean5, label = 'Plate 5')
    plt.plot(t_vector,cBmean6, label = 'Plate 6')
    plt.plot(t_vector,cBmean7, label = 'Plate 7')
    plt.plot(t_vector,cBmean8, label = 'Plate 8')
    plt.plot(t_vector,cBmean9, label = 'Plate 9')
    plt.plot(t_vector,cBmean10, label = 'Plate 10')
    plt.legend()
    plt.title('Mean concentration of cells on each plate [mol/m3]')
    plt.xlabel('Time [min]')
    plt.ylabel('Concentration [mol/m3]')
    plt.xlim(0,Time)
    plt.ylim(0,cBmax)