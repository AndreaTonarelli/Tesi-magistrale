import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from   matplotlib import cm
from   matplotlib.patches import Rectangle
from   numba import njit


# Gamma coefficient for Poisson equation
def gammaCoeff(gamma, hx,hy, nout_start,nout_end, X1, X2, X3, X4, X5, X6, X7, X8):

    xs1 = X1[0]; xe1 = X1[1]; ys1 = X1[2]; ye1 = X1[3]
    xs2 = X2[0]; xe2 = X2[1]; ys2 = X2[2]; ye2 = X2[3]
    xs3 = X3[0]; xe3 = X3[1]; ys3 = X3[2]; ye3 = X3[3]
    xs4 = X4[0]; xe4 = X4[1]; ys4 = X4[2]; ye4 = X4[3]
    xs5 = X5[0]; xe5 = X5[1]; ys5 = X5[2]; ye5 = X5[3]
    xs6 = X6[0]; xe6 = X6[1]; ys6 = X6[2]; ye6 = X6[3]
    xs7 = X7[0]; xe7 = X7[1]; ys7 = X7[2]; ye7 = X7[3]
    xs8 = X8[0]; xe8 = X8[1]; ys8 = X8[2]; ye8 = X8[3]

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

    # Correction of gamma close to the obstacles edges
    gamma[xs1, ye2+1] = hx*hy / (hx**2 + hy**2)          # Obstacle 1
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
def VelocityBCs(u, v, uwall, X1, X2, X3, X4, X5, X6, X7, X8):

    xs1 = X1[0]; xe1 = X1[1]; ys1 = X1[2]; ye1 = X1[3]
    xs2 = X2[0]; xe2 = X2[1]; ys2 = X2[2]; ye2 = X2[3]
    xs3 = X3[0]; xe3 = X3[1]; ys3 = X3[2]; ye3 = X3[3]
    xs4 = X4[0]; xe4 = X4[1]; ys4 = X4[2]; ye4 = X4[3]
    xs5 = X5[0]; xe5 = X5[1]; ys5 = X5[2]; ye5 = X5[3]
    xs6 = X6[0]; xe6 = X6[1]; ys6 = X6[2]; ye6 = X6[3]
    xs7 = X7[0]; xe7 = X7[1]; ys7 = X7[2]; ye7 = X7[3]
    xs8 = X8[0]; xe8 = X8[1]; ys8 = X8[2]; ye8 = X8[3]

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
def SpeciesBCs(phi, nx,ny, X1, X2, X3, X4, X5, X6, X7, X8):

    xs1 = X1[0]; xe1 = X1[1]; ys1 = X1[2]; ye1 = X1[3]
    xs2 = X2[0]; xe2 = X2[1]; ys2 = X2[2]; ye2 = X2[3]
    xs3 = X3[0]; xe3 = X3[1]; ys3 = X3[2]; ye3 = X3[3]
    xs4 = X4[0]; xe4 = X4[1]; ys4 = X4[2]; ye4 = X4[3]
    xs5 = X5[0]; xe5 = X5[1]; ys5 = X5[2]; ye5 = X5[3]
    xs6 = X6[0]; xe6 = X6[1]; ys6 = X6[2]; ye6 = X6[3]
    xs7 = X7[0]; xe7 = X7[1]; ys7 = X7[2]; ye7 = X7[3]
    xs8 = X8[0]; xe8 = X8[1]; ys8 = X8[2]; ye8 = X8[3]

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
def Graphical_obstacles(uu, vv, pp, tau, ccO2, ccB, ccG, X1, X2, X3, X4, X5, X6, X7, X8):

    xs1 = X1[0]; xe1 = X1[1]; ys1 = X1[2]; ye1 = X1[3]
    xs2 = X2[0]; xe2 = X2[1]; ys2 = X2[2]; ye2 = X2[3]
    xs3 = X3[0]; xe3 = X3[1]; ys3 = X3[2]; ye3 = X3[3]
    xs4 = X4[0]; xe4 = X4[1]; ys4 = X4[2]; ye4 = X4[3]
    xs5 = X5[0]; xe5 = X5[1]; ys5 = X5[2]; ye5 = X5[3]
    xs6 = X6[0]; xe6 = X6[1]; ys6 = X6[2]; ye6 = X6[3]
    xs7 = X7[0]; xe7 = X7[1]; ys7 = X7[2]; ye7 = X7[3]
    xs8 = X8[0]; xe8 = X8[1]; ys8 = X8[2]; ye8 = X8[3]

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
def CellsMeanConcentration(cB, cB0, th, X):

    xs = X[0]; xe = X[1]; ys = X[2]; ye = X[3]

    cB[xe+1:xe+th+1, ys:ye] = cB0    # Obstacle n
    cBmean = 0

    for i in range(xe+1, xe+th+1):
        for j in range(ys, ye+1):

            cBmean_o = cBmean
            cBmean = cBmean_o + cB[i,j]

    cBmean = cBmean / ((ye - ys) * th)

    return cBmean



# Plotting the results
def PlotFunctions(xx, yy, phi, x, y, X1, X2, X3, X4, X5, X6, X7, X8, Lx, Ly, title, x_label, y_label):

    xs1 = X1[0]; xe1 = X1[1]; ys1 = X1[2]; ye1 = X1[3]
    xs2 = X2[0]; xe2 = X2[1]; ys2 = X2[2]; ye2 = X2[3]
    xs3 = X3[0]; xe3 = X3[1]; ys3 = X3[2]; ye3 = X3[3]
    xs4 = X4[0]; xe4 = X4[1]; ys4 = X4[2]; ye4 = X4[3]
    xs5 = X5[0]; xe5 = X5[1]; ys5 = X5[2]; ye5 = X5[3]
    xs6 = X6[0]; xe6 = X6[1]; ys6 = X6[2]; ye6 = X6[3]
    xs7 = X7[0]; xe7 = X7[1]; ys7 = X7[2]; ye7 = X7[3]
    xs8 = X8[0]; xe8 = X8[1]; ys8 = X8[2]; ye8 = X8[3]

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
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(0,Lx)
    plt.ylim(0,Ly)



# Plotting the streamlines
def Streamlines(x, y, xx, yy, uu, vv, X1, X2, X3, X4, X5, X6, X7, X8, Lx, Ly, title, x_label, y_label):

    xs1 = X1[0]; xe1 = X1[1]; ys1 = X1[2]; ye1 = X1[3]
    xs2 = X2[0]; xe2 = X2[1]; ys2 = X2[2]; ye2 = X2[3]
    xs3 = X3[0]; xe3 = X3[1]; ys3 = X3[2]; ye3 = X3[3]
    xs4 = X4[0]; xe4 = X4[1]; ys4 = X4[2]; ye4 = X4[3]
    xs5 = X5[0]; xe5 = X5[1]; ys5 = X5[2]; ye5 = X5[3]
    xs6 = X6[0]; xe6 = X6[1]; ys6 = X6[2]; ye6 = X6[3]
    xs7 = X7[0]; xe7 = X7[1]; ys7 = X7[2]; ye7 = X7[3]
    xs8 = X8[0]; xe8 = X8[1]; ys8 = X8[2]; ye8 = X8[3]

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
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(0,Lx)
    plt.ylim(0,Ly)



# Plotting the mean values of cB
def MeanCellsPlot(t_vector, cBmean1, cBmean2, cBmean3, cBmean4, cBmean5, cBmean6, cBmean7, Time, cBmax):
    plt.figure()
    plt.plot(t_vector,cBmean1, label = 'Plate 1')
    plt.plot(t_vector,cBmean2, label = 'Plate 2')
    plt.plot(t_vector,cBmean3, label = 'Plate 3')
    plt.plot(t_vector,cBmean4, label = 'Plate 4')
    plt.plot(t_vector,cBmean5, label = 'Plate 5')
    plt.plot(t_vector,cBmean6, label = 'Plate 6')
    plt.plot(t_vector,cBmean7, label = 'Plate 7')
    plt.legend()
    plt.title('Mean concentration of cells on each plate [mol/m3]')
    plt.xlabel('Time [s]')
    plt.ylabel('Concentration [mol/m3]')
    plt.xlim(0,Time)
    plt.ylim(0,cBmax)