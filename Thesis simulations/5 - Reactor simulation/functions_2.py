import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from   matplotlib import cm
from   matplotlib.patches import Rectangle
from   numba import njit

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
def SpeciesBCs(phi, nx,ny, xs1,xe1,ys1,ye1, xs2,xe2,ys2,ye2, xs3,xe3,ys3,ye3, xs4,xe4,ys4,ye4, xs5,xe5,ys5,ye5, xs6,xe6,ys6,ye6, xs7,xe7,ys7,ye7, xs8,xe8,ys8,ye8, xs9,xe9,ys9,ye9):

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
def Graphical_obstacles(uu, vv, pp, ccO2, ccB, ccG, X1, X2, X3, X4, X5, X6, X7, X8, X9):

    xs1 = X1[0]; xe1 = X1[1]; ys1 = X1[2]; ye1 = X1[3]
    xs2 = X2[0]; xe2 = X2[1]; ys2 = X2[2]; ye2 = X2[3]
    xs3 = X3[0]; xe3 = X3[1]; ys3 = X3[2]; ye3 = X3[3]
    xs4 = X4[0]; xe4 = X4[1]; ys4 = X4[2]; ye4 = X4[3]
    xs5 = X5[0]; xe5 = X5[1]; ys5 = X5[2]; ye5 = X5[3]
    xs6 = X6[0]; xe6 = X6[1]; ys6 = X6[2]; ye6 = X6[3]
    xs7 = X7[0]; xe7 = X7[1]; ys7 = X7[2]; ye7 = X7[3]
    xs8 = X8[0]; xe8 = X8[1]; ys8 = X8[2]; ye8 = X8[3]
    xs9 = X9[0]; xe9 = X9[1]; ys9 = X9[2]; ye9 = X9[3]

    uu[xs1-1:xe1+1, ys1-1:ye1+1] = 0         # Obstacle 1
    vv[xs1-1:xe1+1, ys1-1:ye1+1] = 0
    pp[xs1-1:xe1+1, ys1-1:ye1+1] = 0
    ccO2[xs1-1:xe1+1, ys1-1:ye1+1] = 0
    ccB[xs1-1:xe1+1, ys1-1:ye1+1] = 0
    ccG[xs1-1:xe1+1, ys1-1:ye1+1] = 0

    uu[xs2-1:xe2+1, ys2-1:ye2+1] = 0         # Obstacle 2
    vv[xs2-1:xe2+1, ys2-1:ye2+1] = 0
    pp[xs2-1:xe2+1, ys2-1:ye2+1] = 0
    ccO2[xs2-1:xe2+1, ys2-1:ye2+1] = 0
    ccG[xs2-1:xe2+1, ys2-1:ye2+1] = 0

    uu[xs3-1:xe3+1, ys3-1:ye3+1] = 0         # Obstacle 3
    vv[xs3-1:xe3+1, ys3-1:ye3+1] = 0
    pp[xs3-1:xe3+1, ys3-1:ye3+1] = 0
    ccO2[xs3-1:xe3+1, ys3-1:ye3+1] = 0
    ccB[xs3-1:xe3+1, ys3-1:ye3+1] = 0
    ccG[xs3-1:xe3+1, ys3-1:ye3+1] = 0

    uu[xs4-1:xe4+1, ys4-1:ye4+1] = 0         # Obstacle 4
    vv[xs4-1:xe4+1, ys4-1:ye4+1] = 0
    pp[xs4-1:xe4+1, ys4-1:ye4+1] = 0
    ccO2[xs4-1:xe4+1, ys4-1:ye4+1] = 0
    ccB[xs4-1:xe4+1, ys4-1:ye4+1] = 0
    ccG[xs4-1:xe4+1, ys4-1:ye4+1] = 0

    uu[xs5-1:xe5+1, ys5-1:ye5+1] = 0         # Obstacle 5
    vv[xs5-1:xe5+1, ys5-1:ye5+1] = 0
    pp[xs5-1:xe5+1, ys5-1:ye5+1] = 0
    ccO2[xs5-1:xe5+1, ys5-1:ye5+1] = 0
    ccB[xs5-1:xe5+1, ys5-1:ye5+1] = 0
    ccG[xs5-1:xe5+1, ys5-1:ye5+1] = 0

    uu[xs6-1:xe6+1, ys6-1:ye6+1] = 0         # Obstacle 6
    vv[xs6-1:xe6+1, ys6-1:ye6+1] = 0
    pp[xs6-1:xe6+1, ys6-1:ye6+1] = 0
    ccO2[xs6-1:xe6+1, ys6-1:ye6+1] = 0
    ccB[xs6-1:xe6+1, ys6-1:ye6+1] = 0
    ccG[xs6-1:xe6+1, ys6-1:ye6+1] = 0

    uu[xs7-1:xe7+1, ys7-1:ye7+1] = 0         # Obstacle 7
    vv[xs7-1:xe7+1, ys7-1:ye7+1] = 0
    pp[xs7-1:xe7+1, ys7-1:ye7+1] = 0
    ccO2[xs7-1:xe7+1, ys7-1:ye7+1] = 0
    ccB[xs7-1:xe7+1, ys7-1:ye7+1] = 0
    ccG[xs7-1:xe7+1, ys7-1:ye7+1] = 0

    uu[xs8-1:xe8+1, ys8-1:ye8+1] = 0         # Obstacle 8
    vv[xs8-1:xe8+1, ys8-1:ye8+1] = 0
    pp[xs8-1:xe8+1, ys8-1:ye8+1] = 0
    ccO2[xs8-1:xe8+1, ys8-1:ye8+1] = 0
    ccB[xs8-1:xe8+1, ys8-1:ye8+1] = 0
    ccG[xs8-1:xe8+1, ys8-1:ye8+1] = 0

    uu[xs9-1:xe9+1, ys9-1:ye9+1] = 0         # Obstacle 9
    vv[xs9-1:xe9+1, ys9-1:ye9+1] = 0
    pp[xs9-1:xe9+1, ys9-1:ye9+1] = 0
    ccO2[xs9-1:xe9+1, ys9-1:ye9+1] = 0
    ccB[xs9-1:xe9+1, ys9-1:ye9+1] = 0
    ccG[xs9-1:xe9+1, ys9-1:ye9+1] = 0

    return uu, vv, pp, ccO2, ccB, ccG


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



# Plotting the results
def PlotFunctions(xx, yy, phi, x, y, X1, X2, X3, X4, X5, X6, X7, X8, X9, Lx, Ly, title, x_label, y_label):

    xs1 = X1[0]; xe1 = X1[1]; ys1 = X1[2]; ye1 = X1[3]
    xs2 = X2[0]; xe2 = X2[1]; ys2 = X2[2]; ye2 = X2[3]
    xs3 = X3[0]; xe3 = X3[1]; ys3 = X3[2]; ye3 = X3[3]
    xs4 = X4[0]; xe4 = X4[1]; ys4 = X4[2]; ye4 = X4[3]
    xs5 = X5[0]; xe5 = X5[1]; ys5 = X5[2]; ye5 = X5[3]
    xs6 = X6[0]; xe6 = X6[1]; ys6 = X6[2]; ye6 = X6[3]
    xs7 = X7[0]; xe7 = X7[1]; ys7 = X7[2]; ye7 = X7[3]
    xs8 = X8[0]; xe8 = X8[1]; ys8 = X8[2]; ye8 = X8[3]
    xs9 = X9[0]; xe9 = X9[1]; ys9 = X9[2]; ye9 = X9[3]

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
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(0,Lx)
    plt.ylim(0,Ly)