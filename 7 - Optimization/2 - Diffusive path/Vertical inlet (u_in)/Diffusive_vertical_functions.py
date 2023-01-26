import numpy as np
import matplotlib.pyplot as plt
from   matplotlib.patches import Rectangle
from   numba import njit


# Gamma coefficient for Poisson equation
def gammaCoeff(gamma, hx,hy, nout_start,nout_end, XX):

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
    gamma[-2, nout_start:nout_end]   = hx*hy / (2*hx**2 + 2*hy**2)    # Outlet section is treated as an internal cell
    gamma[-2, nout_start]            = hx*hy / (  hx**2 + 2*hy**2)    # Corners of outlet sections
    gamma[-2, nout_end]              = hx*hy / (  hx**2 + 2*hy**2)

    # Correction of gamma close to the obstacles
    @njit
    def gamma_edges_obst(gamma, X, hx, hy):

        for i in range(len(X) // 4):
            xs = X[4*i]; xe = X[4*i+1]; ys = X[4*i+2]; ye = X[4*i+3]
            gamma[xs-1, ys:ye+1] = hx*hy / (2*hx**2 +   hy**2)
            gamma[xe+1, ys:ye+1] = hx*hy / (2*hx**2 +   hy**2)
            gamma[xs:xe+1, ys-1] = hx*hy / (  hx**2 + 2*hy**2)
            gamma[xs:xe+1, ye+1] = hx*hy / (  hx**2 + 2*hy**2)
        
        return gamma

    for i in range(len(XX)):
        gamma = gamma_edges_obst(gamma, XX[i], hx, hy)  # Obstacle i+1 (including ones near corners!!)

    # Correction of gamma close to the obstacles edges
    @njit
    def gamma_corners_obst(gamma, X, hx, hy, position):

        if position == 'entrance':
            xs = X[0]; ys = X[2]; ye = X[3]; xe_last = X[-3]
            gamma[xs, ye+1] = hx*hy / (hx**2 + hy**2)
            gamma[xe_last+1, ys] = hx*hy / (hx**2 + hy**2)
            for i in range((len(X) // 4)-1):
                xe1 = X[4*i+1]; ye2 = X[4*i+7]
                gamma[xe1+1, ye2+1] = hx*hy / (hx**2 + hy**2)
            return gamma
        elif position == 'south':
            xs = X[0]; ys = X[2]; xe_last = X[-3]
            gamma[xs-1, ys] = hx*hy / (hx**2 + hy**2)
            gamma[xe_last+1, ys] = hx*hy / (hx**2 + hy**2)
            for i in range((len(X) // 4)-1):
                xe1 = X[4*i+1]; ye2 = X[4*i+7]
                gamma[xe1+1, ye2+1] = hx*hy / (hx**2 + hy**2)
            return gamma
        elif position == 'north':
            xs = X[0]; ye = X[3]; xe_last = X[-3]
            gamma[xs-1, ye] = hx*hy / (hx**2 + hy**2)          # Obstacle 2
            gamma[xe_last+1, ye] = hx*hy / (hx**2 + hy**2)
            for i in range((len(X) // 4)-1):
                xe1 = X[4*i+1]; ys2 = X[4*i+6]
                gamma[xe1+1, ys2-1] = hx*hy / (hx**2 + hy**2)
            return gamma
            
    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15 = XX[:]
    xs1 = X1[0]; ys1 = X1[2]; ye1 = X1[3]
    gamma[xs1, ys1-1] = hx*hy / (hx**2 + hy**2)                 # Obstacle 1
    gamma[xs1, ye1+1] = hx*hy / (hx**2 + hy**2)
    gamma = gamma_corners_obst(gamma, X2, hx, hy, 'south')      # Obstacle 2
    gamma = gamma_corners_obst(gamma, X3, hx, hy, 'north')      # Obstacle 3
    gamma = gamma_corners_obst(gamma, X5, hx, hy, 'south')      # Obstacle 5
    gamma = gamma_corners_obst(gamma, X6, hx, hy, 'north')      # Obstacle 6
    gamma = gamma_corners_obst(gamma, X8, hx, hy, 'south')      # Obstacle 8
    gamma = gamma_corners_obst(gamma, X9, hx, hy, 'north')      # Obstacle 9
    gamma = gamma_corners_obst(gamma, X11, hx, hy, 'south')     # Obstacle 11
    gamma = gamma_corners_obst(gamma, X12, hx, hy, 'north')     # Obstacle 12
    xs14 = X14[0]; ys14 = X14[2]
    gamma[xs14-1, ys14] = hx*hy / (hx**2 + hy**2)               # Obstacle 14
    xs15 = X15[0]; ye15 = X15[3]
    gamma[xs15-1, ye15] = hx*hy / (hx**2 + hy**2)               # Obstacle 15

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



# Boundary conditions for velocities
def VelocityBCs(u, v, uwall, XX):

    '''
    @njit
    def velocities_walls(u, v, X):

        for i in range(len(X) // 4):
            xs = X[4*i]; xe = X[4*i+1]; ys = X[4*i+2]; ye = X[4*i+3]
            u[xs-1:xe+1, ye] = 2*uwall - u[xs-1:xe+1, ye+1]    # north face
            u[xs-1:xe+1, ys] = 2*uwall - u[xs-1:xe+1, ys-1]    # south face
            v[xs, ys-1:ye+1] = 2*uwall - v[xs-1, ys-1:ye+1]    # west  face
            v[xe, ys-1:ye+1] = 2*uwall - v[xe+1, ys-1:ye+1]    # east  face
        
        return u, v

    u, v = velocities_walls(u, v, X1)     # Obstacle 1
    u, v = velocities_walls(u, v, X2)     # Obstacle 2
    u, v = velocities_walls(u, v, X3)     # Obstacle 3
    u, v = velocities_walls(u, v, X4)     # Obstacle 4
    u, v = velocities_walls(u, v, X5)     # Obstacle 5
    u, v = velocities_walls(u, v, X6)     # Obstacle 6
    u, v = velocities_walls(u, v, X7)     # Obstacle 7
    u, v = velocities_walls(u, v, X8)     # Obstacle 8
    u, v = velocities_walls(u, v, X9)     # Obstacle 9
    u, v = velocities_walls(u, v, X10)     # Obstacle 10
    '''

    def vel_walls(u, v, xs, xe, ys, ye):
        u[xs:xe+1, ye] = 2*uwall - u[xs:xe+1, ye+1]    # north face
        u[xs:xe+1, ys] = 2*uwall - u[xs:xe+1, ys-1]    # south face
        v[xs, ys:ye+1] = 2*uwall - v[xs-1, ys:ye+1]    # west  face
        v[xe, ys:ye+1] = 2*uwall - v[xe+1, ys:ye+1]    # east  face
        return u, v

    
    def vel_walls_function(u, v, X):
        n = (len(X) // 4) - 1     # n = number of sub-obstacles
        u, v = vel_walls(u, v, X[0],  X[1],  X[2],  X[3])         # Obstacle n+1
        if n > 0:
            u, v = vel_walls(u, v, X[4],  X[5],  X[6],  X[7])     # near Obstacle n+1
        if n > 1:
            u, v = vel_walls(u, v, X[8],  X[9],  X[10], X[11])
        if n > 2:
            u, v = vel_walls(u, v, X[12], X[13], X[14], X[15])
        if n > 3:
            u, v = vel_walls(u, v, X[16], X[17], X[18], X[19])
        if n > 4:
            u, v = vel_walls(u, v, X[20], X[21], X[22], X[23])
        if n > 5:
            u, v = vel_walls(u, v, X[24], X[25], X[26], X[27])
        if n > 6:
            u, v = vel_walls(u, v, X[28], X[29], X[30], X[31])
        if n > 7:
            u, v = vel_walls(u, v, X[32], X[33], X[34], X[35])
        if n > 8:
            u, v = vel_walls(u, v, X[36], X[37], X[38], X[39])
        if n > 9:
            u, v = vel_walls(u, v, X[40], X[41], X[42], X[43])
        if n > 10:
            u, v = vel_walls(u, v, X[44], X[45], X[46], X[47])
        if n > 11:
            u, v = vel_walls(u, v, X[48], X[49], X[50], X[51])
        if n > 12:
            u, v = vel_walls(u, v, X[52], X[53], X[54], X[55])
        if n > 13:
            u, v = vel_walls(u, v, X[56], X[57], X[58], X[59])
        if n > 14:
            u, v = vel_walls(u, v, X[60], X[61], X[62], X[63])
        if n > 15:
            u, v = vel_walls(u, v, X[64], X[65], X[66], X[67])
        if n > 16:
            u, v = vel_walls(u, v, X[68], X[69], X[70], X[71])
        if n > 17:
            u, v = vel_walls(u, v, X[72], X[73], X[74], X[75])
        if n > 18:
            u, v = vel_walls(u, v, X[76], X[77], X[78], X[79])
        if n > 19:
            u, v = vel_walls(u, v, X[80], X[81], X[82], X[83])
        if n > 20:
            u, v = vel_walls(u, v, X[84], X[85], X[86], X[87])
        if n > 21:
            u, v = vel_walls(u, v, X[88], X[89], X[90], X[91])
        if n > 22:
            u, v = vel_walls(u, v, X[92], X[93], X[94], X[95])
        if n > 23:
            u, v = vel_walls(u, v, X[96], X[97], X[98], X[99])
        if n > 24:
            u, v = vel_walls(u, v, X[100], X[101], X[102], X[103])
        if n > 25:
            u, v = vel_walls(u, v, X[104], X[105], X[106], X[107])
        if n > 26:
            u, v = vel_walls(u, v, X[108], X[109], X[110], X[111])
        if n > 27:
            u, v = vel_walls(u, v, X[112], X[113], X[114], X[115])
        if n > 28:
            u, v = vel_walls(u, v, X[116], X[117], X[118], X[119])
        if n > 29:
            u, v = vel_walls(u, v, X[120], X[121], X[122], X[123])
        if n > 30:
            u, v = vel_walls(u, v, X[124], X[125], X[126], X[127])
        return u, v

    for i in range(len(XX)):
        u, v = vel_walls_function(u, v, XX[i])     # Obstacle i+1

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

                phi[i, j] = phio[i, j] + dt/(hx*hy) * (-A + D)

    return phi



# Boundary conditions for species
def SpeciesBCs(phi, XX):

    # Entire domain BCs (Dirichlet)
    phi[1:-1, 0] = phi[1:-1, 1];           # South wall
    phi[1:-1, -1] = phi[1:-1, -2];         # North wall
    phi[0, 1:-1] = phi[1, 1:-1];           # West wall
    phi[-1, 1:-1] = phi[-2, 1:-1];         # East wall

    '''
    @njit
    def obstacles_BCS(phi, X):

        for i in range(len(X) // 4):
            xs = X[4*i]; xe = X[4*i+1]; ys = X[4*i+2]; ye = X[4*i+3]
            phi[xs:xe+1, ys] = phi[xs:xe+1, ys-1];     # South wall
            phi[xs:xe+1, ye] = phi[xs:xe+1, ye+1];     # North wall
            phi[xs, ys:ye+1] = phi[xs-1, ys:ye+1];     # West wall
            phi[xe, ys:ye+1] = phi[xe+1, ys:ye+1];     # East wall

        return phi
    
    phi = obstacles_BCS(phi, X1)    # 1st Obstacle BCS (Dirichlet)  (including ones near corners!!)
    phi = obstacles_BCS(phi, X2)    # 2nd Obstacle BCS (Dirichlet)
    phi = obstacles_BCS(phi, X3)    # 3rd Obstacle BCS (Dirichlet)
    phi = obstacles_BCS(phi, X4)    # 4th Obstacle BCS (Dirichlet)
    phi = obstacles_BCS(phi, X5)    # 5th Obstacle BCS (Dirichlet)
    phi = obstacles_BCS(phi, X6)    # 6th Obstacle BCS (Dirichlet)
    phi = obstacles_BCS(phi, X7)    # 7th Obstacle BCS (Dirichlet)
    phi = obstacles_BCS(phi, X8)    # 8th Obstacle BCS (Dirichlet)
    phi = obstacles_BCS(phi, X9)    # 9th Obstacle BCS (Dirichlet)
    phi = obstacles_BCS(phi, X10)    # 10th Obstacle BCS (Dirichlet)
    '''

    def phi_obstacle(phi, xs, xe, ys, ye):
        phi[xs:xe+1, ys] = phi[xs:xe+1, ys-1];     # South wall
        phi[xs:xe+1, ye] = phi[xs:xe+1, ye+1];     # North wall
        phi[xs, ys:ye+1] = phi[xs-1, ys:ye+1];     # West wall
        phi[xe, ys:ye+1] = phi[xe+1, ys:ye+1];     # East wall
        return phi

    
    def phi_obstacle_function(phi, X):
        n = (len(X) // 4) - 1      # n = number of sub-obstacles
        phi = phi_obstacle(phi, X[0],  X[1],  X[2],  X[3])      # Obstacle n+1
        if n > 0:
            phi = phi_obstacle(phi, X[4],  X[5],  X[6],  X[7])      # near Obstacle n+1
        if n > 1:
            phi = phi_obstacle(phi, X[8],  X[9],  X[10], X[11])
        if n > 2:
            phi = phi_obstacle(phi, X[12], X[13], X[14], X[15])
        if n > 3:
            phi = phi_obstacle(phi, X[16], X[17], X[18], X[19])
        if n > 4:
            phi = phi_obstacle(phi, X[20], X[21], X[22], X[23])
        if n > 5:
            phi = phi_obstacle(phi, X[24], X[25], X[26], X[27])
        if n > 6:
            phi = phi_obstacle(phi, X[28], X[29], X[30], X[31])
        if n > 7:
            phi = phi_obstacle(phi, X[32], X[33], X[34], X[35])
        if n > 8:
            phi = phi_obstacle(phi, X[36], X[37], X[38], X[39])
        if n > 9:
            phi = phi_obstacle(phi, X[40], X[41], X[42], X[43])
        if n > 10:
            phi = phi_obstacle(phi, X[44], X[45], X[46], X[47])
        if n > 11:
            phi = phi_obstacle(phi, X[48], X[49], X[50], X[51])
        if n > 12:
            phi = phi_obstacle(phi, X[52], X[53], X[54], X[55])
        if n > 13:
            phi = phi_obstacle(phi, X[56], X[57], X[58], X[59])
        if n > 14:
            phi = phi_obstacle(phi, X[60], X[61], X[62], X[63])
        if n > 15:
            phi = phi_obstacle(phi, X[64], X[65], X[66], X[67])
        if n > 16:
            phi = phi_obstacle(phi, X[68], X[69], X[70], X[71])
        if n > 17:
            phi = phi_obstacle(phi, X[72], X[73], X[74], X[75])
        if n > 18:
            phi = phi_obstacle(phi, X[76], X[77], X[78], X[79])
        if n > 19:
            phi = phi_obstacle(phi, X[80], X[81], X[82], X[83])
        if n > 20:
            phi = phi_obstacle(phi, X[84], X[85], X[86], X[87])
        if n > 21:
            phi = phi_obstacle(phi, X[88], X[89], X[90], X[91])
        if n > 22:
            phi = phi_obstacle(phi, X[92], X[93], X[94], X[95])
        if n > 23:
            phi = phi_obstacle(phi, X[96], X[97], X[98], X[99])
        if n > 24:
            phi = phi_obstacle(phi, X[100], X[101], X[102], X[103])
        if n > 25:
            phi = phi_obstacle(phi, X[104], X[105], X[106], X[107])
        if n > 26:
            phi = phi_obstacle(phi, X[108], X[109], X[110], X[111])
        if n > 27:
            phi = phi_obstacle(phi, X[112], X[113], X[114], X[115])
        if n > 28:
            phi = phi_obstacle(phi, X[116], X[117], X[118], X[119])
        if n > 29:
            phi = phi_obstacle(phi, X[120], X[121], X[122], X[123])
        if n > 30:
            phi = phi_obstacle(phi, X[124], X[125], X[126], X[127])
        return phi

    for i in range(len(XX)):
        phi = phi_obstacle_function(phi, XX[i])       # i+1 Obstacle BCS (Dirichlet)

    return phi



# Reaction step after Advection-Diffusion of species
@njit
def ReactionStep(cO2star, Xvstar, cGlcstar, cGlnstar, cLacstar, cAmmstar, cB1star, dt, parameters, nx, ny):

    cO2 = cO2star; Xv = Xvstar; cGlc = cGlcstar
    cGln = cGlnstar; cLac = cLacstar; cAmm = cAmmstar; cB1 = cB1star

    mu_max, k_d, K_Glc, K_Gln, KI_Amm, KI_Lac, KD_Amm, KD_Lac, Y_Glc, Y_Gln, Y_Lac, Y_Amm, Q_B1, q_O2 = parameters

    for i in range(1, nx+1):
        for j in range(1, ny+1):

            if cLacstar[i,j] <= 4.6e-3 and cAmmstar[i,j] <= 52e-3:
                mu = mu_max * cGlcstar[i,j]/(K_Glc+cGlcstar[i,j]) * cGlnstar[i,j]/(K_Gln+cGlnstar[i,j])
                mu_d = 0
            elif cLacstar[i,j] > 4.6e-3 and cAmmstar[i,j] <= 52e-3:
                mu = mu_max * cGlcstar[i,j]/(K_Glc+cGlcstar[i,j]) * cGlnstar[i,j]/(K_Gln+cGlnstar[i,j]) * KI_Lac/(KI_Lac+cLacstar[i,j])
                mu_d = 0
            elif cLacstar[i,j] <= 4.6e-3 and cAmmstar[i,j] > 52e-3:
                mu = mu_max * cGlcstar[i,j]/(K_Glc+cGlcstar[i,j]) * cGlnstar[i,j]/(K_Gln+cGlnstar[i,j]) * KI_Amm/(KI_Amm+cAmmstar[i,j])
                mu_d = 0
            else:
                mu = mu_max * cGlcstar[i,j]/(K_Glc+cGlcstar[i,j]) * cGlnstar[i,j]/(K_Gln+cGlnstar[i,j]) * KI_Amm/(KI_Amm+cAmmstar[i,j]) * KI_Lac/(KI_Lac+cLacstar[i,j])
                mu_d = k_d * cLacstar[i,j]/(KD_Lac+cLacstar[i,j]) * cAmmstar[i,j]/(KD_Amm+cAmmstar[i,j])

            Xv[i,j] = Xvstar[i,j] * (1 + dt*(mu-mu_d))
            cGlc[i,j] = cGlcstar[i,j] - dt*(mu-mu_d)/Y_Glc*Xv[i,j]
            cGln[i,j] = cGlnstar[i,j] - dt*(mu-mu_d)/Y_Gln*Xv[i,j]
            cLac[i,j] = cLacstar[i,j] + dt*(mu-mu_d)/Y_Glc*Y_Lac*Xv[i,j]
            cAmm[i,j] = cAmmstar[i,j] + dt*(mu-mu_d)/Y_Gln*Y_Amm*Xv[i,j]
            cO2[i,j]  = cO2star[i,j] - dt*q_O2*Xv[i,j]*cO2star[i,j]/(cO2star[i,j]+1e-12)
            cB1[i,j]  = cB1star[i,j] * (1 + dt*Q_B1*Xv[i,j]*(1 - mu/mu_max) )

    return cO2, Xv, cGlc, cGln, cLac, cAmm, cB1



# Species concentration on inlet sections
def InletConcentration(phi, phi_in, nin_start, nin_end, position):

    if position == 'west':
        phi[0, nin_start:nin_end+1] = 2*phi_in - phi[1, nin_start:nin_end+1]
    if position == 'north':
        phi[nin_start:nin_end+1, -1] = 2*phi_in - phi[nin_start:nin_end+1, -2]
    if position == 'south':
        phi[nin_start:nin_end+1,  0] = 2*phi_in - phi[nin_start:nin_end+1,  1]
    
    return phi



# Immobilized cells initialized over obstacles
def ImmobilizedCells(Xv, Xv0, X, th, position):
    xs = X[0]; xe = X[1]; ys = X[2]; ye = X[3]

    if position == 'left':
        Xv[xs-th:xs, ys:ye] = Xv0
    elif position == 'right':
        Xv[xe:xe+th, ys:ye] = Xv0

    return Xv



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
def Graphical_obstacles(uu, vv, pp, tau, ccO2, ccXv, ccGlc, ccGln, ccLac, ccAmm, ccB1, XX):
    
    @njit
    def graphical(uu, vv, pp, tau, ccO2, ccXv, ccGlc, ccGln, ccLac, ccAmm, ccB1, X):

        for i in range(len(X) // 4):
            xs = X[4*i]; xe = X[4*i+1]; ys = X[4*i+2]; ye = X[4*i+3]
            uu[xs-1:xe+1, ys-1:ye+1] = 0
            vv[xs-1:xe+1, ys-1:ye+1] = 0
            pp[xs-1:xe+1, ys-1:ye+1] = 0
            tau[xs-1:xe+1, ys-1:ye+1] = 0
            ccO2[xs-1:xe+1, ys-1:ye+1] = 0
            ccXv[xs-1:xe+1, ys-1:ye+1] = 0
            ccGlc[xs-1:xe+1, ys-1:ye+1] = 0
            ccGln[xs-1:xe+1, ys-1:ye+1] = 0
            ccLac[xs-1:xe+1, ys-1:ye+1] = 0
            ccAmm[xs-1:xe+1, ys-1:ye+1] = 0
            ccB1[xs-1:xe+1, ys-1:ye+1] = 0

        return uu, vv, pp, tau, ccO2, ccXv, ccGlc, ccGln, ccLac, ccAmm, ccB1

    for i in range(len(XX)):
        uu, vv, pp, tau, ccO2, ccXv, ccGlc, ccGln, ccLac, ccAmm, ccB1 = graphical(uu, vv, pp, tau, ccO2, ccXv, ccGlc, ccGln, ccLac, ccAmm, ccB1, XX[i])

    return uu, vv, pp, tau, ccO2, ccXv, ccGlc, ccGln, ccLac, ccAmm, ccB1


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
def CellsMeanConcentration(cB, th, X, position):

    xs = X[0]; xe = X[1]; ys = X[2]; ye = X[3]

    cBmean = 0

    if position == 'left':
        for i in range(xs-th, xs):     # (pay attention if cells are above or under the plate!!!)
            for j in range(ys, ye+1):

                cBmean_o = cBmean
                cBmean = cBmean_o + cB[i,j]
    if position == 'right':
        for i in range(xe, xe+th):     # (pay attention if cells are above or under the plate!!!)
            for j in range(ys, ye+1):

                cBmean_o = cBmean
                cBmean = cBmean_o + cB[i,j]

    cBmean = cBmean / ((ye - ys) * th)

    return cBmean



# Interpolation of cells concentration needed for graphical purposes
def Interpolation_of_cells(ccXv, Xv, X, th, position):
    # Position = if cells are on the left or on the right of the plates
    xs = X[0]; xe = X[1]; ys = X[2]; ye = X[3]
    if position == 'left':
        ccXv[xs-th:xs, ys:ye] = Xv[xs-th:xs, ys+1:ye+1]
    elif position == 'right':
        ccXv[xe:xe+th, ys:ye] = Xv[xe:xe+th, ys+1:ye+1]
    return ccXv



# Collecting the outlet B1 concentration
@njit
def B1OutletConcentration(cB1, nout_start, nout_end):

    cB1out = 0

    for j in range(nout_start, nout_end+1):

        cB1out_o = cB1out
        cB1out = cB1out_o + cB1[-1, j]

    return cB1out



# Plotting the results
def PlotFunctions(xx, yy, phi, x, y, XX, Lx, Ly, title, x_label, y_label):

    def add_obst_patch(X):

        for i in range(len(X) // 4):
            xs = X[4*i]; xe = X[4*i+1]; ys = X[4*i+2]; ye = X[4*i+3]
            ax.add_patch(Rectangle((x[xs-1]/Lx, y[ys-1]/Ly), x[xe-xs+1]/Lx, y[ye-ys+1]/Ly, facecolor = 'grey'))

    fig, ax = plt.subplots()
    plot = plt.contourf(xx/Lx, yy/Ly, np.transpose(phi), 400, vmin = np.min(phi), vmax = np.max(phi), origin = 'lower')
    plt.colorbar(plot, ax = ax)
    for i in range(len(XX)):
        add_obst_patch(XX[i])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)



# Plotting the streamlines
def Streamlines(x, y, xx, yy, uu, vv, XX, Lx, Ly, title, x_label, y_label):

    def add_obst_patch(X):

        for i in range(len(X) // 4):
            xs = X[4*i]; xe = X[4*i+1]; ys = X[4*i+2]; ye = X[4*i+3]
            ax.add_patch(Rectangle((x[xs-1]/Lx, y[ys-1]/Ly), x[xe-xs+1]/Lx, y[ye-ys+1]/Ly, facecolor = 'grey'))

    fig, ax = plt.subplots()
    # plt.quiver(xx[::10], yy[::10], np.transpose(uu)[::10], np.transpose(vv)[::10])
    plt.streamplot(xx/Lx, yy/Ly, np.transpose(uu), np.transpose(vv), linewidth = 1)
    for i in range(len(XX)):
        add_obst_patch(XX[i])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(0,1)
    plt.ylim(0,1)



# Plotting the mean values of cB
def MeanCellsPlot(t_vec, cBmean, Time, cB0):
    t_vector = t_vec/60
    Time = Time/60
    cBmean2, cBmean3, cBmean4, cBmean5, cBmean6, cBmean7, cBmean8, cBmean9, cBmean10, cBmean11, cBmean12, cBmean13, cBmean14, cBmean15 = cBmean
    cBmax = 0
    plt.figure()
    plt.plot(t_vector,cBmean2, label = 'Plate 2')
    plt.plot(t_vector,cBmean3, label = 'Plate 3')
    plt.plot(t_vector,cBmean5, label = 'Plate 5')
    plt.plot(t_vector,cBmean6, label = 'Plate 6')
    plt.plot(t_vector,cBmean8, label = 'Plate 8')
    plt.plot(t_vector,cBmean9, label = 'Plate 9')
    plt.plot(t_vector,cBmean11, label = 'Plate 11')
    plt.plot(t_vector,cBmean12, label = 'Plate 12')
    plt.plot(t_vector,cBmean14, label = 'Plate 14')
    plt.plot(t_vector,cBmean15, label = 'Plate 15')
    for i in range(len(cBmean)):
        cBmax = max(cBmax, np.max(cBmean[i]))
    plt.legend()
    plt.title('Mean number of cells on each plate [cells]')
    plt.xlabel('Time [min]')
    plt.ylabel('Number of cells [cells]')
    plt.xlim(0, Time)
    plt.ylim(cB0, cBmax)



# Plotting the outlet concentration of B1
def PlotinTime(t_vec, phi, Time, title, y_label):
    t_vector = t_vec/60
    Time = Time/60
    plt.figure()
    plt.plot(t_vector, phi)
    plt.title(title)
    plt.xlabel('Time [min]')
    plt.ylabel(y_label)
    plt.xlim(0,Time)
    plt.ylim(0,np.max(phi)*1.1)