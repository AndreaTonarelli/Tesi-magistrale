import numpy as np
from   numba import njit
from   numba.typed import List

# Defining obstacles coordinates (semi-automatically!!!)
def Obstacles(x, y, Lx, Ly):

    # Obstacle coordinates
    # Obstacle 1
    x_obst1_start = Lx/10 - 0.1
    x_obst1_end = Lx/10
    y_obst1_start = 0.
    y_obst1_end = Ly - Lx/10
    # Obstacle 2
    x_obst2_start = 2*Lx/10 - 0.1
    x_obst2_end = 2*Lx/10
    y_obst2_start = Lx/10
    y_obst2_end = Ly
    # Obstacle 3
    x_obst3_start = 3*Lx/10 - 0.1
    x_obst3_end = 3*Lx/10
    y_obst3_start = 0.
    y_obst3_end = Ly - Lx/10
    # Obstacle 4
    x_obst4_start = 4*Lx/10 - 0.1
    x_obst4_end = 4*Lx/10
    y_obst4_start = Lx/10
    y_obst4_end = Ly
    # Obstacle 5
    x_obst5_start = 5*Lx/10 - 0.1
    x_obst5_end = 5*Lx/10
    y_obst5_start = 0.
    y_obst5_end = Ly - Lx/10
    # Obstacle 6
    x_obst6_start = 6*Lx/10 - 0.1
    x_obst6_end = 6*Lx/10
    y_obst6_start = Lx/10
    y_obst6_end = Ly
    # Obstacle 7
    x_obst7_start = 7*Lx/10 - 0.1
    x_obst7_end = 7*Lx/10
    y_obst7_start = 0.
    y_obst7_end = Ly - Lx/10
    # Obstacle 8
    x_obst8_start = 8*Lx/10 - 0.1
    x_obst8_end = 8*Lx/10
    y_obst8_start = Lx/10
    y_obst8_end = Ly
    # Obstacle 9
    x_obst9_start = 9*Lx/10 - 0.1
    x_obst9_end = 9*Lx/10
    y_obst9_start = 0.
    y_obst9_end = Ly - Lx/10
    # Obstacle 10
    x_obst10_start = Lx - 0.1
    x_obst10_end = Lx
    y_obst10_start = Lx/10
    y_obst10_end = Ly

    # Defining obstacles near each corner
    def defining_obstacles(x_obst_start, x_obst_end, y_obst_start, y_obst_end, y_next_start, y_next_end, x_2next_start, n, m, p, position):
        
        X_obst = [x_obst_start, x_obst_end, y_obst_start, y_obst_end]
        if position == 'south':
            for i in range(n):
                xstart = X_obst[4*i+1]
                xend = xstart + (x_2next_start - X_obst[1])/m
                ystart = X_obst[2]
                X_obst.append(xstart); X_obst.append(xend); X_obst.append(ystart)
                if i == 0:
                    yend = y_next_start*p
                    X_obst.append(yend)
                if i > 0:
                    yend = X_obst[4*i+3]*p
                    X_obst.append(yend)

        if position == 'north':
            for i in range(n):
                xstart = X_obst[4*i+1]
                xend = xstart + (x_2next_start - X_obst[1])/m
                if i == 0:
                    ystart = X_obst[3] - (X_obst[3] - y_next_end)*p
                if i > 0:
                    ystart = X_obst[3] - (X_obst[3] - X_obst[4*i+2])*p
                yend = X_obst[3]
                X_obst.append(xstart); X_obst.append(xend); X_obst.append(ystart); X_obst.append(yend)

        return X_obst

    X_obst1 = defining_obstacles(x_obst1_start, x_obst1_end, y_obst1_start, y_obst1_end, y_obst2_start, y_obst2_end, x_obst3_start, 8, 80, 0.6, 'south')
    X_obst2 = defining_obstacles(x_obst2_start, x_obst2_end, y_obst2_start, y_obst2_end, y_obst3_start, y_obst3_end, x_obst4_start, 8, 80, 0.6, 'north')
    X_obst3 = defining_obstacles(x_obst3_start, x_obst3_end, y_obst3_start, y_obst3_end, y_obst4_start, y_obst4_end, x_obst5_start, 8, 80, 0.6, 'south')
    X_obst4 = defining_obstacles(x_obst4_start, x_obst4_end, y_obst4_start, y_obst4_end, y_obst5_start, y_obst5_end, x_obst6_start, 8, 80, 0.6, 'north')
    X_obst5 = defining_obstacles(x_obst5_start, x_obst5_end, y_obst5_start, y_obst5_end, y_obst6_start, y_obst6_end, x_obst7_start, 8, 80, 0.6, 'south')
    X_obst6 = defining_obstacles(x_obst6_start, x_obst6_end, y_obst6_start, y_obst6_end, y_obst7_start, y_obst7_end, x_obst8_start, 8, 80, 0.6, 'north')
    X_obst7 = defining_obstacles(x_obst7_start, x_obst7_end, y_obst7_start, y_obst7_end, y_obst8_start, y_obst8_end, x_obst9_start, 8, 80, 0.6, 'south')
    X_obst8 = defining_obstacles(x_obst8_start, x_obst8_end, y_obst8_start, y_obst8_end, y_obst9_start, y_obst9_end, x_obst10_start, 8, 80, 0.6, 'north')
    X_obst9 = defining_obstacles(x_obst9_start, x_obst9_end, y_obst9_start, y_obst9_end, y_obst10_start, y_obst10_end, x_obst10_end, 8, 80, 0.6, 'south')
    X_obst10 = [x_obst10_start, x_obst10_end, y_obst10_start, y_obst10_end]

    # Obstacles definition: rectangle with base xs:xe and height ys:ye
    def where_obst(x, y, X_obst):
        
        typed_X = List()
        for i in range(len(X_obst) // 4):
            x_start = X_obst[4*i]; x_end = X_obst[4*i+1]; y_start = X_obst[4*i+2]; y_end = X_obst[4*i+3]
            xs = np.where(x <= x_start)[0][-1] + 1
            xe = np.where(x < x_end)[0][-1] + 1
            ys = np.where(y <= y_start)[0][-1] + 1
            ye = np.where(y < y_end)[0][-1] + 1
            typed_X.append(xs); typed_X.append(xe); typed_X.append(ys); typed_X.append(ye)

        return typed_X

    X1 = where_obst(x, y, X_obst1)
    X2 = where_obst(x, y, X_obst2)
    X3 = where_obst(x, y, X_obst3)
    X4 = where_obst(x, y, X_obst4)
    X5 = where_obst(x, y, X_obst5)
    X6 = where_obst(x, y, X_obst6)
    X7 = where_obst(x, y, X_obst7)
    X8 = where_obst(x, y, X_obst8)
    X9 = where_obst(x, y, X_obst9)
    X10 = where_obst(x, y, X_obst10)

    # Definind the idraulic radius
    dR = x_obst3_start - x_obst2_end

    return [X1, X2, X3, X4, X5, X6, X7, X8, X9, X10], dR



# Set 1 in obstacles coordinates
def flag(flagu,flagv,flagp, XX):

    # Set flag to 1 in obstacle cells
    @njit
    def flag_def(flagu, flagv, flagp, X):

        for i in range(len(X) // 4):
            xs = X[4*i]; xe = X[4*i+1]; ys = X[4*i+2]; ye = X[4*i+3]
            flagu[xs-1:xe+1, ys:ye+1] = 1
            flagv[xs:xe+1, ys-1:ye+1] = 1
            flagp[xs:xe+1, ys:ye+1] = 1

        return flagu, flagv, flagp

    for i in range(len(XX)):
        flagu, flagv, flagp = flag_def(flagu, flagv, flagp, XX[i])

    return flagu, flagv, flagp



# Initialization of u velocity avoiding obstacles
def u_initialize(u, XX):

    # Initialize u velocity on obstacles = 0
    @njit
    def u_velocity(u, X):

        for i in range(len(X) // 4):
            xs = X[4*i]; xe = X[4*i+1]; ys = X[4*i+2]; ye = X[4*i+3]
            u[xs-1:xe+1, ys:ye+1] = 0

        return u

    for i in range(len(XX)):
        u = u_velocity(u, XX[i])

    return u