import numpy as np
from   numba import njit
from   numba.typed import List

# Defining obstacles coordinates (semi-automatically!!!)
def Obstacles(x, y, Lx, Ly):

    # Obstacle coordinates
    Lx = Lx - 0.2
    # Obstacle 1
    x_obst1_start = 0.
    x_obst1_end = 0.05
    y_obst1_start = 0. + 0.4
    y_obst1_end = Ly - 0.4
    # Obstacle 2
    x_obst2_start = Lx/9
    x_obst2_end = Lx/9 + 0.1
    y_obst2_start = 0.
    y_obst2_end = Ly/2 - 0.1
    # Obstacle 3
    x_obst3_start = Lx/9
    x_obst3_end = Lx/9 + 0.1
    y_obst3_start = Ly/2 + 0.1
    y_obst3_end = Ly
    # Obstacle 4
    x_obst4_start = 2*Lx/9
    x_obst4_end = 2*Lx/9 + 0.1
    y_obst4_start = 0.3
    y_obst4_end = Ly - 0.3
    # Obstacle 5
    x_obst5_start = 3*Lx/9
    x_obst5_end = 3*Lx/9 + 0.1
    y_obst5_start = 0.
    y_obst5_end = Ly/2 - 0.1
    # Obstacle 6
    x_obst6_start = 3*Lx/9
    x_obst6_end = 3*Lx/9 + 0.1
    y_obst6_start = Ly/2 + 0.1
    y_obst6_end = Ly
    # Obstacle 7
    x_obst7_start = 4*Lx/9
    x_obst7_end = 4*Lx/9 + 0.1
    y_obst7_start = 0.3
    y_obst7_end = Ly - 0.3
    # Obstacle 8
    x_obst8_start = 5*Lx/9
    x_obst8_end = 5*Lx/9 + 0.1
    y_obst8_start = 0.
    y_obst8_end = Ly/2 - 0.1
    # Obstacle 9
    x_obst9_start = 5*Lx/9
    x_obst9_end = 5*Lx/9 + 0.1
    y_obst9_start = Ly/2 + 0.1
    y_obst9_end = Ly
    # Obstacle 10
    x_obst10_start = 6*Lx/9
    x_obst10_end = 6*Lx/9 + 0.1
    y_obst10_start = 0.3
    y_obst10_end = Ly - 0.3
    # Obstacle 11
    x_obst11_start = 7*Lx/9
    x_obst11_end = 7*Lx/9 + 0.1
    y_obst11_start = 0.
    y_obst11_end = Ly/2 - 0.1
    # Obstacle 12
    x_obst12_start = 7*Lx/9
    x_obst12_end = 7*Lx/9 + 0.1
    y_obst12_start = Ly/2 + 0.1
    y_obst12_end = Ly
    # Obstacle 13
    x_obst13_start = 8*Lx/9
    x_obst13_end = 8*Lx/9 + 0.1
    y_obst13_start = 0.3
    y_obst13_end = Ly - 0.3
    # Obstacle 14
    x_obst14_start = Lx + 0.1
    x_obst14_end = Lx + 0.2
    y_obst14_start = 0.
    y_obst14_end = Ly/2 - 0.1
    # Obstacle 15
    x_obst15_start = Lx + 0.1
    x_obst15_end = Lx + 0.2
    y_obst15_start = Ly/2 + 0.1
    y_obst15_end = Ly

    # Defining obstacles near each corner
    def defining_obstacles(x_obst_start, x_obst_end, y_obst_start, y_obst_end, x_2next_start, n, m, p, position):
        
        X_obst = [x_obst_start, x_obst_end, y_obst_start, y_obst_end]
        if position == 'south':
            for i in range(n):
                xstart = X_obst[4*i+1]
                xend = xstart + (x_2next_start - X_obst[1])/m
                ystart = X_obst[2]
                yend = X_obst[4*i+3]*p
                X_obst.append(xstart); X_obst.append(xend); X_obst.append(ystart); X_obst.append(yend)

        if position == 'north':
            for i in range(n):
                xstart = X_obst[4*i+1]
                xend = xstart + (x_2next_start - X_obst[1])/m
                ystart = X_obst[3] - (X_obst[3] - X_obst[4*i+2])*p
                yend = X_obst[3]
                X_obst.append(xstart); X_obst.append(xend); X_obst.append(ystart); X_obst.append(yend)

        return X_obst

    X_obst1 = [x_obst1_start, x_obst1_end, y_obst1_start, y_obst1_end]
    X_obst2 = defining_obstacles(x_obst2_start, x_obst2_end, y_obst2_start, y_obst2_end, x_obst4_start, 10, 70, 0.7, 'south')
    X_obst3 = defining_obstacles(x_obst3_start, x_obst3_end, y_obst3_start, y_obst3_end, x_obst4_start, 10, 70, 0.7, 'north')
    X_obst4 = [x_obst4_start, x_obst4_end, y_obst4_start, y_obst4_end]
    X_obst5 = defining_obstacles(x_obst5_start, x_obst5_end, y_obst5_start, y_obst5_end, x_obst7_start, 10, 70, 0.7, 'south')
    X_obst6 = defining_obstacles(x_obst6_start, x_obst6_end, y_obst6_start, y_obst6_end, x_obst7_start, 10, 70, 0.7, 'north')
    X_obst7 = [x_obst7_start, x_obst7_end, y_obst7_start, y_obst7_end]
    X_obst8 = defining_obstacles(x_obst8_start, x_obst8_end, y_obst8_start, y_obst8_end, x_obst10_start, 10, 70, 0.7, 'south')
    X_obst9 = defining_obstacles(x_obst9_start, x_obst9_end, y_obst9_start, y_obst9_end, x_obst10_start, 10, 70, 0.7, 'north')
    X_obst10 = [x_obst10_start, x_obst10_end, y_obst10_start, y_obst10_end]
    X_obst11 = defining_obstacles(x_obst11_start, x_obst11_end, y_obst11_start, y_obst11_end, x_obst13_start, 10, 80, 0.7, 'south')
    X_obst12 = defining_obstacles(x_obst12_start, x_obst12_end, y_obst12_start, y_obst12_end, x_obst13_start, 10, 80, 0.7, 'north')
    X_obst13 = [x_obst13_start, x_obst13_end, y_obst13_start, y_obst13_end]
    X_obst14 = [x_obst14_start, x_obst14_end, y_obst14_start, y_obst14_end]
    X_obst15 = [x_obst15_start, x_obst15_end, y_obst15_start, y_obst15_end]

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
    X11 = where_obst(x, y, X_obst11)
    X12 = where_obst(x, y, X_obst12)
    X13 = where_obst(x, y, X_obst13)
    X14 = where_obst(x, y, X_obst14)
    X15 = where_obst(x, y, X_obst15)

    # Definind the idraulic radius
    dR = x_obst4_start - x_obst3_end

    return [X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15], dR



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