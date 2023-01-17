import numpy as np
from   numba import njit
from   numba.typed import List
import matplotlib.pyplot as plt
from   matplotlib.patches import Rectangle

# Defining obstacles coordinates (semi-automatically!!!)
def Obstacles(x, y):

    # Obstacle coordinates
    # Obstacle 1
    x_obst1_start = 0.4
    x_obst1_end = 0.5
    y_obst1_start = 0.2
    y_obst1_end = 1.3
    # Obstacle 2
    x_obst2_start = 1.0
    x_obst2_end = 1.1
    y_obst2_start = 0.2
    y_obst2_end = 1.3
    # Obstacle 3
    x_obst3_start = 1.6
    x_obst3_end = 1.7
    y_obst3_start = 0.2
    y_obst3_end = 1.3
    # Obstacle 4
    x_obst4_start = 2.2
    x_obst4_end = 2.3
    y_obst4_start = 0.2
    y_obst4_end = 1.3
    # Obstacle 5
    x_obst5_start = 2.8
    x_obst5_end = 2.9
    y_obst5_start = 0.2
    y_obst5_end = 1.3
    # Obstacle 6
    x_obst6_start = 3.4
    x_obst6_end = 3.5
    y_obst6_start = 0.2
    y_obst6_end = 1.3
    # Obstacle 7
    x_obst7_start = 4.0
    x_obst7_end = 4.1
    y_obst7_start = 0.2
    y_obst7_end = 1.3
    # Obstacle 8
    x_obst8_start = 4.6
    x_obst8_end = 4.7
    y_obst8_start = 0.2
    y_obst8_end = 1.3

    # Obstacle 18
    x_obst18_start = 0.7
    x_obst18_end = 0.8
    y_obst18_start = 0.
    y_obst18_end = 0.25
    # Obstacle 19
    x_obst19_start = 0.7
    x_obst19_end = 0.8
    y_obst19_start = 1.25
    y_obst19_end = 1.5
    # Obstacle 20
    x_obst20_start = 1.3
    x_obst20_end = 1.4
    y_obst20_start = 0.
    y_obst20_end = 0.25
    # Obstacle 21
    x_obst21_start = 1.3
    x_obst21_end = 1.4
    y_obst21_start = 1.25
    y_obst21_end = 1.5
    # Obstacle 22
    x_obst22_start = 1.9
    x_obst22_end = 2.0
    y_obst22_start = 0.
    y_obst22_end = 0.25
    # Obstacle 23
    x_obst23_start = 1.9
    x_obst23_end = 2.0
    y_obst23_start = 1.25
    y_obst23_end = 1.5
    # Obstacle 24
    x_obst24_start = 2.5
    x_obst24_end = 2.6
    y_obst24_start = 0.
    y_obst24_end = 0.25
    # Obstacle 25
    x_obst25_start = 2.5
    x_obst25_end = 2.6
    y_obst25_start = 1.25
    y_obst25_end = 1.5
    # Obstacle 26
    x_obst26_start = 3.1
    x_obst26_end = 3.2
    y_obst26_start = 0.
    y_obst26_end = 0.25
    # Obstacle 27
    x_obst27_start = 3.1
    x_obst27_end = 3.2
    y_obst27_start = 1.25
    y_obst27_end = 1.5
    # Obstacle 28
    x_obst28_start = 3.7
    x_obst28_end = 3.8
    y_obst28_start = 0.
    y_obst28_end = 0.25
    # Obstacle 29
    x_obst29_start = 3.7
    x_obst29_end = 3.8
    y_obst29_start = 1.25
    y_obst29_end = 1.5
    # Obstacle 30
    x_obst30_start = 4.3
    x_obst30_end = 4.4
    y_obst30_start = 0.
    y_obst30_end = 0.25
    # Obstacle 31
    x_obst31_start = 4.3
    x_obst31_end = 4.4
    y_obst31_start = 1.25
    y_obst31_end = 1.5

    # Defining obstacles near each corner
    def defining_obstacles(x_obst_start, x_obst_end, y_obst_start, y_obst_end, x_2next_start, n, m, p, position):
        
        X_obst = [x_obst_start, x_obst_end, y_obst_start, y_obst_end]
        if position == 'south':
            for i in range(n):
                xstart = X_obst[4*i+1]
                xend = xstart + (x_2next_start - X_obst[1])/m
                ystart = X_obst[2]
                yend = ystart + (X_obst[4*i+3] - ystart)*p
                X_obst.append(xstart); X_obst.append(xend); X_obst.append(ystart); X_obst.append(yend)

        if position == 'north':
            for i in range(n):
                xstart = X_obst[4*i+1]
                xend = xstart + (x_2next_start - X_obst[1])/m
                ystart = X_obst[3] - (X_obst[3] - X_obst[4*i+2])*p
                yend = X_obst[3]
                X_obst.append(xstart); X_obst.append(xend); X_obst.append(ystart); X_obst.append(yend)

        return X_obst

    n = 15; m = 100; p = 0.7   # Obstacles parameters
    X_obst1 = [x_obst1_start, x_obst1_end, y_obst1_start, y_obst1_end]
    X_obst2 = [x_obst2_start, x_obst2_end, y_obst2_start, y_obst2_end]
    X_obst3 = [x_obst3_start, x_obst3_end, y_obst3_start, y_obst3_end]
    X_obst4 = [x_obst4_start, x_obst4_end, y_obst4_start, y_obst4_end]
    X_obst5 = [x_obst5_start, x_obst5_end, y_obst5_start, y_obst5_end]
    X_obst6 = [x_obst6_start, x_obst6_end, y_obst6_start, y_obst6_end]
    X_obst7 = [x_obst7_start, x_obst7_end, y_obst7_start, y_obst7_end]
    X_obst8 = [x_obst8_start, x_obst8_end, y_obst8_start, y_obst8_end]

    X_obst18 = [x_obst18_start, x_obst18_end, y_obst18_start, y_obst18_end]
    X_obst19 = [x_obst19_start, x_obst19_end, y_obst19_start, y_obst19_end]
    X_obst20 = [x_obst20_start, x_obst20_end, y_obst20_start, y_obst20_end]
    X_obst21 = [x_obst21_start, x_obst21_end, y_obst21_start, y_obst21_end]
    X_obst22 = [x_obst22_start, x_obst22_end, y_obst22_start, y_obst22_end]
    X_obst23 = [x_obst23_start, x_obst23_end, y_obst23_start, y_obst23_end]
    X_obst24 = [x_obst24_start, x_obst24_end, y_obst24_start, y_obst24_end]
    X_obst25 = [x_obst25_start, x_obst25_end, y_obst25_start, y_obst25_end]
    X_obst26 = [x_obst26_start, x_obst26_end, y_obst26_start, y_obst26_end]
    X_obst27 = [x_obst27_start, x_obst27_end, y_obst27_start, y_obst27_end]
    X_obst28 = [x_obst28_start, x_obst28_end, y_obst28_start, y_obst28_end]
    X_obst29 = [x_obst29_start, x_obst29_end, y_obst29_start, y_obst29_end]
    X_obst30 = [x_obst30_start, x_obst30_end, y_obst30_start, y_obst30_end]
    X_obst31 = [x_obst31_start, x_obst31_end, y_obst31_start, y_obst31_end]

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

    X18 = where_obst(x, y, X_obst18)
    X19 = where_obst(x, y, X_obst19)
    X20 = where_obst(x, y, X_obst20)
    X21 = where_obst(x, y, X_obst21)
    X22 = where_obst(x, y, X_obst22)
    X23 = where_obst(x, y, X_obst23)
    X24 = where_obst(x, y, X_obst24)
    X25 = where_obst(x, y, X_obst25)
    X26 = where_obst(x, y, X_obst26)
    X27 = where_obst(x, y, X_obst27)
    X28 = where_obst(x, y, X_obst28)
    X29 = where_obst(x, y, X_obst29)
    X30 = where_obst(x, y, X_obst30)
    X31 = where_obst(x, y, X_obst31)

    # Definind the idraulic radius
    dR = x_obst5_start - x_obst4_end

    return [X1, X2, X3, X4, X5, X6, X7, X8, X18, X19, X20, X21, X22, X23, X24, X25, X26, X27, X28, X29, X30, X31], dR



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