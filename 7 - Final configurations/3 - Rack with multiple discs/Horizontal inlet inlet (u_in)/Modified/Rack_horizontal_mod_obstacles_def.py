import numpy as np
from   numba import njit
from   numba.typed import List
import matplotlib.pyplot as plt
from   matplotlib.patches import Rectangle

# Defining obstacles coordinates (semi-automatically!!!)
def Obstacles(x, y, Lx, Ly):
    
    # Obstacle coordinates
    # Obstacle 1
    x_obst1_start = 0.
    x_obst1_end = 8*Lx/9 + 0.1
    y_obst1_start = Ly/2 - 0.05
    y_obst1_end = Ly/2 + 0.05

    Hole_diameter = 0.05
    # Obstacle 2
    x_obst2_start = Lx/9
    x_obst2_end = Lx/9 + 0.1
    y_obst2_1_start = 0.15
    y_obst2_1_end = 0.79 - Hole_diameter/2
    y_obst2_2_start = 0.79 + Hole_diameter/2
    y_obst2_2_end = y_obst1_start
    # Obstacle 3
    x_obst3_start = Lx/9
    x_obst3_end = Lx/9 + 0.1
    y_obst3_1_start = y_obst1_end
    y_obst3_1_end = 1.21 - Hole_diameter/2
    y_obst3_2_start = 1.21 + Hole_diameter/2
    y_obst3_2_end = Ly - 0.15
    # Obstacle 4
    x_obst4_start = 2*Lx/9
    x_obst4_end = 2*Lx/9 + 0.1
    y_obst4_1_start = 0.15
    y_obst4_1_end = 0.79 - Hole_diameter/2
    y_obst4_2_start = 0.79 + Hole_diameter/2
    y_obst4_2_end = y_obst1_start
    # Obstacle 5
    x_obst5_start = 2*Lx/9
    x_obst5_end = 2*Lx/9 + 0.1
    y_obst5_1_start = y_obst1_end
    y_obst5_1_end = 1.21 - Hole_diameter/2
    y_obst5_2_start = 1.21 + Hole_diameter/2
    y_obst5_2_end = Ly - 0.15
    # Obstacle 6
    x_obst6_start = 3*Lx/9
    x_obst6_end = 3*Lx/9 + 0.1
    y_obst6_1_start = 0.15
    y_obst6_1_end = 0.79 - Hole_diameter/2
    y_obst6_2_start = 0.79 + Hole_diameter/2
    y_obst6_2_end = y_obst1_start
    # Obstacle 7
    x_obst7_start = 3*Lx/9
    x_obst7_end = 3*Lx/9 + 0.1
    y_obst7_1_start = y_obst1_end
    y_obst7_1_end = 1.21 - Hole_diameter/2
    y_obst7_2_start = 1.21 + Hole_diameter/2
    y_obst7_2_end = Ly - 0.15
    # Obstacle 8
    x_obst8_start = 4*Lx/9
    x_obst8_end = 4*Lx/9 + 0.1
    y_obst8_1_start = 0.15
    y_obst8_1_end = 0.79 - Hole_diameter/2
    y_obst8_2_start = 0.79 + Hole_diameter/2
    y_obst8_2_end = y_obst1_start
    # Obstacle 9
    x_obst9_start = 4*Lx/9
    x_obst9_end = 4*Lx/9 + 0.1
    y_obst9_1_start = y_obst1_end
    y_obst9_1_end = 1.21 - Hole_diameter/2
    y_obst9_2_start = 1.21 + Hole_diameter/2
    y_obst9_2_end = Ly - 0.15
    # Obstacle 10
    x_obst10_start = 5*Lx/9
    x_obst10_end = 5*Lx/9 + 0.1
    y_obst10_1_start = 0.15
    y_obst10_1_end = 0.79 - Hole_diameter/2
    y_obst10_2_start = 0.79 + Hole_diameter/2
    y_obst10_2_end = y_obst1_start
    # Obstacle 11
    x_obst11_start = 5*Lx/9
    x_obst11_end = 5*Lx/9 + 0.1
    y_obst11_1_start = y_obst1_end
    y_obst11_1_end = 1.21 - Hole_diameter/2
    y_obst11_2_start = 1.21 + Hole_diameter/2
    y_obst11_2_end = Ly - 0.15
    # Obstacle 12
    x_obst12_start = 6*Lx/9
    x_obst12_end = 6*Lx/9 + 0.1
    y_obst12_1_start = 0.15
    y_obst12_1_end = 0.79 - Hole_diameter/2
    y_obst12_2_start = 0.79 + Hole_diameter/2
    y_obst12_2_end = y_obst1_start
    # Obstacle 13
    x_obst13_start = 6*Lx/9
    x_obst13_end = 6*Lx/9 + 0.1
    y_obst13_1_start = y_obst1_end
    y_obst13_1_end = 1.21 - Hole_diameter/2
    y_obst13_2_start = 1.21 + Hole_diameter/2
    y_obst13_2_end = Ly - 0.15
    # Obstacle 14
    x_obst14_start = 7*Lx/9
    x_obst14_end = 7*Lx/9 + 0.1
    y_obst14_1_start = 0.15
    y_obst14_1_end = 0.79 - Hole_diameter/2
    y_obst14_2_start = 0.79 + Hole_diameter/2
    y_obst14_2_end = y_obst1_start
    # Obstacle 15
    x_obst15_start = 7*Lx/9
    x_obst15_end = 7*Lx/9 + 0.1
    y_obst15_1_start = y_obst1_end
    y_obst15_1_end = 1.21 - Hole_diameter/2
    y_obst15_2_start = 1.21 + Hole_diameter/2
    y_obst15_2_end = Ly - 0.15
    # Obstacle 16
    x_obst16_start = 8*Lx/9
    x_obst16_end = 8*Lx/9 + 0.1
    y_obst16_1_start = 0.15
    y_obst16_1_end = 0.79 - Hole_diameter/2
    y_obst16_2_start = 0.79 + Hole_diameter/2
    y_obst16_2_end = y_obst1_start
    # Obstacle 17
    x_obst17_start = 8*Lx/9
    x_obst17_end = 8*Lx/9 + 0.1
    y_obst17_1_start = y_obst1_end
    y_obst17_1_end = 1.21 - Hole_diameter/2
    y_obst17_2_start = 1.21 + Hole_diameter/2
    y_obst17_2_end = Ly - 0.15

    # Obstacle 18
    x_obst18_start = x_obst2_end + (x_obst4_start - x_obst2_end)/2 - 0.05
    x_obst18_end = x_obst18_start + 0.05
    y_obst18_start = 0.
    y_obst18_end = 0.13
    # Obstacle 19
    x_obst19_start = x_obst2_end + (x_obst4_start - x_obst2_end)/2 - 0.05
    x_obst19_end = x_obst19_start + 0.05
    y_obst19_start = Ly - 0.13
    y_obst19_end = Ly
    # Obstacle 20
    x_obst20_start = x_obst4_end + (x_obst6_start - x_obst4_end)/2 - 0.05
    x_obst20_end = x_obst20_start + 0.05
    y_obst20_start = 0.
    y_obst20_end = 0.13
    # Obstacle 21
    x_obst21_start = x_obst4_end + (x_obst6_start - x_obst4_end)/2 - 0.05
    x_obst21_end = x_obst21_start + 0.05
    y_obst21_start = Ly - 0.13
    y_obst21_end = Ly
    # Obstacle 22
    x_obst22_start = x_obst6_end + (x_obst8_start - x_obst6_end)/2 - 0.05
    x_obst22_end = x_obst22_start + 0.05
    y_obst22_start = 0.
    y_obst22_end = 0.13
    # Obstacle 23
    x_obst23_start = x_obst6_end + (x_obst8_start - x_obst6_end)/2 - 0.05
    x_obst23_end = x_obst23_start + 0.05
    y_obst23_start = Ly - 0.13
    y_obst23_end = Ly
    # Obstacle 24
    x_obst24_start = x_obst8_end + (x_obst10_start - x_obst8_end)/2 - 0.05
    x_obst24_end = x_obst24_start + 0.05
    y_obst24_start = 0.
    y_obst24_end = 0.13
    # Obstacle 25
    x_obst25_start = x_obst8_end + (x_obst10_start - x_obst8_end)/2 - 0.05
    x_obst25_end = x_obst25_start + 0.05
    y_obst25_start = Ly - 0.13
    y_obst25_end = Ly
    # Obstacle 26
    x_obst26_start = x_obst10_end + (x_obst12_start - x_obst10_end)/2 - 0.05
    x_obst26_end = x_obst26_start + 0.05
    y_obst26_start = 0.
    y_obst26_end = 0.13
    # Obstacle 27
    x_obst27_start = x_obst10_end + (x_obst12_start - x_obst10_end)/2 - 0.05
    x_obst27_end = x_obst27_start + 0.05
    y_obst27_start = Ly - 0.13
    y_obst27_end = Ly
    # Obstacle 28
    x_obst28_start = x_obst12_end + (x_obst14_start - x_obst12_end)/2 - 0.05
    x_obst28_end = x_obst28_start + 0.05
    y_obst28_start = 0.
    y_obst28_end = 0.13
    # Obstacle 29
    x_obst29_start = x_obst12_end + (x_obst14_start - x_obst12_end)/2 - 0.05
    x_obst29_end = x_obst29_start + 0.05
    y_obst29_start = Ly - 0.13
    y_obst29_end = Ly
    # Obstacle 30
    x_obst30_start = x_obst14_end + (x_obst16_start - x_obst14_end)/2 - 0.05
    x_obst30_end = x_obst30_start + 0.05
    y_obst30_start = 0.
    y_obst30_end = 0.13
    # Obstacle 31
    x_obst31_start = x_obst14_end + (x_obst16_start - x_obst14_end)/2 - 0.05
    x_obst31_end = x_obst31_start + 0.05
    y_obst31_start = Ly - 0.13
    y_obst31_end = Ly

    # Obstacle 32
    x_obst32_start = Lx - 0.05
    x_obst32_end = Lx
    y_obst32_start = 0.
    y_obst32_end = Ly/3
    # Obstacle 33
    x_obst33_start = Lx - 0.05
    x_obst33_end = Lx
    y_obst33_start = 2*Ly/3
    y_obst33_end = Ly
    
    # Entrance coordinates
    x_entrance1_start = x_obst2_start - (x_obst4_start - x_obst2_end)/2 - 0.05 - 0.026
    x_entrance1_end = x_entrance1_start + 0.026
    x_entrance2_start = x_entrance1_start
    x_entrance2_end = x_entrance1_end
    x_entrance3_start = x_obst18_start - 0.026
    x_entrance3_end = x_obst18_start
    x_entrance4_start = x_obst19_start - 0.026
    x_entrance4_end = x_obst19_start
    x_entrance5_start = x_obst20_start - 0.026
    x_entrance5_end = x_obst20_start
    x_entrance6_start = x_obst21_start - 0.026
    x_entrance6_end = x_obst21_start
    x_entrance7_start = x_obst22_start - 0.026
    x_entrance7_end = x_obst22_start
    x_entrance8_start = x_obst23_start - 0.026
    x_entrance8_end = x_obst23_start
    x_entrance9_start = x_obst24_start - 0.026
    x_entrance9_end = x_obst24_start
    x_entrance10_start = x_obst25_start - 0.026
    x_entrance10_end = x_obst25_start
    x_entrance11_start = x_obst26_start - 0.026
    x_entrance11_end = x_obst26_start
    x_entrance12_start = x_obst27_start - 0.026
    x_entrance12_end = x_obst27_start
    x_entrance13_start = x_obst28_start - 0.026
    x_entrance13_end = x_obst28_start
    x_entrance14_start = x_obst29_start - 0.026
    x_entrance14_end = x_obst29_start
    x_entrance15_start = x_obst30_start - 0.026
    x_entrance15_end = x_obst30_start
    x_entrance16_start = x_obst31_start - 0.026
    x_entrance16_end = x_obst31_start

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

    n = 0; m = 100; p = 0.75   # Obstacles parameters
    X_obst1 = [x_obst1_start, x_obst1_end, y_obst1_start, y_obst1_end]
    
    X_obst2_1 = [x_obst2_start, x_obst2_end, y_obst2_1_start, y_obst2_1_end]
    X_obst3_1 = defining_obstacles(x_obst3_start, x_obst3_end, y_obst3_1_start, y_obst3_1_end, x_obst5_start, n, m, p, 'south')
    X_obst4_1 = [x_obst4_start, x_obst4_end, y_obst4_1_start, y_obst4_1_end]
    X_obst5_1 = defining_obstacles(x_obst5_start, x_obst5_end, y_obst5_1_start, y_obst5_1_end, x_obst7_start, n, m, p, 'south')
    X_obst6_1 = [x_obst6_start, x_obst6_end, y_obst6_1_start, y_obst6_1_end]
    X_obst7_1 = defining_obstacles(x_obst7_start, x_obst7_end, y_obst7_1_start, y_obst7_1_end, x_obst9_start, n, m, p, 'south')
    X_obst8_1 = [x_obst8_start, x_obst8_end, y_obst8_1_start, y_obst8_1_end]
    X_obst9_1 = defining_obstacles(x_obst9_start, x_obst9_end, y_obst9_1_start, y_obst9_1_end, x_obst11_start, n, m, p, 'south')
    X_obst10_1 = [x_obst10_start, x_obst10_end, y_obst10_1_start, y_obst10_1_end]
    X_obst11_1 = defining_obstacles(x_obst11_start, x_obst11_end, y_obst11_1_start, y_obst11_1_end, x_obst13_start, n, m, p, 'south')
    X_obst12_1 = [x_obst12_start, x_obst12_end, y_obst12_1_start, y_obst12_1_end]
    X_obst13_1 = defining_obstacles(x_obst13_start, x_obst13_end, y_obst13_1_start, y_obst13_1_end, x_obst15_start, n, m, p, 'south')
    X_obst14_1 = [x_obst14_start, x_obst14_end, y_obst14_1_start, y_obst14_1_end]
    X_obst15_1 = defining_obstacles(x_obst15_start, x_obst15_end, y_obst15_1_start, y_obst15_1_end, x_obst17_start, n, m, p, 'south')
    X_obst16_1 = [x_obst16_start, x_obst16_end, y_obst16_1_start, y_obst16_1_end]
    X_obst17_1 = [x_obst17_start, x_obst17_end, y_obst17_1_start, y_obst17_1_end]
    X_obst2_2 = defining_obstacles(x_obst2_start, x_obst2_end, y_obst2_2_start, y_obst2_2_end, x_obst4_start, n, m, p, 'north')
    X_obst3_2 = [x_obst3_start, x_obst3_end, y_obst3_2_start, y_obst3_2_end]
    X_obst4_2 = defining_obstacles(x_obst4_start, x_obst4_end, y_obst4_2_start, y_obst4_2_end, x_obst6_start, n, m, p, 'north')
    X_obst5_2 = [x_obst5_start, x_obst5_end, y_obst5_2_start, y_obst5_2_end]
    X_obst6_2 = defining_obstacles(x_obst6_start, x_obst6_end, y_obst6_2_start, y_obst6_2_end, x_obst8_start, n, m, p, 'north')
    X_obst7_2 = [x_obst7_start, x_obst7_end, y_obst7_2_start, y_obst7_2_end]
    X_obst8_2 = defining_obstacles(x_obst8_start, x_obst8_end, y_obst8_2_start, y_obst8_2_end, x_obst10_start, n, m, p, 'north')
    X_obst9_2 = [x_obst9_start, x_obst9_end, y_obst9_2_start, y_obst9_2_end]
    X_obst10_2 = defining_obstacles(x_obst10_start, x_obst10_end, y_obst10_2_start, y_obst10_2_end, x_obst12_start, n, m, p, 'north')
    X_obst11_2 = [x_obst11_start, x_obst11_end, y_obst11_2_start, y_obst11_2_end]
    X_obst12_2 = defining_obstacles(x_obst12_start, x_obst12_end, y_obst12_2_start, y_obst12_2_end, x_obst14_start, n, m, p, 'north')
    X_obst13_2 = [x_obst13_start, x_obst13_end, y_obst13_2_start, y_obst13_2_end]
    X_obst14_2 = defining_obstacles(x_obst14_start, x_obst14_end, y_obst14_2_start, y_obst14_2_end, x_obst16_start, n, m, p, 'north')
    X_obst15_2 = [x_obst15_start, x_obst15_end, y_obst15_2_start, y_obst15_2_end]
    X_obst16_2 = [x_obst16_start, x_obst16_end, y_obst16_2_start, y_obst16_2_end]
    X_obst17_2 = [x_obst17_start, x_obst17_end, y_obst17_2_start, y_obst17_2_end]
    
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
    X_obst32 = [x_obst32_start, x_obst32_end, y_obst32_start, y_obst32_end]
    X_obst33 = [x_obst33_start, x_obst33_end, y_obst33_start, y_obst33_end]

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
    
    X2_1 = where_obst(x, y, X_obst2_1)
    X3_1 = where_obst(x, y, X_obst3_1)
    X4_1 = where_obst(x, y, X_obst4_1)
    X5_1 = where_obst(x, y, X_obst5_1)
    X6_1 = where_obst(x, y, X_obst6_1)
    X7_1 = where_obst(x, y, X_obst7_1)
    X8_1 = where_obst(x, y, X_obst8_1)
    X9_1 = where_obst(x, y, X_obst9_1)
    X10_1 = where_obst(x, y, X_obst10_1)
    X11_1 = where_obst(x, y, X_obst11_1)
    X12_1 = where_obst(x, y, X_obst12_1)
    X13_1 = where_obst(x, y, X_obst13_1)
    X14_1 = where_obst(x, y, X_obst14_1)
    X15_1 = where_obst(x, y, X_obst15_1)
    X16_1 = where_obst(x, y, X_obst16_1)
    X17_1 = where_obst(x, y, X_obst17_1)
    X2_2 = where_obst(x, y, X_obst2_2)
    X3_2 = where_obst(x, y, X_obst3_2)
    X4_2 = where_obst(x, y, X_obst4_2)
    X5_2 = where_obst(x, y, X_obst5_2)
    X6_2 = where_obst(x, y, X_obst6_2)
    X7_2 = where_obst(x, y, X_obst7_2)
    X8_2 = where_obst(x, y, X_obst8_2)
    X9_2 = where_obst(x, y, X_obst9_2)
    X10_2 = where_obst(x, y, X_obst10_2)
    X11_2 = where_obst(x, y, X_obst11_2)
    X12_2 = where_obst(x, y, X_obst12_2)
    X13_2 = where_obst(x, y, X_obst13_2)
    X14_2 = where_obst(x, y, X_obst14_2)
    X15_2 = where_obst(x, y, X_obst15_2)
    X16_2 = where_obst(x, y, X_obst16_2)
    X17_2 = where_obst(x, y, X_obst17_2)
    
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
    X32 = where_obst(x, y, X_obst32)
    X33 = where_obst(x, y, X_obst33)
    
    xs_entrance1 = np.where(x <= x_entrance1_start)[0][-1] + 1
    xe_entrance1 = np.where(x < x_entrance1_end)[0][-1] + 1
    xs_entrance2 = np.where(x <= x_entrance2_start)[0][-1] + 1
    xe_entrance2 = np.where(x < x_entrance2_end)[0][-1] + 1
    xs_entrance3 = np.where(x <= x_entrance3_start)[0][-1] + 1
    xe_entrance3 = np.where(x < x_entrance3_end)[0][-1] + 1
    xs_entrance4 = np.where(x <= x_entrance4_start)[0][-1] + 1
    xe_entrance4 = np.where(x < x_entrance4_end)[0][-1] + 1
    xs_entrance5 = np.where(x <= x_entrance5_start)[0][-1] + 1
    xe_entrance5 = np.where(x < x_entrance5_end)[0][-1] + 1
    xs_entrance6 = np.where(x <= x_entrance6_start)[0][-1] + 1
    xe_entrance6 = np.where(x < x_entrance6_end)[0][-1] + 1
    xs_entrance7 = np.where(x <= x_entrance7_start)[0][-1] + 1
    xe_entrance7 = np.where(x < x_entrance7_end)[0][-1] + 1
    xs_entrance8 = np.where(x <= x_entrance8_start)[0][-1] + 1
    xe_entrance8 = np.where(x < x_entrance8_end)[0][-1] + 1
    xs_entrance9 = np.where(x <= x_entrance9_start)[0][-1] + 1
    xe_entrance9 = np.where(x < x_entrance9_end)[0][-1] + 1
    xs_entrance10 = np.where(x <= x_entrance10_start)[0][-1] + 1
    xe_entrance10 = np.where(x < x_entrance10_end)[0][-1] + 1
    xs_entrance11 = np.where(x <= x_entrance11_start)[0][-1] + 1
    xe_entrance11 = np.where(x < x_entrance11_end)[0][-1] + 1
    xs_entrance12 = np.where(x <= x_entrance12_start)[0][-1] + 1
    xe_entrance12 = np.where(x < x_entrance12_end)[0][-1] + 1
    xs_entrance13 = np.where(x <= x_entrance13_start)[0][-1] + 1
    xe_entrance13 = np.where(x < x_entrance13_end)[0][-1] + 1
    xs_entrance14 = np.where(x <= x_entrance14_start)[0][-1] + 1
    xe_entrance14 = np.where(x < x_entrance14_end)[0][-1] + 1
    xs_entrance15 = np.where(x <= x_entrance15_start)[0][-1] + 1
    xe_entrance15 = np.where(x < x_entrance15_end)[0][-1] + 1
    xs_entrance16 = np.where(x <= x_entrance16_start)[0][-1] + 1
    xe_entrance16 = np.where(x < x_entrance16_end)[0][-1] + 1
    Xentrance = [xs_entrance1, xe_entrance1, xs_entrance2, xe_entrance2, xs_entrance3, xe_entrance3, xs_entrance4, xe_entrance4,
                 xs_entrance5, xe_entrance5, xs_entrance6, xe_entrance6, xs_entrance7, xe_entrance7, xs_entrance8, xe_entrance8,
                 xs_entrance9, xe_entrance9, xs_entrance10, xe_entrance10, xs_entrance11, xe_entrance11, xs_entrance12, xe_entrance12,
                 xs_entrance13, xe_entrance13, xs_entrance14, xe_entrance14, xs_entrance15, xe_entrance15, xs_entrance16, xe_entrance16]

    # Definind the idraulic radius
    dR = x_obst4_start - x_obst3_end

    return [X1, X2_1, X3_1, X4_1, X5_1, X6_1, X7_1, X8_1, X9_1, X10_1, X11_1, X12_1, X13_1, X14_1, X15_1, X16_1, X17_1, X2_2, X3_2, X4_2, X5_2, X6_2, X7_2, X8_2,\
            X9_2, X10_2, X11_2, X12_2, X13_2, X14_2, X15_2, X16_2, X17_2, X18, X19, X20, X21, X22, X23, X24, X25, X26, X27, X28, X29, X30, X31, X32, X33], dR, Xentrance



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