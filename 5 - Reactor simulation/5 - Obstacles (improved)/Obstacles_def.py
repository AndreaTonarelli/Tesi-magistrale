import numpy as np

# Defining obstacles coordinates (manually!!!)
def Obstacles(x, y):

    # Obstacle coordinates
    # Obstacle 1
    x_obst1_start = 0.
    x_obst1_end = 0.1
    y_obst1_start = 0.
    y_obst1_end = 0.7
    # Obstacle 2
    x_obst2_start = 0.4
    x_obst2_end = 0.5
    y_obst2_start = 0.5
    y_obst2_end = 1.2
    # Obstacle 3
    x_obst3_start = 0.8
    x_obst3_end = 0.9
    y_obst3_start = 0.
    y_obst3_end = 0.7
    # Obstacle 4
    x_obst4_start = 1.2
    x_obst4_end = 1.3
    y_obst4_start = 0.5
    y_obst4_end = 1.2
    # Obstacle 5
    x_obst5_start = 1.6
    x_obst5_end = 1.7
    y_obst5_start = 0.
    y_obst5_end = 0.7
    # Obstacle 6
    x_obst6_start = 2.0
    x_obst6_end = 2.1
    y_obst6_start = 0.5
    y_obst6_end = 1.2
    # Obstacle 7
    x_obst7_start = 2.4
    x_obst7_end = 2.5
    y_obst7_start = 0.
    y_obst7_end = 0.7
    # Obstacle 8
    x_obst8_start = 2.9
    x_obst8_end = 3.0
    y_obst8_start = 0.5
    y_obst8_end = 1.2
    # Obstacle 9
    x_obst9_start = 3.4
    x_obst9_end = 3.5
    y_obst9_start = 0.
    y_obst9_end = 0.7
    # Obstacle 10
    x_obst10_start = 3.9
    x_obst10_end = 4.0
    y_obst10_start = 0.5
    y_obst10_end = 1.2

    # Defining obstacles near each corner
    # near Obstacle 1 (south border)
    x_obst11_start = x_obst1_end
    x_obst11_end = x_obst1_end + (x_obst3_start-x_obst1_end)/10
    y_obst11_start = y_obst1_start
    y_obst11_end = y_obst2_start/2
    x_obst12_start = x_obst11_end
    x_obst12_end = x_obst11_end + (x_obst3_start-x_obst1_end)/10
    y_obst12_start = y_obst1_start
    y_obst12_end = y_obst11_end/2
    # near Obstacle 2 (north border)
    x_obst21_start = x_obst2_end
    x_obst21_end = x_obst2_end + (x_obst4_start-x_obst2_end)/10
    y_obst21_start = y_obst2_end - (y_obst2_end-y_obst3_end)/2
    y_obst21_end = y_obst2_end
    x_obst22_start = x_obst21_end
    x_obst22_end = x_obst21_end + (x_obst4_start-x_obst2_end)/10
    y_obst22_start = y_obst21_end - (y_obst2_end-y_obst21_start)/2
    y_obst22_end = y_obst2_end
    # near Obstacle 3 (south border)
    x_obst31_start = x_obst3_end
    x_obst31_end = x_obst3_end + (x_obst5_start-x_obst3_end)/10
    y_obst31_start = y_obst3_start
    y_obst31_end = y_obst4_start/2
    x_obst32_start = x_obst31_end
    x_obst32_end = x_obst31_end + (x_obst5_start-x_obst3_end)/10
    y_obst32_start = y_obst3_start
    y_obst32_end = y_obst31_end/2
    # near Obstacle 4 (north border)
    x_obst41_start = x_obst4_end
    x_obst41_end = x_obst4_end + (x_obst6_start-x_obst4_end)/10
    y_obst41_start = y_obst4_end - (y_obst4_end-y_obst5_end)/2
    y_obst41_end = y_obst4_end
    x_obst42_start = x_obst41_end
    x_obst42_end = x_obst41_end + (x_obst6_start-x_obst4_end)/10
    y_obst42_start = y_obst41_end - (y_obst4_end-y_obst41_start)/2
    y_obst42_end = y_obst4_end
    # near Obstacle 5 (south border)
    x_obst51_start = x_obst5_end
    x_obst51_end = x_obst5_end + (x_obst7_start-x_obst5_end)/10
    y_obst51_start = y_obst5_start
    y_obst51_end = y_obst6_start/2
    x_obst52_start = x_obst51_end
    x_obst52_end = x_obst51_end + (x_obst7_start-x_obst5_end)/10
    y_obst52_start = y_obst5_start
    y_obst52_end = y_obst51_end/2
    # near Obstacle 6 (north border)
    x_obst61_start = x_obst6_end
    x_obst61_end = x_obst6_end + (x_obst8_start-x_obst6_end)/10
    y_obst61_start = y_obst6_end - (y_obst6_end-y_obst7_end)/2
    y_obst61_end = y_obst6_end
    x_obst62_start = x_obst61_end
    x_obst62_end = x_obst61_end + (x_obst8_start-x_obst6_end)/10
    y_obst62_start = y_obst61_end - (y_obst6_end-y_obst61_start)/2
    y_obst62_end = y_obst6_end
    # near Obstacle 7 (south border)
    x_obst71_start = x_obst7_end
    x_obst71_end = x_obst7_end + (x_obst9_start-x_obst7_end)/10
    y_obst71_start = y_obst7_start
    y_obst71_end = y_obst8_start/2
    x_obst72_start = x_obst71_end
    x_obst72_end = x_obst71_end + (x_obst9_start-x_obst7_end)/10
    y_obst72_start = y_obst7_start
    y_obst72_end = y_obst71_end/2
    # near Obstacle 8 (north border)
    x_obst81_start = x_obst8_end
    x_obst81_end = x_obst8_end + (x_obst10_start-x_obst8_end)/10
    y_obst81_start = y_obst8_end - (y_obst8_end-y_obst9_end)/2
    y_obst81_end = y_obst8_end
    x_obst82_start = x_obst81_end
    x_obst82_end = x_obst81_end + (x_obst10_start-x_obst8_end)/10
    y_obst82_start = y_obst81_end - (y_obst8_end-y_obst81_start)/2
    y_obst82_end = y_obst8_end
    # near Obstacle 9 (south border)
    x_obst91_start = x_obst9_end
    x_obst91_end = x_obst9_end + (x_obst10_end-x_obst9_end)/10
    y_obst91_start = y_obst9_start
    y_obst91_end = y_obst10_start/2
    x_obst92_start = x_obst91_end
    x_obst92_end = x_obst91_end + (x_obst10_end-x_obst9_end)/10
    y_obst92_start = y_obst9_start
    y_obst92_end = y_obst91_end/2

    # Obstacles definition: rectangle with base xs:xe and height ys:ye
    # Obstacle 1
    xs1 = np.where(x <= x_obst1_start)[0][-1] + 1
    xe1 = np.where(x < x_obst1_end)[0][-1] + 1
    ys1 = np.where(y <= y_obst1_start)[0][-1] + 1
    ye1 = np.where(y < y_obst1_end)[0][-1] + 1
    # Obstacle 2
    xs2 = np.where(x <= x_obst2_start)[0][-1] + 1
    xe2 = np.where(x < x_obst2_end)[0][-1] + 1
    ys2 = np.where(y <= y_obst2_start)[0][-1] + 1
    ye2 = np.where(y < y_obst2_end)[0][-1] + 1
    # Obstacle 3
    xs3 = np.where(x <= x_obst3_start)[0][-1] + 1
    xe3 = np.where(x < x_obst3_end)[0][-1] + 1
    ys3 = np.where(y <= y_obst3_start)[0][-1] + 1
    ye3 = np.where(y < y_obst3_end)[0][-1] + 1
    # Obstacle 4
    xs4 = np.where(x <= x_obst4_start)[0][-1] + 1
    xe4 = np.where(x < x_obst4_end)[0][-1] + 1
    ys4 = np.where(y <= y_obst4_start)[0][-1] + 1
    ye4 = np.where(y < y_obst4_end)[0][-1] + 1
    # Obstacle 5
    xs5 = np.where(x <= x_obst5_start)[0][-1] + 1
    xe5 = np.where(x < x_obst5_end)[0][-1] + 1
    ys5 = np.where(y <= y_obst5_start)[0][-1] + 1
    ye5 = np.where(y < y_obst5_end)[0][-1] + 1
    # Obstacle 6
    xs6 = np.where(x <= x_obst6_start)[0][-1] + 1
    xe6 = np.where(x < x_obst6_end)[0][-1] + 1
    ys6 = np.where(y <= y_obst6_start)[0][-1] + 1
    ye6 = np.where(y < y_obst6_end)[0][-1] + 1
    # Obstacle 7
    xs7 = np.where(x <= x_obst7_start)[0][-1] + 1
    xe7 = np.where(x < x_obst7_end)[0][-1] + 1
    ys7 = np.where(y <= y_obst7_start)[0][-1] + 1
    ye7 = np.where(y < y_obst7_end)[0][-1] + 1
    # Obstacle 8
    xs8 = np.where(x <= x_obst8_start)[0][-1] + 1
    xe8 = np.where(x < x_obst8_end)[0][-1] + 1
    ys8 = np.where(y <= y_obst8_start)[0][-1] + 1
    ye8 = np.where(y < y_obst8_end)[0][-1] + 1
    # Obstacle 9
    xs9 = np.where(x <= x_obst9_start)[0][-1] + 1
    xe9 = np.where(x < x_obst9_end)[0][-1] + 1
    ys9 = np.where(y <= y_obst9_start)[0][-1] + 1
    ye9 = np.where(y < y_obst9_end)[0][-1] + 1
    # Obstacle 10
    xs10 = np.where(x <= x_obst10_start)[0][-1] + 1
    xe10 = np.where(x < x_obst10_end)[0][-1] + 1
    ys10 = np.where(y <= y_obst10_start)[0][-1] + 1
    ye10 = np.where(y < y_obst10_end)[0][-1] + 1
    # near Obstacle 1
    xs11 = np.where(x <= x_obst11_start)[0][-1] + 1
    xe11 = np.where(x < x_obst11_end)[0][-1] + 1
    ys11 = np.where(y <= y_obst11_start)[0][-1] + 1
    ye11 = np.where(y < y_obst11_end)[0][-1] + 1
    xs12 = np.where(x <= x_obst12_start)[0][-1] + 1
    xe12 = np.where(x < x_obst12_end)[0][-1] + 1
    ys12 = np.where(y <= y_obst12_start)[0][-1] + 1
    ye12 = np.where(y < y_obst12_end)[0][-1] + 1
    # near Obstacle 2
    xs21 = np.where(x <= x_obst21_start)[0][-1] + 1
    xe21 = np.where(x < x_obst21_end)[0][-1] + 1
    ys21 = np.where(y <= y_obst21_start)[0][-1] + 1
    ye21 = np.where(y < y_obst21_end)[0][-1] + 1
    xs22 = np.where(x <= x_obst22_start)[0][-1] + 1
    xe22 = np.where(x < x_obst22_end)[0][-1] + 1
    ys22 = np.where(y <= y_obst22_start)[0][-1] + 1
    ye22 = np.where(y < y_obst22_end)[0][-1] + 1
    # near Obstacle 3
    xs31 = np.where(x <= x_obst31_start)[0][-1] + 1
    xe31 = np.where(x < x_obst31_end)[0][-1] + 1
    ys31 = np.where(y <= y_obst31_start)[0][-1] + 1
    ye31 = np.where(y < y_obst31_end)[0][-1] + 1
    xs32 = np.where(x <= x_obst32_start)[0][-1] + 1
    xe32 = np.where(x < x_obst32_end)[0][-1] + 1
    ys32 = np.where(y <= y_obst32_start)[0][-1] + 1
    ye32 = np.where(y < y_obst32_end)[0][-1] + 1
    # near Obstacle 4
    xs41 = np.where(x <= x_obst41_start)[0][-1] + 1
    xe41 = np.where(x < x_obst41_end)[0][-1] + 1
    ys41 = np.where(y <= y_obst41_start)[0][-1] + 1
    ye41 = np.where(y < y_obst41_end)[0][-1] + 1
    xs42 = np.where(x <= x_obst42_start)[0][-1] + 1
    xe42 = np.where(x < x_obst42_end)[0][-1] + 1
    ys42 = np.where(y <= y_obst42_start)[0][-1] + 1
    ye42 = np.where(y < y_obst42_end)[0][-1] + 1
    # near Obstacle 5
    xs51 = np.where(x <= x_obst51_start)[0][-1] + 1
    xe51 = np.where(x < x_obst51_end)[0][-1] + 1
    ys51 = np.where(y <= y_obst51_start)[0][-1] + 1
    ye51 = np.where(y < y_obst51_end)[0][-1] + 1
    xs52 = np.where(x <= x_obst52_start)[0][-1] + 1
    xe52 = np.where(x < x_obst52_end)[0][-1] + 1
    ys52 = np.where(y <= y_obst52_start)[0][-1] + 1
    ye52 = np.where(y < y_obst52_end)[0][-1] + 1
    # near Obstacle 6
    xs61 = np.where(x <= x_obst61_start)[0][-1] + 1
    xe61 = np.where(x < x_obst61_end)[0][-1] + 1
    ys61 = np.where(y <= y_obst61_start)[0][-1] + 1
    ye61 = np.where(y < y_obst61_end)[0][-1] + 1
    xs62 = np.where(x <= x_obst62_start)[0][-1] + 1
    xe62 = np.where(x < x_obst62_end)[0][-1] + 1
    ys62 = np.where(y <= y_obst62_start)[0][-1] + 1
    ye62 = np.where(y < y_obst62_end)[0][-1] + 1
    # near Obstacle 7
    xs71 = np.where(x <= x_obst71_start)[0][-1] + 1
    xe71 = np.where(x < x_obst71_end)[0][-1] + 1
    ys71 = np.where(y <= y_obst71_start)[0][-1] + 1
    ye71 = np.where(y < y_obst71_end)[0][-1] + 1
    xs72 = np.where(x <= x_obst72_start)[0][-1] + 1
    xe72 = np.where(x < x_obst72_end)[0][-1] + 1
    ys72 = np.where(y <= y_obst72_start)[0][-1] + 1
    ye72 = np.where(y < y_obst72_end)[0][-1] + 1
    # near Obstacle 8
    xs81 = np.where(x <= x_obst81_start)[0][-1] + 1
    xe81 = np.where(x < x_obst81_end)[0][-1] + 1
    ys81 = np.where(y <= y_obst81_start)[0][-1] + 1
    ye81 = np.where(y < y_obst81_end)[0][-1] + 1
    xs82 = np.where(x <= x_obst82_start)[0][-1] + 1
    xe82 = np.where(x < x_obst82_end)[0][-1] + 1
    ys82 = np.where(y <= y_obst82_start)[0][-1] + 1
    ye82 = np.where(y < y_obst82_end)[0][-1] + 1
    # near Obstacle 9
    xs91 = np.where(x <= x_obst91_start)[0][-1] + 1
    xe91 = np.where(x < x_obst91_end)[0][-1] + 1
    ys91 = np.where(y <= y_obst91_start)[0][-1] + 1
    ye91 = np.where(y < y_obst91_end)[0][-1] + 1
    xs92 = np.where(x <= x_obst92_start)[0][-1] + 1
    xe92 = np.where(x < x_obst92_end)[0][-1] + 1
    ys92 = np.where(y <= y_obst92_start)[0][-1] + 1
    ye92 = np.where(y < y_obst92_end)[0][-1] + 1

    # Defining the obstacles coordinates
    X1 = (xs1, xe1, ys1, ye1, xs11, xe11, ys11, ye11, xs12, xe12, ys12, ye12)
    X2 = (xs2, xe2, ys2, ye2, xs21, xe21, ys21, ye21, xs22, xe22, ys22, ye22)
    X3 = (xs3, xe3, ys3, ye3, xs31, xe31, ys31, ye31, xs32, xe32, ys32, ye32)
    X4 = (xs4, xe4, ys4, ye4, xs41, xe41, ys41, ye41, xs42, xe42, ys42, ye42)
    X5 = (xs5, xe5, ys5, ye5, xs51, xe51, ys51, ye51, xs52, xe52, ys52, ye52)
    X6 = (xs6, xe6, ys6, ye6, xs61, xe61, ys61, ye61, xs62, xe62, ys62, ye62)
    X7 = (xs7, xe7, ys7, ye7, xs71, xe71, ys71, ye71, xs72, xe72, ys72, ye72)
    X8 = (xs8, xe8, ys8, ye8, xs81, xe81, ys81, ye81, xs82, xe82, ys82, ye82)
    X9 = (xs9, xe9, ys9, ye9, xs91, xe91, ys91, ye91, xs92, xe92, ys92, ye92)
    X10 = (xs10, xe10, ys10, ye10)

    # Definind the idraulic radius
    dR = x_obst3_start - x_obst2_end

    return X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, dR


# Set 1 in obstacles coordinates
def flag(flagu,flagv,flagp, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10):

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

    # Set flag to 1 in obstacle cells
    flagu[xs1-1:xe1+1, ys1:ye1+1] = 1       # Obstacle 1
    flagv[xs1:xe1+1, ys1-1:ye1+1] = 1
    flagp[xs1:xe1+1, ys1:ye1+1] = 1

    flagu[xs2-1:xe2+1, ys2:ye2+1] = 1       # Obstacle 2
    flagv[xs2:xe2+1, ys2-1:ye2+1] = 1
    flagp[xs2:xe2+1, ys2:ye2+1] = 1

    flagu[xs3-1:xe3+1, ys3:ye3+1] = 1       # Obstacle 3
    flagv[xs3:xe3+1, ys3-1:ye3+1] = 1
    flagp[xs3:xe3+1, ys3:ye3+1] = 1

    flagu[xs4-1:xe4+1, ys4:ye4+1] = 1       # Obstacle 4
    flagv[xs4:xe4+1, ys4-1:ye4+1] = 1
    flagp[xs4:xe4+1, ys4:ye4+1] = 1

    flagu[xs5-1:xe5+1, ys5:ye5+1] = 1       # Obstacle 5
    flagv[xs5:xe5+1, ys5-1:ye5+1] = 1
    flagp[xs5:xe5+1, ys5:ye5+1] = 1

    flagu[xs6-1:xe6+1, ys6:ye6+1] = 1       # Obstacle 6
    flagv[xs6:xe6+1, ys6-1:ye6+1] = 1
    flagp[xs6:xe6+1, ys6:ye6+1] = 1

    flagu[xs7-1:xe7+1, ys7:ye7+1] = 1       # Obstacle 7
    flagv[xs7:xe7+1, ys7-1:ye7+1] = 1
    flagp[xs7:xe7+1, ys7:ye7+1] = 1

    flagu[xs8-1:xe8+1, ys8:ye8+1] = 1       # Obstacle 8
    flagv[xs8:xe8+1, ys8-1:ye8+1] = 1
    flagp[xs8:xe8+1, ys8:ye8+1] = 1

    flagu[xs9-1:xe9+1, ys9:ye9+1] = 1       # Obstacle 9
    flagv[xs9:xe9+1, ys9-1:ye9+1] = 1
    flagp[xs9:xe9+1, ys9:ye9+1] = 1
    flagu[xs10-1:xe10+1, ys10:ye10+1] = 1       # Obstacle 10
    flagv[xs10:xe10+1, ys10-1:ye10+1] = 1
    flagp[xs10:xe10+1, ys10:ye10+1] = 1


    # Set flag to 1 in obstacle cells near corners
    flagu[xs11-1:xe11+1, ys11:ye11+1] = 1       # Obstacle 1
    flagv[xs11:xe11+1, ys11-1:ye11+1] = 1
    flagp[xs11:xe11+1, ys11:ye11+1] = 1
    flagu[xs12-1:xe12+1, ys12:ye12+1] = 1
    flagv[xs12:xe12+1, ys12-1:ye12+1] = 1
    flagp[xs12:xe12+1, ys12:ye12+1] = 1

    flagu[xs21-1:xe21+1, ys21:ye21+1] = 1       # Obstacle 2
    flagv[xs21:xe21+1, ys21-1:ye21+1] = 1
    flagp[xs21:xe21+1, ys21:ye21+1] = 1
    flagu[xs22-1:xe22+1, ys22:ye22+1] = 1
    flagv[xs22:xe22+1, ys22-1:ye22+1] = 1
    flagp[xs22:xe22+1, ys22:ye22+1] = 1

    flagu[xs31-1:xe31+1, ys31:ye31+1] = 1       # Obstacle 3
    flagv[xs31:xe31+1, ys31-1:ye31+1] = 1
    flagp[xs31:xe31+1, ys31:ye31+1] = 1
    flagu[xs32-1:xe32+1, ys32:ye32+1] = 1
    flagv[xs32:xe32+1, ys32-1:ye32+1] = 1
    flagp[xs32:xe32+1, ys32:ye32+1] = 1

    flagu[xs41-1:xe41+1, ys41:ye41+1] = 1       # Obstacle 4
    flagv[xs41:xe41+1, ys41-1:ye41+1] = 1
    flagp[xs41:xe41+1, ys41:ye41+1] = 1
    flagu[xs42-1:xe42+1, ys42:ye42+1] = 1
    flagv[xs42:xe42+1, ys42-1:ye42+1] = 1
    flagp[xs42:xe42+1, ys42:ye42+1] = 1

    flagu[xs51-1:xe51+1, ys51:ye51+1] = 1       # Obstacle 5
    flagv[xs51:xe51+1, ys51-1:ye51+1] = 1
    flagp[xs51:xe51+1, ys51:ye51+1] = 1
    flagu[xs52-1:xe52+1, ys52:ye52+1] = 1
    flagv[xs52:xe52+1, ys52-1:ye52+1] = 1
    flagp[xs52:xe52+1, ys52:ye52+1] = 1

    flagu[xs61-1:xe61+1, ys61:ye61+1] = 1       # Obstacle 6
    flagv[xs61:xe61+1, ys61-1:ye61+1] = 1
    flagp[xs61:xe61+1, ys61:ye61+1] = 1
    flagu[xs62-1:xe62+1, ys62:ye62+1] = 1
    flagv[xs62:xe62+1, ys62-1:ye62+1] = 1
    flagp[xs62:xe62+1, ys62:ye62+1] = 1

    flagu[xs71-1:xe71+1, ys71:ye71+1] = 1       # Obstacle 7
    flagv[xs71:xe71+1, ys71-1:ye71+1] = 1
    flagp[xs71:xe71+1, ys71:ye71+1] = 1
    flagu[xs72-1:xe72+1, ys72:ye72+1] = 1
    flagv[xs72:xe72+1, ys72-1:ye72+1] = 1
    flagp[xs72:xe72+1, ys72:ye72+1] = 1

    flagu[xs81-1:xe81+1, ys81:ye81+1] = 1       # Obstacle 8
    flagv[xs81:xe81+1, ys81-1:ye81+1] = 1
    flagp[xs81:xe81+1, ys81:ye81+1] = 1
    flagu[xs82-1:xe82+1, ys82:ye82+1] = 1
    flagv[xs82:xe82+1, ys82-1:ye82+1] = 1
    flagp[xs82:xe82+1, ys82:ye82+1] = 1

    flagu[xs91-1:xe91+1, ys91:ye91+1] = 1       # Obstacle 9
    flagv[xs91:xe91+1, ys91-1:ye91+1] = 1
    flagp[xs91:xe91+1, ys91:ye91+1] = 1
    flagu[xs92-1:xe92+1, ys92:ye92+1] = 1
    flagv[xs92:xe92+1, ys92-1:ye92+1] = 1
    flagp[xs92:xe92+1, ys92:ye92+1] = 1

    return flagu, flagv, flagp



# Initialization of u velocity avoiding obstacles
def u_initialize(u, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10):

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

    # Initialize u velocity on obstacles = 0
    u[xs1-1:xe1+1, ys1:ye1+1] = 0       # Obstacle 1
    u[xs2-1:xe2+1, ys2:ye2+1] = 0       # Obstacle 2
    u[xs3-1:xe3+1, ys3:ye3+1] = 0       # Obstacle 3
    u[xs4-1:xe4+1, ys4:ye4+1] = 0       # Obstacle 4
    u[xs5-1:xe5+1, ys5:ye5+1] = 0       # Obstacle 5
    u[xs6-1:xe6+1, ys6:ye6+1] = 0       # Obstacle 6
    u[xs7-1:xe7+1, ys7:ye7+1] = 0       # Obstacle 7
    u[xs8-1:xe8+1, ys8:ye8+1] = 0       # Obstacle 8
    u[xs9-1:xe9+1, ys9:ye9+1] = 0       # Obstacle 9
    u[xs10-1:xe10+1, ys10:ye10+1] = 0   # Obstacle 10

    u[xs11-1:xe11+1, ys11:ye11+1] = 0   # near Obstacle 1
    u[xs12-1:xe12+1, ys12:ye12+1] = 0
    u[xs21-1:xe21+1, ys21:ye21+1] = 0   # near Obstacle 2
    u[xs22-1:xe22+1, ys22:ye22+1] = 0
    u[xs31-1:xe31+1, ys31:ye31+1] = 0   # near Obstacle 3
    u[xs32-1:xe32+1, ys32:ye32+1] = 0
    u[xs41-1:xe41+1, ys41:ye41+1] = 0   # near Obstacle 4
    u[xs42-1:xe42+1, ys42:ye42+1] = 0
    u[xs51-1:xe51+1, ys51:ye51+1] = 0   # near Obstacle 5
    u[xs52-1:xe52+1, ys52:ye52+1] = 0
    u[xs61-1:xe61+1, ys61:ye61+1] = 0   # near Obstacle 6
    u[xs62-1:xe62+1, ys62:ye62+1] = 0
    u[xs71-1:xe71+1, ys71:ye71+1] = 0   # near Obstacle 7
    u[xs72-1:xe72+1, ys72:ye72+1] = 0
    u[xs81-1:xe81+1, ys81:ye81+1] = 0   # near Obstacle 8
    u[xs82-1:xe82+1, ys82:ye82+1] = 0
    u[xs91-1:xe91+1, ys91:ye91+1] = 0   # near Obstacle 9
    u[xs92-1:xe92+1, ys92:ye92+1] = 0

    return u