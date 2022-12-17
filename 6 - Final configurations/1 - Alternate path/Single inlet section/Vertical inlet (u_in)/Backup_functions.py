# Defining obstacles near each corner
# near Obstacle 1 (south border)
n = 60
x_obst11_start = x_obst1_end
x_obst11_end = x_obst11_start + (x_obst3_start-x_obst1_end)/n
y_obst11_start = y_obst1_start
y_obst11_end = y_obst2_start*1/2
x_obst12_start = x_obst11_end
x_obst12_end = x_obst12_start + (x_obst3_start-x_obst1_end)/n
y_obst12_start = y_obst1_start
y_obst12_end = y_obst11_end*2/3
x_obst13_start = x_obst12_end
x_obst13_end = x_obst13_start + (x_obst3_start-x_obst1_end)/n
y_obst13_start = y_obst1_start
y_obst13_end = y_obst12_end*2/3
x_obst14_start = x_obst13_end
x_obst14_end = x_obst14_start + (x_obst3_start-x_obst1_end)/n
y_obst14_start = y_obst1_start
y_obst14_end = y_obst13_end*2/3
x_obst15_start = x_obst14_end
x_obst15_end = x_obst15_start + (x_obst3_start-x_obst1_end)/n
y_obst15_start = y_obst1_start
y_obst15_end = y_obst14_end*2/3
x_obst16_start = x_obst15_end
x_obst16_end = x_obst16_start + (x_obst3_start-x_obst1_end)/n
y_obst16_start = y_obst1_start
y_obst16_end = y_obst15_end*1/2
x_obst17_start = x_obst16_end
x_obst17_end = x_obst17_start + (x_obst3_start-x_obst1_end)/n
y_obst17_start = y_obst1_start
y_obst17_end = y_obst16_end*1/2
x_obst18_start = x_obst17_end
x_obst18_end = x_obst18_start + (x_obst3_start-x_obst1_end)/n
y_obst18_start = y_obst1_start
y_obst18_end = y_obst17_end*1/2
x_obst19_start = x_obst18_end
x_obst19_end = x_obst19_start + (x_obst3_start-x_obst1_end)/n
y_obst19_start = y_obst1_start
y_obst19_end = y_obst18_end*1/2

# near Obstacle 2 (north border)
x_obst21_start = x_obst2_end
x_obst21_end = x_obst21_start + (x_obst4_start-x_obst2_end)/n
y_obst21_start = y_obst2_end - (y_obst2_end-y_obst3_end)*1/2
y_obst21_end = y_obst2_end
x_obst22_start = x_obst21_end
x_obst22_end = x_obst22_start + (x_obst4_start-x_obst2_end)/n
y_obst22_start = y_obst2_end - (y_obst2_end-y_obst21_start)*2/3
y_obst22_end = y_obst2_end
x_obst23_start = x_obst22_end
x_obst23_end = x_obst23_start + (x_obst4_start-x_obst2_end)/n
y_obst23_start = y_obst2_end - (y_obst2_end-y_obst22_start)*2/3
y_obst23_end = y_obst2_end
x_obst24_start = x_obst23_end
x_obst24_end = x_obst24_start + (x_obst4_start-x_obst2_end)/n
y_obst24_start = y_obst2_end - (y_obst2_end-y_obst23_start)*2/3
y_obst24_end = y_obst2_end
x_obst25_start = x_obst24_end
x_obst25_end = x_obst25_start + (x_obst4_start-x_obst2_end)/n
y_obst25_start = y_obst2_end - (y_obst2_end-y_obst24_start)*2/3
y_obst25_end = y_obst2_end
x_obst26_start = x_obst25_end
x_obst26_end = x_obst26_start + (x_obst4_start-x_obst2_end)/n
y_obst26_start = y_obst2_end - (y_obst2_end-y_obst25_start)*1/2
y_obst26_end = y_obst2_end
x_obst27_start = x_obst26_end
x_obst27_end = x_obst26_start + (x_obst4_start-x_obst2_end)/n
y_obst27_start = y_obst2_end - (y_obst2_end-y_obst26_start)*1/2
y_obst27_end = y_obst2_end
x_obst28_start = x_obst27_end
x_obst28_end = x_obst28_start + (x_obst4_start-x_obst2_end)/n
y_obst28_start = y_obst2_end - (y_obst2_end-y_obst27_start)*1/2
y_obst28_end = y_obst2_end
x_obst29_start = x_obst28_end
x_obst29_end = x_obst29_start + (x_obst4_start-x_obst2_end)/n
y_obst29_start = y_obst2_end - (y_obst2_end-y_obst28_start)*1/2
y_obst29_end = y_obst2_end

# near Obstacle 3 (south border)
x_obst31_start = x_obst3_end
x_obst31_end = x_obst31_start + (x_obst5_start-x_obst3_end)/n
y_obst31_start = y_obst3_start
y_obst31_end = y_obst4_start*1/2
x_obst32_start = x_obst31_end
x_obst32_end = x_obst32_start + (x_obst5_start-x_obst3_end)/n
y_obst32_start = y_obst3_start
y_obst32_end = y_obst31_end*2/3
x_obst33_start = x_obst32_end
x_obst33_end = x_obst33_start + (x_obst5_start-x_obst3_end)/n
y_obst33_start = y_obst3_start
y_obst33_end = y_obst32_end*2/3
x_obst34_start = x_obst33_end
x_obst34_end = x_obst34_start + (x_obst5_start-x_obst3_end)/n
y_obst34_start = y_obst3_start
y_obst34_end = y_obst33_end*2/3
x_obst35_start = x_obst34_end
x_obst35_end = x_obst35_start + (x_obst5_start-x_obst3_end)/n
y_obst35_start = y_obst3_start
y_obst35_end = y_obst34_end*2/3
x_obst36_start = x_obst35_end
x_obst36_end = x_obst36_start + (x_obst5_start-x_obst3_end)/n
y_obst36_start = y_obst3_start
y_obst36_end = y_obst35_end*1/2
x_obst37_start = x_obst36_end
x_obst37_end = x_obst37_start + (x_obst5_start-x_obst3_end)/n
y_obst37_start = y_obst3_start
y_obst37_end = y_obst36_end*1/2
x_obst38_start = x_obst37_end
x_obst38_end = x_obst38_start + (x_obst5_start-x_obst3_end)/n
y_obst38_start = y_obst3_start
y_obst38_end = y_obst37_end*1/2
x_obst39_start = x_obst38_end
x_obst39_end = x_obst39_start + (x_obst5_start-x_obst3_end)/n
y_obst39_start = y_obst3_start
y_obst39_end = y_obst38_end*1/2

# near Obstacle 4 (north border)
x_obst41_start = x_obst4_end
x_obst41_end = x_obst41_start + (x_obst6_start-x_obst4_end)/n
y_obst41_start = y_obst4_end - (y_obst4_end-y_obst5_end)*1/2
y_obst41_end = y_obst4_end
x_obst42_start = x_obst41_end
x_obst42_end = x_obst42_start + (x_obst6_start-x_obst4_end)/n
y_obst42_start = y_obst4_end - (y_obst4_end-y_obst41_start)*2/3
y_obst42_end = y_obst4_end
x_obst43_start = x_obst42_end
x_obst43_end = x_obst43_start + (x_obst6_start-x_obst4_end)/n
y_obst43_start = y_obst4_end - (y_obst4_end-y_obst42_start)*2/3
y_obst43_end = y_obst4_end
x_obst44_start = x_obst43_end
x_obst44_end = x_obst44_start + (x_obst6_start-x_obst4_end)/n
y_obst44_start = y_obst4_end - (y_obst4_end-y_obst43_start)*2/3
y_obst44_end = y_obst4_end
x_obst45_start = x_obst44_end
x_obst45_end = x_obst45_start + (x_obst6_start-x_obst4_end)/n
y_obst45_start = y_obst4_end - (y_obst4_end-y_obst44_start)*2/3
y_obst45_end = y_obst4_end
x_obst46_start = x_obst45_end
x_obst46_end = x_obst46_start + (x_obst6_start-x_obst4_end)/n
y_obst46_start = y_obst4_end - (y_obst4_end-y_obst45_start)*1/2
y_obst46_end = y_obst4_end
x_obst47_start = x_obst46_end
x_obst47_end = x_obst46_start + (x_obst6_start-x_obst4_end)/n
y_obst47_start = y_obst4_end - (y_obst4_end-y_obst46_start)*1/2
y_obst47_end = y_obst4_end
x_obst48_start = x_obst47_end
x_obst48_end = x_obst48_start + (x_obst6_start-x_obst4_end)/n
y_obst48_start = y_obst4_end - (y_obst4_end-y_obst47_start)*1/2
y_obst48_end = y_obst4_end
x_obst49_start = x_obst48_end
x_obst49_end = x_obst49_start + (x_obst6_start-x_obst4_end)/n
y_obst49_start = y_obst4_end - (y_obst4_end-y_obst48_start)*1/2
y_obst49_end = y_obst4_end

# near Obstacle 5 (south border)
x_obst51_start = x_obst5_end
x_obst51_end = x_obst51_start + (x_obst7_start-x_obst5_end)/n
y_obst51_start = y_obst5_start
y_obst51_end = y_obst6_start*1/2
x_obst52_start = x_obst51_end
x_obst52_end = x_obst52_start + (x_obst7_start-x_obst5_end)/n
y_obst52_start = y_obst5_start
y_obst52_end = y_obst51_end*2/3
x_obst53_start = x_obst52_end
x_obst53_end = x_obst53_start + (x_obst7_start-x_obst5_end)/n
y_obst53_start = y_obst5_start
y_obst53_end = y_obst52_end*2/3
x_obst54_start = x_obst53_end
x_obst54_end = x_obst54_start + (x_obst7_start-x_obst5_end)/n
y_obst54_start = y_obst5_start
y_obst54_end = y_obst53_end*2/3
x_obst55_start = x_obst54_end
x_obst55_end = x_obst55_start + (x_obst7_start-x_obst5_end)/n
y_obst55_start = y_obst5_start
y_obst55_end = y_obst54_end*2/3
x_obst56_start = x_obst55_end
x_obst56_end = x_obst56_start + (x_obst7_start-x_obst5_end)/n
y_obst56_start = y_obst5_start
y_obst56_end = y_obst55_end*1/2
x_obst57_start = x_obst56_end
x_obst57_end = x_obst57_start + (x_obst7_start-x_obst5_end)/n
y_obst57_start = y_obst5_start
y_obst57_end = y_obst56_end*1/2
x_obst58_start = x_obst57_end
x_obst58_end = x_obst58_start + (x_obst7_start-x_obst5_end)/n
y_obst58_start = y_obst5_start
y_obst58_end = y_obst57_end*1/2
x_obst59_start = x_obst58_end
x_obst59_end = x_obst59_start + (x_obst7_start-x_obst5_end)/n
y_obst59_start = y_obst5_start
y_obst59_end = y_obst58_end*1/2

# near Obstacle 6 (north border)
x_obst61_start = x_obst6_end
x_obst61_end = x_obst61_start + (x_obst8_start-x_obst6_end)/n
y_obst61_start = y_obst6_end - (y_obst6_end-y_obst7_end)*1/2
y_obst61_end = y_obst6_end
x_obst62_start = x_obst61_end
x_obst62_end = x_obst62_start + (x_obst8_start-x_obst6_end)/n
y_obst62_start = y_obst6_end - (y_obst6_end-y_obst61_start)*2/3
y_obst62_end = y_obst6_end
x_obst63_start = x_obst62_end
x_obst63_end = x_obst63_start + (x_obst8_start-x_obst6_end)/n
y_obst63_start = y_obst6_end - (y_obst6_end-y_obst62_start)*2/3
y_obst63_end = y_obst6_end
x_obst64_start = x_obst63_end
x_obst64_end = x_obst64_start + (x_obst8_start-x_obst6_end)/n
y_obst64_start = y_obst6_end - (y_obst6_end-y_obst63_start)*2/3
y_obst64_end = y_obst6_end
x_obst65_start = x_obst64_end
x_obst65_end = x_obst65_start + (x_obst8_start-x_obst6_end)/n
y_obst65_start = y_obst6_end - (y_obst6_end-y_obst64_start)*2/3
y_obst65_end = y_obst6_end
x_obst66_start = x_obst65_end
x_obst66_end = x_obst66_start + (x_obst8_start-x_obst6_end)/n
y_obst66_start = y_obst6_end - (y_obst6_end-y_obst65_start)*1/2
y_obst66_end = y_obst6_end
x_obst67_start = x_obst66_end
x_obst67_end = x_obst66_start + (x_obst8_start-x_obst6_end)/n
y_obst67_start = y_obst6_end - (y_obst6_end-y_obst66_start)*1/2
y_obst67_end = y_obst6_end
x_obst68_start = x_obst67_end
x_obst68_end = x_obst68_start + (x_obst8_start-x_obst6_end)/n
y_obst68_start = y_obst6_end - (y_obst6_end-y_obst67_start)*1/2
y_obst68_end = y_obst6_end
x_obst69_start = x_obst68_end
x_obst69_end = x_obst69_start + (x_obst8_start-x_obst6_end)/n
y_obst69_start = y_obst6_end - (y_obst6_end-y_obst68_start)*1/2
y_obst69_end = y_obst6_end

# near Obstacle 7 (south border)
x_obst71_start = x_obst7_end
x_obst71_end = x_obst71_start + (x_obst9_start-x_obst7_end)/n
y_obst71_start = y_obst7_start
y_obst71_end = y_obst8_start*1/2
x_obst72_start = x_obst71_end
x_obst72_end = x_obst72_start + (x_obst9_start-x_obst7_end)/n
y_obst72_start = y_obst7_start
y_obst72_end = y_obst71_end*2/3
x_obst73_start = x_obst72_end
x_obst73_end = x_obst73_start + (x_obst9_start-x_obst7_end)/n
y_obst73_start = y_obst7_start
y_obst73_end = y_obst72_end*2/3
x_obst74_start = x_obst73_end
x_obst74_end = x_obst74_start + (x_obst9_start-x_obst7_end)/n
y_obst74_start = y_obst7_start
y_obst74_end = y_obst73_end*2/3
x_obst75_start = x_obst74_end
x_obst75_end = x_obst75_start + (x_obst9_start-x_obst7_end)/n
y_obst75_start = y_obst7_start
y_obst75_end = y_obst74_end*2/3
x_obst76_start = x_obst75_end
x_obst76_end = x_obst76_start + (x_obst9_start-x_obst7_end)/n
y_obst76_start = y_obst7_start
y_obst76_end = y_obst75_end*1/2
x_obst77_start = x_obst76_end
x_obst77_end = x_obst77_start + (x_obst9_start-x_obst7_end)/n
y_obst77_start = y_obst7_start
y_obst77_end = y_obst76_end*1/2
x_obst78_start = x_obst77_end
x_obst78_end = x_obst78_start + (x_obst9_start-x_obst7_end)/n
y_obst78_start = y_obst7_start
y_obst78_end = y_obst77_end*1/2
x_obst79_start = x_obst78_end
x_obst79_end = x_obst79_start + (x_obst9_start-x_obst7_end)/n
y_obst79_start = y_obst7_start
y_obst79_end = y_obst78_end*1/2

# near Obstacle 8 (north border)
x_obst81_start = x_obst8_end
x_obst81_end = x_obst81_start + (x_obst10_start-x_obst8_end)/n
y_obst81_start = y_obst8_end - (y_obst8_end-y_obst9_end)*1/2
y_obst81_end = y_obst8_end
x_obst82_start = x_obst81_end
x_obst82_end = x_obst82_start + (x_obst10_start-x_obst8_end)/n
y_obst82_start = y_obst8_end - (y_obst8_end-y_obst81_start)*2/3
y_obst82_end = y_obst8_end
x_obst83_start = x_obst82_end
x_obst83_end = x_obst83_start + (x_obst10_start-x_obst8_end)/n
y_obst83_start = y_obst8_end - (y_obst8_end-y_obst82_start)*2/3
y_obst83_end = y_obst8_end
x_obst84_start = x_obst83_end
x_obst84_end = x_obst84_start + (x_obst10_start-x_obst8_end)/n
y_obst84_start = y_obst8_end - (y_obst8_end-y_obst83_start)*2/3
y_obst84_end = y_obst8_end
x_obst85_start = x_obst84_end
x_obst85_end = x_obst85_start + (x_obst10_start-x_obst8_end)/n
y_obst85_start = y_obst8_end - (y_obst8_end-y_obst84_start)*2/3
y_obst85_end = y_obst8_end
x_obst86_start = x_obst85_end
x_obst86_end = x_obst86_start + (x_obst10_start-x_obst8_end)/n
y_obst86_start = y_obst8_end - (y_obst8_end-y_obst85_start)*1/2
y_obst86_end = y_obst8_end
x_obst87_start = x_obst86_end
x_obst87_end = x_obst86_start + (x_obst10_start-x_obst8_end)/n
y_obst87_start = y_obst8_end - (y_obst8_end-y_obst86_start)*1/2
y_obst87_end = y_obst8_end
x_obst88_start = x_obst87_end
x_obst88_end = x_obst88_start + (x_obst10_start-x_obst8_end)/n
y_obst88_start = y_obst8_end - (y_obst8_end-y_obst87_start)*1/2
y_obst88_end = y_obst8_end
x_obst89_start = x_obst88_end
x_obst89_end = x_obst89_start + (x_obst10_start-x_obst8_end)/n
y_obst89_start = y_obst8_end - (y_obst8_end-y_obst88_start)*1/2
y_obst89_end = y_obst8_end

# near Obstacle 9 (south border)
x_obst91_start = x_obst9_end
x_obst91_end = x_obst91_start + (x_obst10_end-x_obst9_end)/n
y_obst91_start = y_obst9_start
y_obst91_end = y_obst10_start*1/2
x_obst92_start = x_obst91_end
x_obst92_end = x_obst92_start + (x_obst10_end-x_obst9_end)/n
y_obst92_start = y_obst9_start
y_obst92_end = y_obst91_end*2/3
x_obst93_start = x_obst92_end
x_obst93_end = x_obst93_start + (x_obst10_end-x_obst9_end)/n
y_obst93_start = y_obst9_start
y_obst93_end = y_obst92_end*2/3
x_obst94_start = x_obst93_end
x_obst94_end = x_obst94_start + (x_obst10_end-x_obst9_end)/n
y_obst94_start = y_obst9_start
y_obst94_end = y_obst93_end*2/3
x_obst95_start = x_obst94_end
x_obst95_end = x_obst95_start + (x_obst10_end-x_obst9_end)/n
y_obst95_start = y_obst9_start
y_obst95_end = y_obst94_end*2/3
x_obst96_start = x_obst95_end
x_obst96_end = x_obst96_start + (x_obst10_end-x_obst9_end)/n
y_obst96_start = y_obst9_start
y_obst96_end = y_obst95_end*1/2
x_obst97_start = x_obst96_end
x_obst97_end = x_obst97_start + (x_obst10_end-x_obst9_end)/n
y_obst97_start = y_obst9_start
y_obst97_end = y_obst96_end*1/2
x_obst98_start = x_obst97_end
x_obst98_end = x_obst98_start + (x_obst10_end-x_obst9_end)/n
y_obst98_start = y_obst9_start
y_obst98_end = y_obst97_end*1/2
x_obst99_start = x_obst98_end
x_obst99_end = x_obst99_start + (x_obst10_end-x_obst9_end)/n
y_obst99_start = y_obst9_start
y_obst99_end = y_obst98_end*1/2



# Obstacles definition: rectangle with base xs:xe and height ys:ye
def where_obst(x, y, X_obst):
        X = []
        for i in range(len(X_obst) // 4):
            x_start = X_obst[4*i]; x_end = X_obst[4*i+1]; y_start = X_obst[4*i+2]; y_end = X_obst[4*i+3]
            xs = np.where(x <= x_start)[0][-1] + 1
            xe = np.where(x < x_end)[0][-1] + 1
            ys = np.where(y <= y_start)[0][-1] + 1
            ye = np.where(y < y_end)[0][-1] + 1
            X.append(xs, xe, ys, ye)
        return X

    # Obstacle 1
    xs1, xe1, ys1, ye1 = where_obst(x, y, x_obst1_start, x_obst1_end, y_obst1_start, y_obst1_end)
    # Obstacle 2
    xs2, xe2, ys2, ye2 = where_obst(x, y, x_obst2_start, x_obst2_end, y_obst2_start, y_obst2_end)
    # Obstacle 3
    xs3, xe3, ys3, ye3 = where_obst(x, y, x_obst3_start, x_obst3_end, y_obst3_start, y_obst3_end)
    # Obstacle 4
    xs4, xe4, ys4, ye4 = where_obst(x, y, x_obst4_start, x_obst4_end, y_obst4_start, y_obst4_end)
    # Obstacle 5
    xs5, xe5, ys5, ye5 = where_obst(x, y, x_obst5_start, x_obst5_end, y_obst5_start, y_obst5_end)
    # Obstacle 6
    xs6, xe6, ys6, ye6 = where_obst(x, y, x_obst6_start, x_obst6_end, y_obst6_start, y_obst6_end)
    # Obstacle 7
    xs7, xe7, ys7, ye7 = where_obst(x, y, x_obst7_start, x_obst7_end, y_obst7_start, y_obst7_end)
    # Obstacle 8
    xs8, xe8, ys8, ye8 = where_obst(x, y, x_obst8_start, x_obst8_end, y_obst8_start, y_obst8_end)
    # Obstacle 9
    xs9, xe9, ys9, ye9 = where_obst(x, y, x_obst9_start, x_obst9_end, y_obst9_start, y_obst9_end)
    # Obstacle 10
    xs10, xe10, ys10, ye10 = where_obst(x, y, x_obst10_start, x_obst10_end, y_obst10_start, y_obst10_end)
    # near Obstacle 1
    xs11, xe11, ys11, ye11 = where_obst(x, y, x_obst11_start, x_obst11_end, y_obst11_start, y_obst11_end)
    xs12, xe12, ys12, ye12 = where_obst(x, y, x_obst12_start, x_obst12_end, y_obst12_start, y_obst12_end)
    xs13, xe13, ys13, ye13 = where_obst(x, y, x_obst13_start, x_obst13_end, y_obst13_start, y_obst13_end)
    xs14, xe14, ys14, ye14 = where_obst(x, y, x_obst14_start, x_obst14_end, y_obst14_start, y_obst14_end)
    xs15, xe15, ys15, ye15 = where_obst(x, y, x_obst15_start, x_obst15_end, y_obst15_start, y_obst15_end)
    xs16, xe16, ys16, ye16 = where_obst(x, y, x_obst16_start, x_obst16_end, y_obst16_start, y_obst16_end)
    xs17, xe17, ys17, ye17 = where_obst(x, y, x_obst17_start, x_obst17_end, y_obst17_start, y_obst17_end)
    xs18, xe18, ys18, ye18 = where_obst(x, y, x_obst18_start, x_obst18_end, y_obst18_start, y_obst18_end)
    xs19, xe19, ys19, ye19 = where_obst(x, y, x_obst19_start, x_obst19_end, y_obst19_start, y_obst19_end)
    # near Obstacle 2
    xs21, xe21, ys21, ye21 = where_obst(x, y, x_obst21_start, x_obst21_end, y_obst21_start, y_obst21_end)
    xs22, xe22, ys22, ye22 = where_obst(x, y, x_obst22_start, x_obst22_end, y_obst22_start, y_obst22_end)
    xs23, xe23, ys23, ye23 = where_obst(x, y, x_obst23_start, x_obst23_end, y_obst23_start, y_obst23_end)
    xs24, xe24, ys24, ye24 = where_obst(x, y, x_obst24_start, x_obst24_end, y_obst24_start, y_obst24_end)
    xs25, xe25, ys25, ye25 = where_obst(x, y, x_obst25_start, x_obst25_end, y_obst25_start, y_obst25_end)
    xs26, xe26, ys26, ye26 = where_obst(x, y, x_obst26_start, x_obst26_end, y_obst26_start, y_obst26_end)
    xs27, xe27, ys27, ye27 = where_obst(x, y, x_obst27_start, x_obst27_end, y_obst27_start, y_obst27_end)
    xs28, xe28, ys28, ye28 = where_obst(x, y, x_obst28_start, x_obst28_end, y_obst28_start, y_obst28_end)
    xs29, xe29, ys29, ye29 = where_obst(x, y, x_obst29_start, x_obst29_end, y_obst29_start, y_obst29_end)
    # near Obstacle 3
    xs31, xe31, ys31, ye31 = where_obst(x, y, x_obst31_start, x_obst31_end, y_obst31_start, y_obst31_end)
    xs32, xe32, ys32, ye32 = where_obst(x, y, x_obst32_start, x_obst32_end, y_obst32_start, y_obst32_end)
    xs33, xe33, ys33, ye33 = where_obst(x, y, x_obst33_start, x_obst33_end, y_obst33_start, y_obst33_end)
    xs34, xe34, ys34, ye34 = where_obst(x, y, x_obst34_start, x_obst34_end, y_obst34_start, y_obst34_end)
    xs35, xe35, ys35, ye35 = where_obst(x, y, x_obst35_start, x_obst35_end, y_obst35_start, y_obst35_end)
    xs36, xe36, ys36, ye36 = where_obst(x, y, x_obst36_start, x_obst36_end, y_obst36_start, y_obst36_end)
    xs37, xe37, ys37, ye37 = where_obst(x, y, x_obst37_start, x_obst37_end, y_obst37_start, y_obst37_end)
    xs38, xe38, ys38, ye38 = where_obst(x, y, x_obst38_start, x_obst38_end, y_obst38_start, y_obst38_end)
    xs39, xe39, ys39, ye39 = where_obst(x, y, x_obst39_start, x_obst39_end, y_obst39_start, y_obst39_end)
    # near Obstacle 4
    xs41, xe41, ys41, ye41 = where_obst(x, y, x_obst41_start, x_obst41_end, y_obst41_start, y_obst41_end)
    xs42, xe42, ys42, ye42 = where_obst(x, y, x_obst42_start, x_obst42_end, y_obst42_start, y_obst42_end)
    xs43, xe43, ys43, ye43 = where_obst(x, y, x_obst43_start, x_obst43_end, y_obst43_start, y_obst43_end)
    xs44, xe44, ys44, ye44 = where_obst(x, y, x_obst44_start, x_obst44_end, y_obst44_start, y_obst44_end)
    xs45, xe45, ys45, ye45 = where_obst(x, y, x_obst45_start, x_obst45_end, y_obst45_start, y_obst45_end)
    xs46, xe46, ys46, ye46 = where_obst(x, y, x_obst46_start, x_obst46_end, y_obst46_start, y_obst46_end)
    xs47, xe47, ys47, ye47 = where_obst(x, y, x_obst47_start, x_obst47_end, y_obst47_start, y_obst47_end)
    xs48, xe48, ys48, ye48 = where_obst(x, y, x_obst48_start, x_obst48_end, y_obst48_start, y_obst48_end)
    xs49, xe49, ys49, ye49 = where_obst(x, y, x_obst49_start, x_obst49_end, y_obst49_start, y_obst49_end)
    # near Obstacle 5
    xs51, xe51, ys51, ye51 = where_obst(x, y, x_obst51_start, x_obst51_end, y_obst51_start, y_obst51_end)
    xs52, xe52, ys52, ye52 = where_obst(x, y, x_obst52_start, x_obst52_end, y_obst52_start, y_obst52_end)
    xs53, xe53, ys53, ye53 = where_obst(x, y, x_obst53_start, x_obst53_end, y_obst53_start, y_obst53_end)
    xs54, xe54, ys54, ye54 = where_obst(x, y, x_obst54_start, x_obst54_end, y_obst54_start, y_obst54_end)
    xs55, xe55, ys55, ye55 = where_obst(x, y, x_obst55_start, x_obst55_end, y_obst55_start, y_obst55_end)
    xs56, xe56, ys56, ye56 = where_obst(x, y, x_obst56_start, x_obst56_end, y_obst56_start, y_obst56_end)
    xs57, xe57, ys57, ye57 = where_obst(x, y, x_obst57_start, x_obst57_end, y_obst57_start, y_obst57_end)
    xs58, xe58, ys58, ye58 = where_obst(x, y, x_obst58_start, x_obst58_end, y_obst58_start, y_obst58_end)
    xs59, xe59, ys59, ye59 = where_obst(x, y, x_obst59_start, x_obst59_end, y_obst59_start, y_obst59_end)
    # near Obstacle 6
    xs61, xe61, ys61, ye61 = where_obst(x, y, x_obst61_start, x_obst61_end, y_obst61_start, y_obst61_end)
    xs62, xe62, ys62, ye62 = where_obst(x, y, x_obst62_start, x_obst62_end, y_obst62_start, y_obst62_end)
    xs63, xe63, ys63, ye63 = where_obst(x, y, x_obst63_start, x_obst63_end, y_obst63_start, y_obst63_end)
    xs64, xe64, ys64, ye64 = where_obst(x, y, x_obst64_start, x_obst64_end, y_obst64_start, y_obst64_end)
    xs65, xe65, ys65, ye65 = where_obst(x, y, x_obst65_start, x_obst65_end, y_obst65_start, y_obst65_end)
    xs66, xe66, ys66, ye66 = where_obst(x, y, x_obst66_start, x_obst66_end, y_obst66_start, y_obst66_end)
    xs67, xe67, ys67, ye67 = where_obst(x, y, x_obst67_start, x_obst67_end, y_obst67_start, y_obst67_end)
    xs68, xe68, ys68, ye68 = where_obst(x, y, x_obst68_start, x_obst68_end, y_obst68_start, y_obst68_end)
    xs69, xe69, ys69, ye69 = where_obst(x, y, x_obst69_start, x_obst69_end, y_obst69_start, y_obst69_end)
    # near Obstacle 7
    xs71, xe71, ys71, ye71 = where_obst(x, y, x_obst71_start, x_obst71_end, y_obst71_start, y_obst71_end)
    xs72, xe72, ys72, ye72 = where_obst(x, y, x_obst72_start, x_obst72_end, y_obst72_start, y_obst72_end)
    xs73, xe73, ys73, ye73 = where_obst(x, y, x_obst73_start, x_obst73_end, y_obst73_start, y_obst73_end)
    xs74, xe74, ys74, ye74 = where_obst(x, y, x_obst74_start, x_obst74_end, y_obst74_start, y_obst74_end)
    xs75, xe75, ys75, ye75 = where_obst(x, y, x_obst75_start, x_obst75_end, y_obst75_start, y_obst75_end)
    xs76, xe76, ys76, ye76 = where_obst(x, y, x_obst76_start, x_obst76_end, y_obst76_start, y_obst76_end)
    xs77, xe77, ys77, ye77 = where_obst(x, y, x_obst77_start, x_obst77_end, y_obst77_start, y_obst77_end)
    xs78, xe78, ys78, ye78 = where_obst(x, y, x_obst78_start, x_obst78_end, y_obst78_start, y_obst78_end)
    xs79, xe79, ys79, ye79 = where_obst(x, y, x_obst79_start, x_obst79_end, y_obst79_start, y_obst79_end)
    # near Obstacle 8
    xs81, xe81, ys81, ye81 = where_obst(x, y, x_obst81_start, x_obst81_end, y_obst81_start, y_obst81_end)
    xs82, xe82, ys82, ye82 = where_obst(x, y, x_obst82_start, x_obst82_end, y_obst82_start, y_obst82_end)
    xs83, xe83, ys83, ye83 = where_obst(x, y, x_obst83_start, x_obst83_end, y_obst83_start, y_obst83_end)
    xs84, xe84, ys84, ye84 = where_obst(x, y, x_obst84_start, x_obst84_end, y_obst84_start, y_obst84_end)
    xs85, xe85, ys85, ye85 = where_obst(x, y, x_obst85_start, x_obst85_end, y_obst85_start, y_obst85_end)
    xs86, xe86, ys86, ye86 = where_obst(x, y, x_obst86_start, x_obst86_end, y_obst86_start, y_obst86_end)
    xs87, xe87, ys87, ye87 = where_obst(x, y, x_obst87_start, x_obst87_end, y_obst87_start, y_obst87_end)
    xs88, xe88, ys88, ye88 = where_obst(x, y, x_obst88_start, x_obst88_end, y_obst88_start, y_obst88_end)
    xs89, xe89, ys89, ye89 = where_obst(x, y, x_obst89_start, x_obst89_end, y_obst89_start, y_obst89_end)
    # near Obstacle 9
    xs91, xe91, ys91, ye91 = where_obst(x, y, x_obst91_start, x_obst91_end, y_obst91_start, y_obst91_end)
    xs92, xe92, ys92, ye92 = where_obst(x, y, x_obst92_start, x_obst92_end, y_obst92_start, y_obst92_end)
    xs93, xe93, ys93, ye93 = where_obst(x, y, x_obst93_start, x_obst93_end, y_obst93_start, y_obst93_end)
    xs94, xe94, ys94, ye94 = where_obst(x, y, x_obst94_start, x_obst94_end, y_obst94_start, y_obst94_end)
    xs95, xe95, ys95, ye95 = where_obst(x, y, x_obst95_start, x_obst95_end, y_obst95_start, y_obst95_end)
    xs96, xe96, ys96, ye96 = where_obst(x, y, x_obst96_start, x_obst96_end, y_obst96_start, y_obst96_end)
    xs97, xe97, ys97, ye97 = where_obst(x, y, x_obst97_start, x_obst97_end, y_obst97_start, y_obst97_end)
    xs98, xe98, ys98, ye98 = where_obst(x, y, x_obst98_start, x_obst98_end, y_obst98_start, y_obst98_end)
    xs99, xe99, ys99, ye99 = where_obst(x, y, x_obst99_start, x_obst99_end, y_obst99_start, y_obst99_end)

    # Defining the obstacles coordinates
    X1 = (xs1, xe1, ys1, ye1, xs11, xe11, ys11, ye11, xs12, xe12, ys12, ye12, xs13, xe13, ys13, ye13, xs14, xe14, ys14, ye14, xs15, xe15, ys15, ye15, xs16, xe16, ys16, ye16, xs17, xe17, ys17, ye17, xs18, xe18, ys18, ye18, xs19, xe19, ys19, ye19)
    X2 = (xs2, xe2, ys2, ye2, xs21, xe21, ys21, ye21, xs22, xe22, ys22, ye22, xs23, xe23, ys23, ye23, xs24, xe24, ys24, ye24, xs25, xe25, ys25, ye25, xs26, xe26, ys26, ye26, xs27, xe27, ys27, ye27, xs28, xe28, ys28, ye28, xs29, xe29, ys29, ye29)
    X3 = (xs3, xe3, ys3, ye3, xs31, xe31, ys31, ye31, xs32, xe32, ys32, ye32, xs33, xe33, ys33, ye33, xs34, xe34, ys34, ye34, xs35, xe35, ys35, ye35, xs36, xe36, ys36, ye36, xs37, xe37, ys37, ye37, xs38, xe38, ys38, ye38, xs39, xe39, ys39, ye39)
    X4 = (xs4, xe4, ys4, ye4, xs41, xe41, ys41, ye41, xs42, xe42, ys42, ye42, xs43, xe43, ys43, ye43, xs44, xe44, ys44, ye44, xs45, xe45, ys45, ye45, xs46, xe46, ys46, ye46, xs47, xe47, ys47, ye47, xs48, xe48, ys48, ye48, xs49, xe49, ys49, ye49)
    X5 = (xs5, xe5, ys5, ye5, xs51, xe51, ys51, ye51, xs52, xe52, ys52, ye52, xs53, xe53, ys53, ye53, xs54, xe54, ys54, ye54, xs55, xe55, ys55, ye55, xs56, xe56, ys56, ye56, xs57, xe57, ys57, ye57, xs58, xe58, ys58, ye58, xs59, xe59, ys59, ye59)
    X6 = (xs6, xe6, ys6, ye6, xs61, xe61, ys61, ye61, xs62, xe62, ys62, ye62, xs63, xe63, ys63, ye63, xs64, xe64, ys64, ye64, xs65, xe65, ys65, ye65, xs66, xe66, ys66, ye66, xs67, xe67, ys67, ye67, xs68, xe68, ys68, ye68, xs69, xe69, ys69, ye69)
    X7 = (xs7, xe7, ys7, ye7, xs71, xe71, ys71, ye71, xs72, xe72, ys72, ye72, xs73, xe73, ys73, ye73, xs74, xe74, ys74, ye74, xs75, xe75, ys75, ye75, xs76, xe76, ys76, ye76, xs77, xe77, ys77, ye77, xs78, xe78, ys78, ye78, xs79, xe79, ys79, ye79)
    X8 = (xs8, xe8, ys8, ye8, xs81, xe81, ys81, ye81, xs82, xe82, ys82, ye82, xs83, xe83, ys83, ye83, xs84, xe84, ys84, ye84, xs85, xe85, ys85, ye85, xs86, xe86, ys86, ye86, xs87, xe87, ys87, ye87, xs88, xe88, ys88, ye88, xs89, xe89, ys89, ye89)
    X9 = (xs9, xe9, ys9, ye9, xs91, xe91, ys91, ye91, xs92, xe92, ys92, ye92, xs93, xe93, ys93, ye93, xs94, xe94, ys94, ye94, xs95, xe95, ys95, ye95, xs96, xe96, ys96, ye96, xs97, xe97, ys97, ye97, xs98, xe98, ys98, ye98, xs99, xe99, ys99, ye99)
    X10 = (xs10, xe10, ys10, ye10)



# Function to define the indexes dor each obstacle
def defining_position(X):         # !!!!! n = number of sub-obstacles near the main one!!!!!

    n = (len(X) // 4) - 1
    if n == 0:
        xs = X[0]; xe = X[1]; ys = X[2]; ye = X[3]
        return xs,xe,ys,ye
    elif n == 1:
        xs = X[0]; xe = X[1]; ys = X[2]; ye = X[3]
        xs1 = X[4]; xe1 = X[5]; ys1 = X[6]; ye1 = X[7]
        return xs,xe,ys,ye, xs1,xe1,ys1,ye1
    elif n == 2:
        xs = X[0]; xe = X[1]; ys = X[2]; ye = X[3]
        xs1 = X[4]; xe1 = X[5]; ys1 = X[6]; ye1 = X[7]; xs2 = X[8]; xe2 = X[9]; ys2 = X[10]; ye2 = X[11]
        return xs,xe,ys,ye, xs1,xe1,ys1,ye1, xs2,xe2,ys2,ye2
    elif n == 3:
        xs = X[0]; xe = X[1]; ys = X[2]; ye = X[3]
        xs1 = X[4]; xe1 = X[5]; ys1 = X[6]; ye1 = X[7]; xs2 = X[8]; xe2 = X[9]; ys2 = X[10]; ye2 = X[11]
        xs3 = X[12]; xe3 = X[13]; ys3 = X[14]; ye3 = X[15]
        return xs,xe,ys,ye, xs1,xe1,ys1,ye1, xs2,xe2,ys2,ye2, xs3,xe3,ys3,ye3
    elif n == 4:
        xs = X[0]; xe = X[1]; ys = X[2]; ye = X[3]
        xs1 = X[4]; xe1 = X[5]; ys1 = X[6]; ye1 = X[7]; xs2 = X[8]; xe2 = X[9]; ys2 = X[10]; ye2 = X[11]
        xs3 = X[12]; xe3 = X[13]; ys3 = X[14]; ye3 = X[15]; xs4 = X[16]; xe4 = X[17]; ys4 = X[18]; ye4 = X[19]
        return xs,xe,ys,ye, xs1,xe1,ys1,ye1, xs2,xe2,ys2,ye2, xs3,xe3,ys3,ye3, xs4,xe4,ys4,ye4
    elif n == 5:
        xs = X[0]; xe = X[1]; ys = X[2]; ye = X[3]
        xs1 = X[4]; xe1 = X[5]; ys1 = X[6]; ye1 = X[7]; xs2 = X[8]; xe2 = X[9]; ys2 = X[10]; ye2 = X[11]
        xs3 = X[12]; xe3 = X[13]; ys3 = X[14]; ye3 = X[15]; xs4 = X[16]; xe4 = X[17]; ys4 = X[18]; ye4 = X[19]
        xs5 = X[20]; xe5 = X[21]; ys5 = X[22]; ye5 = X[23]
        return xs,xe,ys,ye, xs1,xe1,ys1,ye1, xs2,xe2,ys2,ye2, xs3,xe3,ys3,ye3, xs4,xe4,ys4,ye4, xs5,xe5,ys5,ye5
    elif n == 6:
        xs = X[0]; xe = X[1]; ys = X[2]; ye = X[3]
        xs1 = X[4]; xe1 = X[5]; ys1 = X[6]; ye1 = X[7]; xs2 = X[8]; xe2 = X[9]; ys2 = X[10]; ye2 = X[11]
        xs3 = X[12]; xe3 = X[13]; ys3 = X[14]; ye3 = X[15]; xs4 = X[16]; xe4 = X[17]; ys4 = X[18]; ye4 = X[19]
        xs5 = X[20]; xe5 = X[21]; ys5 = X[22]; ye5 = X[23]; xs6 = X[24]; xe6 = X[25]; ys6 = X[26]; ye6 = X[27]
        return xs,xe,ys,ye, xs1,xe1,ys1,ye1, xs2,xe2,ys2,ye2, xs3,xe3,ys3,ye3, xs4,xe4,ys4,ye4, xs5,xe5,ys5,ye5, xs6,xe6,ys6,ye6
    elif n == 7:
        xs = X[0]; xe = X[1]; ys = X[2]; ye = X[3]
        xs1 = X[4]; xe1 = X[5]; ys1 = X[6]; ye1 = X[7]; xs2 = X[8]; xe2 = X[9]; ys2 = X[10]; ye2 = X[11]
        xs3 = X[12]; xe3 = X[13]; ys3 = X[14]; ye3 = X[15]; xs4 = X[16]; xe4 = X[17]; ys4 = X[18]; ye4 = X[19]
        xs5 = X[20]; xe5 = X[21]; ys5 = X[22]; ye5 = X[23]; xs6 = X[24]; xe6 = X[25]; ys6 = X[26]; ye6 = X[27]
        xs7 = X[28]; xe7 = X[29]; ys7 = X[30]; ye7 = X[31]
        return xs,xe,ys,ye, xs1,xe1,ys1,ye1, xs2,xe2,ys2,ye2, xs3,xe3,ys3,ye3, xs4,xe4,ys4,ye4, xs5,xe5,ys5,ye5, xs6,xe6,ys6,ye6, xs7,xe7,ys7,ye7
    elif n == 8:
        xs = X[0]; xe = X[1]; ys = X[2]; ye = X[3]
        xs1 = X[4]; xe1 = X[5]; ys1 = X[6]; ye1 = X[7]; xs2 = X[8]; xe2 = X[9]; ys2 = X[10]; ye2 = X[11]
        xs3 = X[12]; xe3 = X[13]; ys3 = X[14]; ye3 = X[15]; xs4 = X[16]; xe4 = X[17]; ys4 = X[18]; ye4 = X[19]
        xs5 = X[20]; xe5 = X[21]; ys5 = X[22]; ye5 = X[23]; xs6 = X[24]; xe6 = X[25]; ys6 = X[26]; ye6 = X[27]
        xs7 = X[28]; xe7 = X[29]; ys7 = X[30]; ye7 = X[31]; xs8 = X[32]; xe8 = X[33]; ys8 = X[34]; ye8 = X[35]
        return xs,xe,ys,ye, xs1,xe1,ys1,ye1, xs2,xe2,ys2,ye2, xs3,xe3,ys3,ye3, xs4,xe4,ys4,ye4, xs5,xe5,ys5,ye5, xs6,xe6,ys6,ye6, xs7,xe7,ys7,ye7, xs8,xe8,ys8,ye8
    elif n == 9:
        xs = X[0]; xe = X[1]; ys = X[2]; ye = X[3]
        xs1 = X[4]; xe1 = X[5]; ys1 = X[6]; ye1 = X[7]; xs2 = X[8]; xe2 = X[9]; ys2 = X[10]; ye2 = X[11]
        xs3 = X[12]; xe3 = X[13]; ys3 = X[14]; ye3 = X[15]; xs4 = X[16]; xe4 = X[17]; ys4 = X[18]; ye4 = X[19]
        xs5 = X[20]; xe5 = X[21]; ys5 = X[22]; ye5 = X[23]; xs6 = X[24]; xe6 = X[25]; ys6 = X[26]; ye6 = X[27]
        xs7 = X[28]; xe7 = X[29]; ys7 = X[30]; ye7 = X[31]; xs8 = X[32]; xe8 = X[33]; ys8 = X[34]; ye8 = X[35]
        xs9 = X[36]; xe9 = X[37]; ys9 = X[38]; ye9 = X[39]
        return xs,xe,ys,ye, xs1,xe1,ys1,ye1, xs2,xe2,ys2,ye2, xs3,xe3,ys3,ye3, xs4,xe4,ys4,ye4, xs5,xe5,ys5,ye5, xs6,xe6,ys6,ye6, xs7,xe7,ys7,ye7, xs8,xe8,ys8,ye8, xs9,xe9,ys9,ye9


# Boundary conditions for velocities
def vel_walls(u, v, xs, xe, ys, ye):
        u[xs-1:xe+1, ye] = 2*uwall - u[xs-1:xe+1, ye+1]    # north face
        u[xs-1:xe+1, ys] = 2*uwall - u[xs-1:xe+1, ys-1]    # south face
        v[xs, ys-1:ye+1] = 2*uwall - v[xs-1, ys-1:ye+1]    # west  face
        v[xe, ys-1:ye+1] = 2*uwall - v[xe+1, ys-1:ye+1]    # east  face
        return u, v

u, v = vel_walls(u, v, xs1, xe1, ys1, ye1)         # Obstacle 1
u, v = vel_walls(u, v, xs2, xe2, ys2, ye2)         # Obstacle 2
u, v = vel_walls(u, v, xs3, xe3, ys3, ye3)         # Obstacle 3
u, v = vel_walls(u, v, xs4, xe4, ys4, ye4)         # Obstacle 4
u, v = vel_walls(u, v, xs5, xe5, ys5, ye5)         # Obstacle 5
u, v = vel_walls(u, v, xs6, xe6, ys6, ye6)         # Obstacle 6
u, v = vel_walls(u, v, xs7, xe7, ys7, ye7)         # Obstacle 7
u, v = vel_walls(u, v, xs8, xe8, ys8, ye8)         # Obstacle 8
u, v = vel_walls(u, v, xs9, xe9, ys9, ye9)         # Obstacle 9
u, v = vel_walls(u, v, xs10, xe10, ys10, ye10)     # Obstacle 10

u, v = vel_walls(u, v, xs11, xe11, ys11, ye11)         # near Obstacle 1
u, v = vel_walls(u, v, xs12, xe12, ys12, ye12)
u, v = vel_walls(u, v, xs13, xe13, ys13, ye13)
u, v = vel_walls(u, v, xs14, xe14, ys14, ye14)
u, v = vel_walls(u, v, xs15, xe15, ys15, ye15)
u, v = vel_walls(u, v, xs16, xe16, ys16, ye16)
u, v = vel_walls(u, v, xs17, xe17, ys17, ye17)
u, v = vel_walls(u, v, xs18, xe18, ys18, ye18)
u, v = vel_walls(u, v, xs19, xe19, ys19, ye19)
u, v = vel_walls(u, v, xs21, xe21, ys21, ye21)         # near Obstacle 2
u, v = vel_walls(u, v, xs22, xe22, ys22, ye22)
u, v = vel_walls(u, v, xs23, xe23, ys23, ye23)
u, v = vel_walls(u, v, xs24, xe24, ys24, ye24)
u, v = vel_walls(u, v, xs25, xe25, ys25, ye25)
u, v = vel_walls(u, v, xs26, xe26, ys26, ye26)
u, v = vel_walls(u, v, xs27, xe27, ys27, ye27)
u, v = vel_walls(u, v, xs28, xe28, ys28, ye28)
u, v = vel_walls(u, v, xs29, xe29, ys29, ye29)
u, v = vel_walls(u, v, xs31, xe31, ys31, ye31)         # near Obstacle 3
u, v = vel_walls(u, v, xs32, xe32, ys32, ye32)
u, v = vel_walls(u, v, xs33, xe33, ys33, ye33)
u, v = vel_walls(u, v, xs34, xe34, ys34, ye34)
u, v = vel_walls(u, v, xs35, xe35, ys35, ye35)
u, v = vel_walls(u, v, xs36, xe36, ys36, ye36)
u, v = vel_walls(u, v, xs37, xe37, ys37, ye37)
u, v = vel_walls(u, v, xs38, xe38, ys38, ye38)
u, v = vel_walls(u, v, xs39, xe39, ys39, ye39)
u, v = vel_walls(u, v, xs41, xe41, ys41, ye41)         # near Obstacle 4
u, v = vel_walls(u, v, xs42, xe42, ys42, ye42)
u, v = vel_walls(u, v, xs43, xe43, ys43, ye43)
u, v = vel_walls(u, v, xs44, xe44, ys44, ye44)
u, v = vel_walls(u, v, xs45, xe45, ys45, ye45)
u, v = vel_walls(u, v, xs46, xe46, ys46, ye46)
u, v = vel_walls(u, v, xs47, xe47, ys47, ye47)
u, v = vel_walls(u, v, xs48, xe48, ys48, ye48)
u, v = vel_walls(u, v, xs49, xe49, ys49, ye49)
u, v = vel_walls(u, v, xs51, xe51, ys51, ye51)         # near Obstacle 5
u, v = vel_walls(u, v, xs52, xe52, ys52, ye52)
u, v = vel_walls(u, v, xs53, xe53, ys53, ye53)
u, v = vel_walls(u, v, xs54, xe54, ys54, ye54)
u, v = vel_walls(u, v, xs55, xe55, ys55, ye55)
u, v = vel_walls(u, v, xs56, xe56, ys56, ye56)
u, v = vel_walls(u, v, xs57, xe57, ys57, ye57)
u, v = vel_walls(u, v, xs58, xe58, ys58, ye58)
u, v = vel_walls(u, v, xs59, xe59, ys59, ye59)
u, v = vel_walls(u, v, xs61, xe61, ys61, ye61)         # near Obstacle 6
u, v = vel_walls(u, v, xs62, xe62, ys62, ye62)
u, v = vel_walls(u, v, xs63, xe63, ys63, ye63)
u, v = vel_walls(u, v, xs64, xe64, ys64, ye64)
u, v = vel_walls(u, v, xs65, xe65, ys65, ye65)
u, v = vel_walls(u, v, xs66, xe66, ys66, ye66)
u, v = vel_walls(u, v, xs67, xe67, ys67, ye67)
u, v = vel_walls(u, v, xs68, xe68, ys68, ye68)
u, v = vel_walls(u, v, xs69, xe69, ys69, ye69)
u, v = vel_walls(u, v, xs71, xe71, ys71, ye71)         # near Obstacle 7
u, v = vel_walls(u, v, xs72, xe72, ys72, ye72)
u, v = vel_walls(u, v, xs73, xe73, ys73, ye73)
u, v = vel_walls(u, v, xs74, xe74, ys74, ye74)
u, v = vel_walls(u, v, xs75, xe75, ys75, ye75)
u, v = vel_walls(u, v, xs76, xe76, ys76, ye76)
u, v = vel_walls(u, v, xs77, xe77, ys77, ye77)
u, v = vel_walls(u, v, xs78, xe78, ys78, ye78)
u, v = vel_walls(u, v, xs79, xe79, ys79, ye79)
u, v = vel_walls(u, v, xs81, xe81, ys81, ye81)         # near Obstacle 8
u, v = vel_walls(u, v, xs82, xe82, ys82, ye82)
u, v = vel_walls(u, v, xs83, xe83, ys83, ye83)
u, v = vel_walls(u, v, xs84, xe84, ys84, ye84)
u, v = vel_walls(u, v, xs85, xe85, ys85, ye85)
u, v = vel_walls(u, v, xs86, xe86, ys86, ye86)
u, v = vel_walls(u, v, xs87, xe87, ys87, ye87)
u, v = vel_walls(u, v, xs88, xe88, ys88, ye88)
u, v = vel_walls(u, v, xs89, xe89, ys89, ye89)
u, v = vel_walls(u, v, xs91, xe91, ys91, ye91)         # near Obstacle 9
u, v = vel_walls(u, v, xs92, xe92, ys92, ye92)
u, v = vel_walls(u, v, xs93, xe93, ys93, ye93)
u, v = vel_walls(u, v, xs94, xe94, ys94, ye94)
u, v = vel_walls(u, v, xs95, xe95, ys95, ye95)
u, v = vel_walls(u, v, xs96, xe96, ys96, ye96)
u, v = vel_walls(u, v, xs97, xe97, ys97, ye97)
u, v = vel_walls(u, v, xs98, xe98, ys98, ye98)
u, v = vel_walls(u, v, xs99, xe99, ys99, ye99)



# Boundary conditions for species
def phi_obstacle(phi, xs, xe, ys, ye):
    phi[xs:xe+1, ys] = phi[xs:xe+1, ys-1];     # South wall
    phi[xs:xe+1, ye] = phi[xs:xe+1, ye+1];     # North wall
    phi[xs, ys:ye+1] = phi[xs-1, ys:ye+1];     # West wall
    phi[xe, ys:ye+1] = phi[xe+1, ys:ye+1];     # East wall
    return phi

# near 1st Obstacle BCS (Dirichlet)
phi = phi_obstacle(phi, xs11, xe11, ys11, ye11)   # near Obstacle 1
phi = phi_obstacle(phi, xs12, xe12, ys12, ye12)
phi = phi_obstacle(phi, xs13, xe13, ys13, ye13)
phi = phi_obstacle(phi, xs14, xe14, ys14, ye14)
phi = phi_obstacle(phi, xs15, xe15, ys15, ye15)
phi = phi_obstacle(phi, xs16, xe16, ys16, ye16)
phi = phi_obstacle(phi, xs17, xe17, ys17, ye17)
phi = phi_obstacle(phi, xs18, xe18, ys18, ye18)
phi = phi_obstacle(phi, xs19, xe19, ys19, ye19)
phi = phi_obstacle(phi, xs21, xe21, ys21, ye21)   # near Obstacle 2
phi = phi_obstacle(phi, xs22, xe22, ys22, ye22)
phi = phi_obstacle(phi, xs23, xe23, ys23, ye23)
phi = phi_obstacle(phi, xs24, xe24, ys24, ye24)
phi = phi_obstacle(phi, xs25, xe25, ys25, ye25)
phi = phi_obstacle(phi, xs26, xe26, ys26, ye26)
phi = phi_obstacle(phi, xs27, xe27, ys27, ye27)
phi = phi_obstacle(phi, xs28, xe28, ys28, ye28)
phi = phi_obstacle(phi, xs29, xe29, ys29, ye29)
phi = phi_obstacle(phi, xs31, xe31, ys31, ye31)   # near Obstacle 3
phi = phi_obstacle(phi, xs32, xe32, ys32, ye32)
phi = phi_obstacle(phi, xs33, xe33, ys33, ye33)
phi = phi_obstacle(phi, xs34, xe34, ys34, ye34)
phi = phi_obstacle(phi, xs35, xe35, ys35, ye35)
phi = phi_obstacle(phi, xs36, xe36, ys36, ye36)
phi = phi_obstacle(phi, xs37, xe37, ys37, ye37)
phi = phi_obstacle(phi, xs38, xe38, ys38, ye38)
phi = phi_obstacle(phi, xs39, xe39, ys39, ye39)
phi = phi_obstacle(phi, xs41, xe41, ys41, ye41)   # near Obstacle 4
phi = phi_obstacle(phi, xs42, xe42, ys42, ye42)
phi = phi_obstacle(phi, xs43, xe43, ys43, ye43)
phi = phi_obstacle(phi, xs44, xe44, ys44, ye44)
phi = phi_obstacle(phi, xs45, xe45, ys45, ye45)
phi = phi_obstacle(phi, xs46, xe46, ys46, ye46)
phi = phi_obstacle(phi, xs47, xe47, ys47, ye47)
phi = phi_obstacle(phi, xs48, xe48, ys48, ye48)
phi = phi_obstacle(phi, xs49, xe49, ys49, ye49)
phi = phi_obstacle(phi, xs51, xe51, ys51, ye51)   # near Obstacle 5
phi = phi_obstacle(phi, xs52, xe52, ys52, ye52)
phi = phi_obstacle(phi, xs53, xe53, ys53, ye53)
phi = phi_obstacle(phi, xs54, xe54, ys54, ye54)
phi = phi_obstacle(phi, xs55, xe55, ys55, ye55)
phi = phi_obstacle(phi, xs56, xe56, ys56, ye56)
phi = phi_obstacle(phi, xs57, xe57, ys57, ye57)
phi = phi_obstacle(phi, xs58, xe58, ys58, ye58)
phi = phi_obstacle(phi, xs59, xe59, ys59, ye59)
phi = phi_obstacle(phi, xs61, xe61, ys61, ye61)   # near Obstacle 6
phi = phi_obstacle(phi, xs62, xe62, ys62, ye62)
phi = phi_obstacle(phi, xs63, xe63, ys63, ye63)
phi = phi_obstacle(phi, xs64, xe64, ys64, ye64)
phi = phi_obstacle(phi, xs65, xe65, ys65, ye65)
phi = phi_obstacle(phi, xs66, xe66, ys66, ye66)
phi = phi_obstacle(phi, xs67, xe67, ys67, ye67)
phi = phi_obstacle(phi, xs68, xe68, ys68, ye68)
phi = phi_obstacle(phi, xs69, xe69, ys69, ye69)
phi = phi_obstacle(phi, xs71, xe71, ys71, ye71)   # near Obstacle 7
phi = phi_obstacle(phi, xs72, xe72, ys72, ye72)
phi = phi_obstacle(phi, xs73, xe73, ys73, ye73)
phi = phi_obstacle(phi, xs74, xe74, ys74, ye74)
phi = phi_obstacle(phi, xs75, xe75, ys75, ye75)
phi = phi_obstacle(phi, xs76, xe76, ys76, ye76)
phi = phi_obstacle(phi, xs77, xe77, ys77, ye77)
phi = phi_obstacle(phi, xs78, xe78, ys78, ye78)
phi = phi_obstacle(phi, xs79, xe79, ys79, ye79)
phi = phi_obstacle(phi, xs81, xe81, ys81, ye81)   # near Obstacle 8
phi = phi_obstacle(phi, xs82, xe82, ys82, ye82)
phi = phi_obstacle(phi, xs83, xe83, ys83, ye83)
phi = phi_obstacle(phi, xs84, xe84, ys84, ye84)
phi = phi_obstacle(phi, xs85, xe85, ys85, ye85)
phi = phi_obstacle(phi, xs86, xe86, ys86, ye86)
phi = phi_obstacle(phi, xs87, xe87, ys87, ye87)
phi = phi_obstacle(phi, xs88, xe88, ys88, ye88)
phi = phi_obstacle(phi, xs89, xe89, ys89, ye89)
phi = phi_obstacle(phi, xs91, xe91, ys91, ye91)   # near Obstacle 9
phi = phi_obstacle(phi, xs92, xe92, ys92, ye92)
phi = phi_obstacle(phi, xs93, xe93, ys93, ye93)
phi = phi_obstacle(phi, xs94, xe94, ys94, ye94)
phi = phi_obstacle(phi, xs95, xe95, ys95, ye95)
phi = phi_obstacle(phi, xs96, xe96, ys96, ye96)
phi = phi_obstacle(phi, xs97, xe97, ys97, ye97)
phi = phi_obstacle(phi, xs98, xe98, ys98, ye98)
phi = phi_obstacle(phi, xs99, xe99, ys99, ye99)



# Defining obstacles near corners
xs1,xe1,ys1,ye1,xs11,xe11,ys11,ye11,xs12,xe12,ys12,ye12,xs13,xe13,ys13,ye13,xs14,xe14,ys14,ye14,xs15,xe15,ys15,ye15,xs16,xe16,ys16,ye16,xs17,xe17,ys17,ye17,xs18,xe18,ys18,ye18,xs19,xe19,ys19,ye19 = defining_position(9, X1)
xs2,xe2,ys2,ye2,xs21,xe21,ys21,ye21,xs22,xe22,ys22,ye22,xs23,xe23,ys23,ye23,xs24,xe24,ys24,ye24,xs25,xe25,ys25,ye25,xs26,xe26,ys26,ye26,xs27,xe27,ys27,ye27,xs28,xe28,ys28,ye28,xs29,xe29,ys29,ye29 = defining_position(9, X2)
xs3,xe3,ys3,ye3,xs31,xe31,ys31,ye31,xs32,xe32,ys32,ye32,xs33,xe33,ys33,ye33,xs34,xe34,ys34,ye34,xs35,xe35,ys35,ye35,xs36,xe36,ys36,ye36,xs37,xe37,ys37,ye37,xs38,xe38,ys38,ye38,xs39,xe39,ys39,ye39 = defining_position(9, X3)
xs4,xe4,ys4,ye4,xs41,xe41,ys41,ye41,xs42,xe42,ys42,ye42,xs43,xe43,ys43,ye43,xs44,xe44,ys44,ye44,xs45,xe45,ys45,ye45,xs46,xe46,ys46,ye46,xs47,xe47,ys47,ye47,xs48,xe48,ys48,ye48,xs49,xe49,ys49,ye49 = defining_position(9, X4)
xs5,xe5,ys5,ye5,xs51,xe51,ys51,ye51,xs52,xe52,ys52,ye52,xs53,xe53,ys53,ye53,xs54,xe54,ys54,ye54,xs55,xe55,ys55,ye55,xs56,xe56,ys56,ye56,xs57,xe57,ys57,ye57,xs58,xe58,ys58,ye58,xs59,xe59,ys59,ye59 = defining_position(9, X5)
xs6,xe6,ys6,ye6,xs61,xe61,ys61,ye61,xs62,xe62,ys62,ye62,xs63,xe63,ys63,ye63,xs64,xe64,ys64,ye64,xs65,xe65,ys65,ye65,xs66,xe66,ys66,ye66,xs67,xe67,ys67,ye67,xs68,xe68,ys68,ye68,xs69,xe69,ys69,ye69 = defining_position(9, X6)
xs7,xe7,ys7,ye7,xs71,xe71,ys71,ye71,xs72,xe72,ys72,ye72,xs73,xe73,ys73,ye73,xs74,xe74,ys74,ye74,xs75,xe75,ys75,ye75,xs76,xe76,ys76,ye76,xs77,xe77,ys77,ye77,xs78,xe78,ys78,ye78,xs79,xe79,ys79,ye79 = defining_position(9, X7)
xs8,xe8,ys8,ye8,xs81,xe81,ys81,ye81,xs82,xe82,ys82,ye82,xs83,xe83,ys83,ye83,xs84,xe84,ys84,ye84,xs85,xe85,ys85,ye85,xs86,xe86,ys86,ye86,xs87,xe87,ys87,ye87,xs88,xe88,ys88,ye88,xs89,xe89,ys89,ye89 = defining_position(9, X8)
xs9,xe9,ys9,ye9,xs91,xe91,ys91,ye91,xs92,xe92,ys92,ye92,xs93,xe93,ys93,ye93,xs94,xe94,ys94,ye94,xs95,xe95,ys95,ye95,xs96,xe96,ys96,ye96,xs97,xe97,ys97,ye97,xs98,xe98,ys98,ye98,xs99,xe99,ys99,ye99 = defining_position(9, X9)
xs10,xe10,ys10,ye10 = defining_position(0, X10)