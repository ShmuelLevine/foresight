#!/usr/bin/python3

import sys
import TS_funcs as f

if len(sys.argv) != 3:
    print ("Invalid arguments passed to this script.")
    print ("Usage: ./Process_Timeseries.py [timeseries_path] [output_matrix_path]")
    sys.exit( 128)


#ts_path = r"/home/shmuel/src/fx_hpx/src/Jul2005-Jul2008.csv"
ts_path = sys.argv[1]
out_path = sys.argv[2]

ts = f.Read_Timeseries(ts_path)


# For this model, use the following inputs:
'''
1: L(0)
2: L(1)
3: L(2)
4: L(3)
5: L(4)
6: D(1)
7: MA(4)
8: EWMA(4)
'''

first_point = 200
forecast_horizon = 5

Lags = [0,1,2,3,4]
MAs = [4]

total_length = len(ts) - first_point - forecast_horizon

num_inputs = 8

# The number of rows is equal to num_inputs + 1, to allow for 1 additional
# row for the bias inputs
Matrix = [[0 for x in range(total_length)] for y in range(num_inputs + 1)]

row_idx = 0


# Prepare moving averages
# Since the main loop advances the index and adjusts the MA variable
# by adding the newest point and subtracting the oldest point, set-up
# the variable to match the state as-if this was for first_point-1:
MA = 0.0
for idx in range(4):
    MA += ts[first_point - 1 - idx - 1]

# Prepare EWMA
ewma_alpha = 0.35; # Damping factor
EWMA_length = 4
a1 = 1.0 - ewma_alpha;

EWMA_coeffs = [pow(a1,exp) for exp in range(1,EWMA_length+1)]


col = 0


for cur_idx in range(first_point - 1,total_length - 1):
    row = 0

    ## Fill rows 0 - 4 with Lags
    for l in Lags:
        Matrix[row][col] = ts[cur_idx - l]
        row += 1

    ## Set row 5 to D(1)
    Matrix[row][col] = ts[cur_idx] - ts[cur_idx - 1]
    row += 1

    ## Update MA and set row 6 to MA
    MA -= ts[cur_idx - MAs[0] - 1]
    MA += ts[cur_idx]
    Matrix[row][col] = MA / float(MAs[0])
    row += 1

    ## Calculate EWMA and set row 7
    # Get the index and co-efficient for each term in the exponentially-weighted MA
    EWMA = 0.0
    for C,idx in zip(EWMA_coeffs, range(len(EWMA_coeffs))):
        EWMA += ts[cur_idx - idx] * C
    Matrix[row][col] = EWMA
    row += 1

    # Fill row 8 - ANN Bias
    Matrix[row][col] = 1.0
    col += 1



## Output to file
# Format:
#   timeseries_path
#   initial_time
#   Matrix_dims
#   Matrix ... Data...
#   ...    ...     ...

of = open(out_path, 'w')
of.write(ts_path + "\n")
of.write("T_0 = " + str(first_point) + "\n")
of.write(str(len(Matrix)) + " " + str(len(Matrix[0])) + "\n")

for i in range(len(Matrix)):
    for j in range(len(Matrix[0])):
        of.write(str(Matrix[i][j]) + " ")
    of.write('\n')

of.close()
