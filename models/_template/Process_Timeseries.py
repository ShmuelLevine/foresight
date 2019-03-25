#!/bin/python3

import TS_funcs as f
import sys

if len(sys.argv) != 1:
    print "Invalid arguments passed to this script."
    print "Usage: ./Process_Timeseries.py [timeseries_path]"
    sys.exit( 128)


ts_path = r"/home/shmuel/src/fx_hpx/src/Jul2005-Jul2008.csv"
out_path = r"./Matrix.out"

ts = f.Read_Timeseries(ts_path)

first_point = 200
forecast_horizon = 5

total_length = len(ts) - first_point - forecast_horizon

num_inputs = 32

Matrix = [[0 for x in range(total_length)] for y in range(num_inputs)]

row_idx = 0
MA_lengths = [5, 10, 25, 50, 100, 150]
lags = [1,2,3,4,5,10,20,50,100,150,200]
diffs = [1,2,3,4,5,10,20,50]
diffs2 = [1,2,3,4,5]

print (len(Matrix))
print (len(Matrix[0]))

lst = [i for i in range(8)]
print(lst)
print ("total_length: ", total_length)

# Prepare moving averages
MA = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

for m,j in zip(MA_lengths,range(len(MA_lengths))):
    for i in range(m):
        MA[j] += ts[first_point - i - 1]

# Prepare RSI
ewma_alpha = 0.35; # Damping factor
RSI_length = 14
a1 = 1.0 - ewma_alpha;
a2 = a1 * a1
a3 = a2 * a1
a4 = a3 * a1
a5 = a4 * a1

RSI_EWMA = [pow(a1,exp) for exp in range(RSI_length)]

def RSI_pos(v):
    if v > 0.0:
        return v
    return 0.0

def RSI_neg(v):
    if v < 0.0:
        return -v
    return 0.0


col = 0

## Fill rows 0 - 5 with Moving Averages
for cur_idx in range(first_point,total_length):
    row = 0
    for m,j in zip(MA_lengths,range(len(MA_lengths))):
        MA[j] -= ts[cur_idx - 1 - m]
        MA[j] += ts[cur_idx]
        Matrix[row][col] = MA[j]/float(m)
        row += 1

# Fill rows 6 - 16 - Lags
    for l in lags:
        Matrix[row][col] = ts[cur_idx - l]
        row += 1

# Fill rows 17 - 24 - Diffs
    for d in diffs:
        Matrix[row][col] = ts[cur_idx - d] - ts[cur_idx - d - 1]
        row += 1

# Fill rows 25-29 - 2nd Diffs
    for d in diffs2:
        Matrix[row][col] = ts[cur_idx - d] - 2 * ts[cur_idx - d - 1] + ts[cur_idx - d - 2]
        row += 1

# Fill row 30 - RSI

    RSI_U = [RSI_pos(ts[cur_idx - a] - ts[cur_idx-a-1]) for a in range(RSI_length)]
    RSI_D = [RSI_neg(ts[cur_idx - a] - ts[cur_idx - a -1]) for a in range(RSI_length)]

    RS_U = sum(i * j for i,j in zip(RSI_U,RSI_EWMA))
    RS_D = sum(i * j for i,j in zip(RSI_D,RSI_EWMA)) + 0.00000000000000001 # to prevent div/0

    Matrix[row][col] = 1 - (1 / (1 + (RS_U/RS_D)))
    row += 1
# Fill row 31 - ANN Bias

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
of.write(str(first_point) + "\n")
of.write(str(len(Matrix)) + " " + str(len(Matrix[0])) + "\n")

for i in range(len(Matrix)):
    for j in range(len(Matrix[0])):
        of.write(str(Matrix[i][j]) + " ")
    of.write('\n')

of.close()
