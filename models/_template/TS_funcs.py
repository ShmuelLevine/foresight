#!/bin/python3

def Read_Timeseries(file_path):
    ts = [0.0];
    with open(file_path) as f:
        for line in f:
            ts.append(float(line));

    return ts;


def Lag(ts, degree, current_idx):
    return ts[current_idx - degree];

def Diff(ts, d, current_idx):
    return ts[current_idx - d] - ts[current_idx - d - 1]

def Diff2(ts, d, current_idx):
    ts_0 = ts[current_idx - 0 - d];
    ts_1 = ts[current_idx - 1 - d]
    ts_2 = ts[current_idx - 2 - d]
    return ts_0 - 2 * ts_1 + ts_2;

def MA(ts, points, current_idx):
    return sum([ts[current_idx -i] for i in range(points)])/ points


def RSI(ts, length, current_idx):
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

    RSI_U = [RSI_pos((ts[current_idx - a] - ts[current_idx-a-1]) for a in range(RSI_length))]
    RSI_D = [RSI_neg((ts[current_idx - a] - ts[current_idx - a -1]) for a in range(RSI_length))]

    RS_U = sum(i * j for i,j in zip(RSI_U,RSI_U))
    RS_U = sum(i * j for i,j in zip(RSI_U,RSI_U)) + 0.00000000000000001 # to prevent div/0

    return 1 - (1 / ( 1 + RS_U / RS_D))
