import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
import matplotlib as mpl
from scipy import stats
import sys
import pandas as pd
import datetime
import math
from os.path import exists
import cupy as cp
from numba import cuda, int16, float32, int32
import numba
from numpy import e, pi
import math 
import time

@numba.cuda.jit
def computation (zarray2):
    i = cuda.threadIdx.x
    j = cuda.blockIdx.x
    k = cuda.blockIdx.y

    zarray2[i, j, k]+=1

N = 250
frn = 200

threads_per_block = N
blocks_per_grid = N,frn

zarray2=np.zeros((N,N,frn), dtype = np.int16) #max value is 32767

# computation[blocks_per_grid, threads_per_block](x_gpu, y_gpu, rnge, increment, distarray_gpu, N, zburst_gpu, y_data_gpu, alpha, E_gpu, sigma_gpu, zarray2_gpu, pi,index)
computation [blocks_per_grid, threads_per_block] (zarray2)
print (zarray2)