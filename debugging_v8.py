# void core(float* zarray2, int index, int fps, int intslice, float* timeofburst, float* E, float* gaussiandata, float* stacked_result) // This function takes the inslice variable and processes the NUM_OF_DATA_PER_THRD of x slices applying the Friedlander Equation for any particular Blast Wave. This is all determined when we call the threads in main()
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
def computation (index, zarray2, stacked_result, E, sigma, times, fps, x, y, zburst, e, pi):
    tval = times[index]+2
    dummy = tval+fps
    difference = 3*0.1*fps
    start = int (dummy-difference)
    end = int (dummy+difference)
    if (end>zarray2.shape [2]):
        end = zarray2.shape [2]
    for j in range (intslice, intslice+NUM_DATA_PER_THRD):
        for h in range (YDIM):
            sl = 1000000 * ((x[j]-xpositionofburst [index])**2 + (y[h] - ypositionofburst[index])**2 + zburst**2)
            amp = E[index]
            atmospheric = 1.02 * e**(-(0.0099*(sl**1/2)/1000))
            eflux = 0.1*amp*1/(sl)*1/ (4*pi)*atmospheric
            for i in range (start, end):
                d = sigma [i-start]*eflux*1/(0.1*(2*pi)**1/2)
                zarray2 [int (i*XDIM*YDIM+h*YDIM+j)] +=d
                if (i == int (dummy) and not eflux == 0):
                    stacked_result [int (i*XDIM*YDIM+h*YDIM+j)] += eflux

# The following Python variables need to fit in somewhere:

# y_data
# hundredpower
# distarray