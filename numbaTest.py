from numba import cuda
import numpy as np
import cupy as cp
import time
# # print (cuda.detect())

# array_cpu = np.random.randint (0, 10, size=(2000, 2000))

# d_array = cuda.to_device (array_cpu) #sends data to GPU
# print (d_array)

# cp.asarray (d_array)
# d_array.copy_to_host () #sends data back to CPU as numpy array

@cuda.jit
def matmul (A, B, C):
    i, j = cuda.grid (2)
    if i < C.shape [0] and j < C.shape [1]:
        tmp = 0.
        for k in range (A.shape [1]):
            tmp+=A[i, k] * B[k, j]
        C[i, j] = tmp

A = np.random.uniform (1, 10, size= (10000, 10000)) #, dtype=np.float64
B = np.random.uniform (1, 10, size= (10000, 10000))
C = np.zeros ((10000, 10000))

threadsperblock = (16, 16) #each block has 16X 16 threads (between 128-512)
blockspergrid_x = int (np.ceil (C.shape [0]/threadsperblock[0]))
blockspergrid_y = int (np.ceil (C.shape [1]/threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

print (blockspergrid)


A_gpu = cp.asarray (A)
B_gpu = cp.asarray (B)
C_gpu = cp.zeros ((2000, 2000))
start = time.time ()
matmul [blockspergrid, threadsperblock] (A_gpu, B_gpu, C_gpu)
end = time.time ()
print ("Numba times with cupy: ", end-start)

start = time.time ()
matmul [blockspergrid, threadsperblock] (A, B, C)
end = time.time ()
print ("Numba times with numpy: ", end-start)
# start = time.time ()
# C = np.matmul (A, B)
# end = time.time ()
# print ("Numpy times with numpy: " , end-start)

# start = time.time ()
# C = np.matmul (A_gpu, B_gpu)
# end = time.time ()
# print ("Numpy times with cupy: " , end-start)

start = time.time ()
C = cp.matmul (A_gpu, B_gpu)
end = time.time ()
print ("Cupy Time #", 1, ": ", end-start)

start = time.time ()
C = cp.matmul (A_gpu, B_gpu)
end = time.time ()
print ("Cupy Time #", 2, ": ", end-start)