import cupy as cp
import numpy as np
import time

array_cpu = np.random.randint (0, 100, size=(500, 500))
array_gpu = cp.asarray (array_cpu)

start = time.time ()
np.matmul (array_cpu, array_cpu)
end = time.time ()
print (end-start)

mat_mul = cp.ElementwiseKernel(
   'T x, T y',
   'T z',
   'z = x * y',
   'matmul')

start = time.time ()
mat_mul (array_gpu, array_gpu)
end = time.time ()
print (end-start)

mat_mul2 = cp.ElementwiseKernel(
   'T x, T y',
   'T z',
   'z = x * y',
   'matmul')

