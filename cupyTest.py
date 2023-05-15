import cupy as cp
import numpy as np
import time

array_cpu = np.random.randint (0, 255, size=(5000, 5000))
array_gpu = cp.asarray(array_cpu) #sends data to gpu from cpu, which takes more time than cp.random.randint (0, 255, size = (5000, 5000))

from scipy import fft

from cupyx.scipy import fft as fft_gpu
for i in range (20):
    start = time.time ()
    fft_gpu.fftn (array_gpu)
    end = time.time()
    print ("CuPy FFT Time", (end-start))

    start = time.time ()
    fft.fftn (array_cpu)
    end = time.time()
    print ("NumPy FFT Time", (end-start))

fft_cpu = fft.fftn(array_cpu)
fft_sent_back = cp.asnumpy (fft_gpu.fftn (array_gpu))
print (np.allclose (fft_sent_back, fft_cpu))
