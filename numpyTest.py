import numpy as np
import time
for i in range (20):
    arr1 = np.arange (0, 1000000)
    arr2 = np.arange (1000000, 2000000)
    outputArr = np.zeros ((1000000,))
    start = time.time ()
    for i in range (1000000):
        outputArr [i] = ((arr1[i]) -(arr2[i]))/ (arr1[i]+arr2[i])
    end = time.time ()

    print (end-start)
