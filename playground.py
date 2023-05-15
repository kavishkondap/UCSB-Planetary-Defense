# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation 
# import matplotlib as mpl
# from scipy import stats
# import sys
# import pandas as pd
# import datetime
# import math
# from os.path import exists
# import cupy as cp
# from numba import cuda
# # import nvidia_smi


# # nvidia_smi.nvmlInit()


# # # GPU_MEMORY = 4000000000 # in bytes
# # # mempool = cp.get_default_memory_pool()
# # # mempool.set_limit(size=3.5*1024**3)
# # # a=["a","b","c","d","e","f","g","h","i",'j','k','l','m']
# # a=["100m_in_2000_1d","1000","2000","200","THIS WAS THE ERROR","100m_in_2000_1d","0","0","100","2000","100m","in","2000-Fragments(1Day)"]
# # # for input in sys.stdin: # take inputs from a shell script
# # #     input1 = input.strip()
# # #     print(input1)
# # #     b,c,d,e,f,g,h,i,j,k,l,m,n = input1.split()
# # #     a[0] = b
# # #     a[1] = c
# # #     a[2] = d
# # #     a[3] = e
# # #     a[4] = f
# # #     a[5] = g
# # #     a[6] = h
# # #     a[7] = i
# # #     a[8] = j
# # #     a[9] = k
# # #     a[10] = l
# # #     a[11] = m
# # #     a[12] = n
# # name = a[0]  #filename of the input bin files
# # N = int(a[1]) #This is the dimensions of any particular frame (always the same as XDIM from cpp)
# # frn = int(a[2]) #This is the total number of frames in the simulation (has to be the same as frn from cpp) 
# # fps= 10 #sets the fps (This has to be the same as fps from cpp) 
# # rnge = int(a[3]) #The Range of the plots. SO if a plot covers -200 to 200 km on the x axis, this would be 200 
# # title = a[4] #The Title of the plots
# # outname = a[5] #The Name of the output files
# # xcenter = int(a[6]) # Enter the x component of the center
# # ycenter = int(a[7]) # Enter the y component of the center
# # timesleep = int(a[8]) #The time computer sleeps before starting the program. Can use this to run this simultaneously with the cpp file 
# # NOI = int(a[9])
# # title1 = a[10]
# # title2 = a[11]
# # title3 = a[12]
# # alpha = 0.1
# # fps = 10
# # increment = 2*rnge/N
# # xticks = cp.linspace(xcenter-rnge,xcenter+rnge,5,endpoint = True)
# # print (xticks.shape)
# # print (xticks)
# # yticks = cp.linspace(ycenter-rnge,ycenter+rnge,5,endpoint=True)
# # print (yticks.shape)
# # print (yticks)
# # xls = pd.ExcelFile(name+".xlsx", engine = "openpyxl")
# # df = pd.read_excel(xls, 'Ring Fragment Bombardment -sort')
# # x = df['OUTPUT - Asteroid Position x(km)'].to_numpy()
# # y = df['OUTPUT - Asteroid Position y(km)'].to_numpy()
# # # print (type (x))
# # # y = cp.array (np.delete(y,[0,1,2]), dtype= np.float64)
# # sigma = df['Sigma for Gaussian (in time) optical pulse power (s)'].to_numpy()
# # # sigma = cp.array (np.delete(sigma,[0,1,2]), dtype = np.float64)
# # E = df['OUTPUT.28'].to_numpy()
# # # E = cp.array (np.delete(E,[0,1,2]), dtype = np.float64)
# # zburst = df['OUTPUT.22'].to_numpy()
# # times = df['OUTPU'].to_numpy()
# # # zburst = cp.array (np.delete(zburst,[0,1,2]), dtype = np.float64)
# # # times = cp.array(np.delete(times,[0,1,2]), dtype = np.float64)
# # hundredpower=df['P_opt_atm (w/m^2) - no projection - horizon cut - with atm transmission - assuming 100% conversion of blast to optical'].to_numpy()
# # # hundredpower = cp.array(np.delete(hundredpower,[0,1,2]), dtype = np.float64)
# # x = np.delete (x, [0, 1, 2])
# # y = np.delete (y, [0, 1, 2])
# # sigma = np.delete (sigma, [0, 1, 2])
# # E = np.delete (E, [0, 1, 2])
# # zburst = np.delete (zburst, [0, 1, 2])
# # times = np.delete (times, [0, 1, 2])
# # print ('N: ', N)
# # print ('frn: ', frn)
# # times = times[:NOI]
# # mindisp = times.min()
# # if mindisp < 0:
# #     times = times - mindisp + 5
# # else:
# #     times = times - mindisp + 5


# # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
# # # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

# # info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

# # print("Total memory:", info.total)
# # print("Free memory:", info.free)
# # print("Used memory:", info.used)

# # nvidia_smi.nvmlShutdown()

# # zarray2 = np.zeros((N,N,frn))
# # print (zarray2.nbytes)


# @cuda.jit
# def computation (zarray2, NOI, x_gpu, y_gpu, rnge, increment, distarray, N, zburst_gpu, b, y_data, alpha, E_gpu, sigma_gpu):
#     index = cuda.grid (1)
#     if index < NOI:
#         xpos = int((x_gpu[index]+rnge)/increment)
#         ypos = int((y_gpu[index]+rnge)/increment)
#         temparr = distarray[-(N+ypos):,N-xpos:]
#         temparr = temparr[:N,:N]
#         temparr = 1/temparr
#         temparr += zburst_gpu[index]**2
#         temparr = 1/temparr
#         print ("TEMPARR: ", temparr.shape)
#         alpha_a = b*cp.exp(b*temparr)
#         zarray2[:,:,:] += y_data[index] * temparr[...,None] * alpha * E_gpu[index] * 1/(4*cp.pi) * 1/(cp.sqrt(2*cp.pi)*sigma_gpu[index]) * alpha_a[...,None]
import numpy as np
temparr = [[1, 2, 3, 4], [5, 6, 7, 8]]
temparr2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
temparr4 = [[1, 2, 3, 4], [5, 6, 7, 8]]
temparr = np.asarray (temparr)
temparr2 = np.asarray (temparr2)
temparr4 = np.asarray (temparr4)
# print (temparr)
temparr3 = temparr [..., None]*temparr2[..., None] * temparr4[0]
print (temparr3.shape)
print (temparr3)

#currNum = y_data[index] * temparr[...,None] * alpha * E_gpu[index] * 1/(4*pi) * 1/(sqrt(2*pi)*sigma_gpu[index]) * alpha_a[...,None]
# @numba.guvectorize (['(int16[:, :, :], float32 [:, :], int32, float32 [:], float32 [:], int32, int32, float32 [:,:], int32, float32 [:], int32, float32 [:], int32, float32 [:], float32 [:])'], '(n),(n)->(n)', target = 'cuda')#assign a thread on GPU to each value in zarray2
# def computation (zarray2, alpha_a, NOI, x_gpu, y_gpu, rnge, increment, distarray, N, zburst_gpu, b, y_data, alpha, E_gpu, sigma_gpu):



# import numpy as np
# from numpy import pi, e, sqrt
# N = 250

# temparr = np.zeros ((N, N))
# y_data = np.zeros ((2000, 2000))
# alpha = 0.1
# E_gpu = np.zeros ((1997, ))
# sigma_gpu = np.zeros ((1997, ))
# alpha_a = np.zeros ((250, 250))
# index = 0
# currNum = y_data[index] * temparr[...,None] * alpha * E_gpu[index] * 1/(4*pi) * 1/(sqrt(2*pi)*sigma_gpu[index]) * alpha_a[...,None]
# print (currNum)
# print (currNum.shape)