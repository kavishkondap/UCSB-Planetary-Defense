import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
import matplotlib as mpl
from scipy import stats
import sys
import pandas as pd
import datetime
import math
import time
from os.path import exists
import multiprocessing as mp
from multiprocessing import shared_memory
maxload = 10
maxloadcalc = 100
lock = mp.Lock()
dtype = np.float64
def makeblasts(shr_name, index):
    if index < NOI:
        existing_shm = shared_memory.SharedMemory(name=shr_name)
        xpos = int((x[index]+rnge)/increment)
        ypos = int((y[index]+rnge)/increment)
        temparr = distarray[-(N+ypos):,N-xpos:]
        temparr = temparr[:N,:N]
        temparr = 1/temparr
        temparr += zburst[index]**2
        temparr = 1/temparr
        alpha_a = b*np.exp(a*temparr)
        newarr = y_data[index]
        start = np.min(np.nonzero(newarr))
        end = np.max(np.nonzero(newarr))
        newarr = newarr[start:end]
        arr = newarr * temparr[...,None] * alpha * E[index] * 1/(4*np.pi) * 1/(np.sqrt(2*np.pi)*sigma[index]) * alpha_a[...,None]
        zarr = np.ndarray((N,N,frn),dtype = dtype, buffer = existing_shm.buf)        
        lock.acquire()
        zarr[:,:,start:end] += arr
        del arr
        lock.release()
        del temparr
        del xpos
        del ypos
        existing_shm.close()
        if index % 10 == 0:
            print(index)
def update_plot1(i):
    ax1.cla()
    time = i/fps - 5
    time = format(time, '#.3f')
    fig.suptitle(title1 + ' Asteroid ' + title2 + ' ' + title3 + ' time t = ' + str(time) + ' seconds after first fragment bursts',fontsize=20)
    ax1.imshow(zarray2[:,:,i],cmap = cmap1, norm = norm1,origin='lower')
    #fig.colorbar(mpl.cm.ScalarMappable(norm = norm1, cmap=cmap2), ax = ax1, label ="Optical Power Flux(W/m^2)")
    ax1.set_xticks(np.linspace(0,N,5))
    ax1.set_xticklabels(xticks) 
    ax1.set_yticks(np.linspace(0,N,5))
    ax1.set_yticklabels(yticks)
    ax1.set_xlabel('x-axis(km)')
    ax1.set_ylabel('y-axis(km)')
    ax1.set_title('Optical Power Flux Distribution in W/m^2')
    ax2.cla()
    ax2.set_title('Optical Energy Flux Distribution in J/m^2')
    ax2.set_xticks(np.linspace(0,N,5))
    ax2.set_xticklabels(xticks) 
    ax2.set_yticks(np.linspace(0,N,5))
    ax2.set_yticklabels(yticks)
    ax2.imshow(Energyarray[:,:,i],norm=norm2,cmap=cmap1,origin='lower')    
    ax2.set_xlabel('x-axis(km)')
    ax2.set_ylabel('y-axis(km)')
    ax3.cla()
    ax3.set_xlabel('Time(s) after first fragment burst')
    ax3.set_ylabel('Optical Power Flux(W/m^2)')
    ax3.set_title('Line chart showing the real time max power \n')
    if (i)/10 - 5 < 0:
        ax3.plot(np.linspace(-5,i/fps - 5,i), maxpower[0:i],label = 'Max Power')
    else:
        ax3.plot(np.linspace(-5,i/fps - 5,i), maxpower[0:i],label = 'Max Power')
    ax4.cla()
    ax4.set_title('Histogram showing the real time power flux distribution')
    ax4.set_ylim(1,N*N)
    ax4.hist(zarray2[:,:,i].flatten(),bins=100,color='blue',range=(1,zarray2.max()))
    ax4.set_yscale('log')
    ax4.set_xlabel('Optical Flux Power (W/m^2)')
    ax4.set_ylabel('Frequency')

def makearray(shr_name,i):
    if i < frn:
        existing_shm=shared_memory.SharedMemory(name = shr_name)
        earr = np.ndarray((N,N,frn),dtype = dtype, buffer=existing_shm.buf)
        temparr = zarray2[:,:,:i]-1
        arr = np.sum(temparr,axis=2)*1/fps
        earr[:,:,i] = arr
        print('done w'+str(i))
        del arr
        del temparr
        existing_shm.close()
if __name__ == '__main__':
    a=["100m_in_2000_1d","1000","2000","200","THIS WAS THE ERROR","100m_in_2000_1d","0","0","100","2000","100m","in","2000-Fragments(1Day)"]
#    for input in sys.stdin: # take inputs from a shell script
#        print(input)
#        input1 = input.strip()
#        b,c,d,e,f,g,h,i,j,k,l,m,n = input1.split()
#        a[0] = b
#        a[1] = c
#        a[2] = d
#        a[3] = e
#        a[4] = f
#        a[5] = g
#        a[6] = h
#        a[7] = i
#        a[8] = j
#        a[9] = k
#        a[10] = l
#        a[11] = m
#        a[12] = n
    name = a[0]  #filename of the input bin files
    N = int(a[1]) #This is the dimensions of any particular frame (always the same as XDIM from cpp)
    frn = int(a[2]) #This is the total number of frames in the simulation (has to be the same as frn from cpp) 
    fps= int(a[8]) #sets the fps (This has to be the same as fps from cpp) 
    rnge = int(a[3]) #The Range of the plots. SO if a plot covers -200 to 200 km on the x axis, this would be 200 
    title = a[4] #The Title of the plots
    outname = a[5] #The Name of the output files
    xcenter = int(a[6]) # Enter the x component of the center
    ycenter = int(a[7]) # Enter the y component of the center
    timesleep = 0 #The time computer sleeps before starting the program. Can use this to run this simultaneously with the cpp file 
    NOI = int(a[9])
    title1 = a[10]
    title2 = a[11]
    title3 = a[12]
    alpha = 0.1
    increment = 2*rnge/N
    xticks = np.linspace(xcenter-rnge,xcenter+rnge,5,endpoint = True)
    yticks = np.linspace(ycenter-rnge,ycenter+rnge,5,endpoint=True)
    xls = pd.ExcelFile(name+".xlsx", engine = "openpyxl")
    df = pd.read_excel(xls, 'Ring Fragment Bombardment -sort')
    x = df['OUTPUT - Asteroid Position x(km)'].to_numpy()
    y = df['OUTPUT - Asteroid Position y(km)'].to_numpy()
    x = np.delete(x,[0,1,2]) #delete the non-numeric data from each "array"
    y = np.delete(y,[0,1,2])
    sigma = df['Sigma for Gaussian (in time) optical pulse power (s)'].to_numpy()
    sigma = np.delete(sigma,[0,1,2])
    E = df['OUTPUT.28'].to_numpy()
    E = np.delete(E,[0,1,2])
    zburst = df['OUTPUT.22'].to_numpy()
    times = df['OUTPU'].to_numpy()
    zburst = np.delete(zburst,[0,1,2])
    times = np.delete(times,[0,1,2])
    hundredpower=df['P_opt_atm (w/m^2) - no projection - horizon cut - with atm transmission - assuming 100% conversion of blast to optical'].to_numpy()
    hundredpower = np.delete(hundredpower,[0,1,2])
    times = times[:NOI]
    mindisp = times.min()
    if mindisp < 0:
        times = times - mindisp + 5
    else:
        times = times - mindisp + 5
    distarray = np.zeros((2*N,2*N))
    xval = N
    yval = N
    cmap1 = mpl.cm.magma
    cmap2 = mpl.cm.Reds
    x_data = np.arange(0, frn, 1)
    y_data = []
    zarray2 = np.zeros((N,N,frn),dtype = dtype)
    zarray2[:,:,:]=0
    E = E * 4.184 * 10**12
    for i in range(len(times)):
        data = stats.norm.pdf(x_data, times[i]*fps, sigma[i]*fps)
        data[0:int(times[i]*fps  - 3 * sigma[i]*fps)]=0
        data[int(times[i]*fps + 3 * sigma[i]*fps):frn] = 0
        y_data.append(data)
    for j in range(0,2*N):
        for h in range(0,2*N):
            if (j-xval)**2+(h-yval)**2 != 0:
                distarray[j][h] = 1/((((j-xval)*increment*1000)**2+((h-yval)*increment*1000)**2))
            else:
                distarray[j][h] = 1
    index=0
    params = df['Alpha - BB weighted atmospheric transmission - exponential fit model parameters'].to_numpy()
    params = np.delete(params, [0,1,2,4,5])
    params = params[:2]
    a = params[0]
    b = params[1]
    begin_time = datetime.datetime.now()
    print('starting calculations for' + name)
    if (exists(name + '_lightpulse.bin')==False):
        starting = 0
        shm = shared_memory.SharedMemory(create = True, size = zarray2.nbytes)
        arr = np.ndarray(zarray2.shape,dtype = zarray2.dtype,buffer = shm.buf)
        arr[:,:,:] = zarray2[:,:,:]
        while starting < (NOI):
            jobs = []
            for i in range(starting,starting+maxloadcalc):
                p = mp.Process(target=makeblasts,args=(shm.name,i))
                jobs.append(p)
                p.start()
                starting += 1
            for z in jobs:
                z.join()
        zarray2 = arr
        #print(zarray2.max())
        zarray2 += 1
        zarray2 = zarray2.reshape((N,N,frn))
        newFile = open(name + "_lightpulse.bin", "wb")
        newFile.write(zarray2.flatten())
        newFile.close()
    else:
        zarray2 = np.zeros((N, N, frn))
        file = open(str(name) + "_lightpulse.bin", "rb") #open the bin file with the data
        raw_data = np.fromfile(file,dtype = np.float64)
        file.close()
        zarray2=raw_data.reshape((N,N,frn), order = "C") 
    print(fps)
    plt.figure()
    finish_time=datetime.datetime.now()
    print(finish_time-begin_time)
    order = math.ceil(np.log10(zarray2.max()))
    norm1 = mpl.colors.LogNorm(vmin = 1, vmax = 10**order)
    fig = plt.figure(figsize = (16,9))
    gs = mpl.gridspec.GridSpec(3,6,figure = fig)
    ax1 = plt.subplot(gs.new_subplotspec((0,0),colspan = 3,rowspan=2))
    ax2 = plt.subplot(gs.new_subplotspec((0,3),colspan = 3,rowspan=2))
    ax3 = plt.subplot(gs.new_subplotspec((2,0),colspan = 2,rowspan=1))
    ax4 = plt.subplot(gs.new_subplotspec((2,2),colspan = 2,rowspan=1))
    ax5 = plt.subplot(gs.new_subplotspec((2,4),colspan = 2,rowspan=1))
    fig.suptitle(title1 + ' Asteroid ' + title2 + ' ' + title3)
    ax1.imshow(zarray2[:,:,0],origin='lower',cmap = cmap1)
    ax2.set_xticks(np.linspace(0,N,5))
    ax2.set_xticklabels(xticks) 
    ax2.set_yticks(np.linspace(0,N,5))
    ax2.set_yticklabels(yticks)
    ax2.set_title('Optical Energy Flux Distribution')
    ax1.set_title('Optical Power Flux Distribution')
    Energyarray = np.zeros((N,N,frn),dtype=dtype)
    shm = shared_memory.SharedMemory(create = True, size = Energyarray.nbytes)
    arr = np.ndarray(Energyarray.shape,dtype = Energyarray.dtype,buffer = shm.buf)
    arr[:,:,:] = Energyarray[:,:,:]
    jobs=[]
    starting = 1
    while starting < frn:
        for i in range(starting, starting+maxload):
            p = mp.Process(target=makearray,args = (shm.name,i))
            starting += 1
            p.start()
            if starting == 1500:
                maxload = 10
            if starting ==3500:
                maxload = 10
            #time.sleep(0.5)
            jobs.append(p)
        for z in jobs:
            z.join()
    Energyarray = arr   
    #for i in range(1,frn):
     #   temparr = zarray2[:,:,:i] - 1
      #  Energyarray[:,:,i] = np.sum(temparr,axis=2)*1/fps
    Energyarray += 1
    print(fps)
    order2 = math.ceil(np.log10(Energyarray.max()))
    norm2 = mpl.colors.LogNorm(vmin = 1, vmax = 10**order2)
    ax2.imshow(Energyarray[:,:,0],norm=norm2,cmap=cmap1,origin='lower')
    fig.colorbar(mpl.cm.ScalarMappable(norm = norm2, cmap=cmap1),ax = ax2, label = "Optical Energy Flux(J/m^2)")
    fig.colorbar(mpl.cm.ScalarMappable(norm = norm1, cmap=cmap1),ax = ax1, label = "Optical Power Flux(W/m^2)")
    maxpower = []
    overallmaxarray = Energyarray[:,:,frn-1].flatten()
    count,bins_count = np.histogram(overallmaxarray, bins = 100)
    print(fps)
    pdf = count/sum(count)
    pdf = np.flip(pdf)
    cdf = np.cumsum(pdf)
    cdf = np.flip(cdf)
    ax4.hist([],bins=10)
    ax5.plot(bins_count[1:], cdf)
    ax5.set_title('Cumulative Distribution of the Final Energy')
    ax5.set_xlabel('Energy Flux (J/m^2)')
    ax5.locator_params(axis='x',nbins=5)
    ax5.set_ylabel('Cumulative Probability')
    ax5.set_yscale('log')
    plt.subplots_adjust(wspace=0.9,hspace=0.5)
    for i in range(frn):
        maxpower.append(zarray2[:,:,i].max())
    print(fps)
    anim = animation.FuncAnimation(fig,update_plot1,frn,interval = 1000/fps)
    anim.save(str(outname) + '_lightpulse-k.mp4')
    print('finished ' + name)
    del arr
    shm.close()
    shm.unlink()