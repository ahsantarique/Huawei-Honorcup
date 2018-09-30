import matplotlib.pyplot as plt
import numpy as np
import os
import sys


if(len(sys.argv) < 2):
	print("Usage: python3 plot_data.py path_to_data")
	print("Example: python3 plot_data.py data/")
	exit(0)


path = sys.argv[1]

def plot(X, name):
    plt.title(name)
    plt.plot(X, 'y-')
    plt.show()


def fftplot(data, name):
    rate = 500
    spectre = np.fft.fft(data)
    freq = np.fft.fftfreq(data.size, 1/rate)
    mask=freq>0   
    plt.plot(freq[mask],np.abs(spectre[mask]))
    plt.title(name)
    plt.show()

    
def loadData(path):
    for file in os.listdir(path):
        data = np.loadtxt(path+'/'+file, delimiter=',', dtype='int')
        X_local = data[1:]

        ppg = X_local[:,0]
        ecg = X_local[:,1]

        # X_local = (np.array(X_local)-np.mean(X_local, axis=0))/(np.max(X_local)-np.min(X_local))  # normalization
        # print(X_local) # okay, checked
        fftplot(ecg, 'ecg')
        fftplot(ppg, 'ppg')


if(__name__=="__main__"):
    loadData(path)