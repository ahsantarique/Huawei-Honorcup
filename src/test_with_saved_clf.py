import sys
import os
import random
import gc
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsRegressor as KNNR

if(len(sys.argv) < 2):
	print("Usage: python3 test.py path_to_data [number_of_samples_to_use]")
	print("Example: python3 regression.py data/  [200]")
	exit(0)

MAX_FILE_COUNT = 1e8
ECG_DATA_POINTS = 60
print("ECG_DATA_POINTS", ECG_DATA_POINTS)
PPG_DATA_POINTS = 20
# DISCARD_BACK = 5000
TRAIN_TEST_RATIO = 0.5


path = sys.argv[1]
if(len(sys.argv) > 2):
	MAX_FILE_COUNT = int(sys.argv[2])

random.seed(42)

clf_sbp = []
clf_dbp = []

clf_final_sbp = None
clf_final_dbp = None

for f in os.listdir("./"):
	if(f.endswith(".pkl")):
		if("clf_final_sbp" in f):
			clf_final_sbp = joblib.load(f)
		elif("clf_final_dbp" in f):
			clf_final_dbp = joblib.load(f)
		elif("sbp" in f):
			clf_sbp.append(joblib.load(f))
			print(f)

		elif("dbp" in f):
			clf_dbp.append(joblib.load(f))

print("final", clf_final_sbp)
print("sbp", clf_sbp)


count = 0
NUMBER_OF_CLF = len(clf_sbp)
files = os.listdir(path)
random.shuffle(files)
for file in files:
	data = np.loadtxt(path+'/'+file, delimiter=',', dtype='int')
	y_local = data[0]

	X_local = data[1:]
	ppg = X_local[:,0]
	ecg = X_local[:,1]

	rate = 500
	#X_local = (np.array(X_local)-np.mean(X_local, axis=0))/(np.max(X_local)-np.min(X_local))
	spectre = np.fft.fft(ecg)
	freq = np.fft.fftfreq(ecg.size, 1/rate)
	mask = freq > 0
	ecg_fft = np.abs(spectre[mask])
	ecg_fft = ecg_fft[:ECG_DATA_POINTS]
	#ecg_fft = np.log(ecg_fft)
	ecg_fft = (np.array(ecg_fft)-np.mean(ecg_fft, axis=0))/(np.max(ecg_fft) - np.min(ecg_fft))

	spectre = np.fft.fft(ppg)
	freq = np.fft.fftfreq(ppg.size, 1/rate)
	mask = freq > 0
	ppg_fft = np.abs(spectre[mask])
	ppg_fft = ppg_fft[:PPG_DATA_POINTS]
	#ppg_fft = np.log(ppg_fft)
	ppg_fft = (np.array(ppg_fft)-np.mean(ppg_fft, axis=0))/(np.max(ppg_fft) - np.min(ppg_fft))

	fft_local = np.append(ecg_fft, ppg_fft)

	count += 1
	# if(count % 100 == 0):
	# 	print("completed {} files".format(count))
	if(count > MAX_FILE_COUNT):
		break

	x_s_final_test = []
	x_d_final_test = []
	y_s_pred_test = 0
	y_d_pred_test = 0
	for i in range(NUMBER_OF_CLF):
		######### pred on test data
		y_s_pred_test = clf_sbp[i].predict([fft_local])
		y_d_pred_test = clf_dbp[i].predict([fft_local])

		x_s_final_test = np.concatenate((x_s_final_test, y_s_pred_test))
		x_d_final_test = np.concatenate((x_d_final_test, y_d_pred_test))

	x_s_final_test = np.reshape(x_s_final_test, (NUMBER_OF_CLF, -1))
	x_s_final_test = x_s_final_test.transpose()

	x_d_final_test = np.reshape(x_d_final_test, (NUMBER_OF_CLF, -1))
	x_d_final_test = x_d_final_test.transpose()


	y_s_pred_test_final = clf_final_sbp.predict(x_s_final_test)
	y_d_pred_test_final = clf_final_dbp.predict(x_d_final_test)


	if(y_local[0]==0):
		print("{},{},{}".format(file,int(np.round(y_s_pred_test_final[0])), int(np.round(y_d_pred_test_final[0]))))
	else:
		print("{},{},{}".format(file,y_local[0], y_local[1]))
print("done")


