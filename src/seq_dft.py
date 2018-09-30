import sys
import os
import random
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

if(len(sys.argv) < 2):
	print("Usage: python3 regression.py path_to_data [number_of_samples_to_use]")
	print("Example: python3 regression.py data/  [200]")
	exit(0)

MAX_FILE_COUNT = 1e8
DISCARD_FRONT = 4000
ECG_DATA_POINTS = 80
PPG_DATA_POINTS = 20
# DISCARD_BACK = 5000
TRAIN_TEST_RATIO = 0.5


path = sys.argv[1]
if(len(sys.argv) > 2):
	MAX_FILE_COUNT = int(sys.argv[2])

random.seed(42)

clf_sbp = []
clf_dbp = []

clf_sbp.append(MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(64, 64, 64), learning_rate='adaptive',
       learning_rate_init=0.0001, max_iter=10000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=1e-08, validation_fraction=0.1,
       verbose=False, warm_start=True) )

clf_dbp.append(MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(64, 64, 64), learning_rate='adaptive',
       learning_rate_init=0.0001, max_iter=10000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=1e-08, validation_fraction=0.1,
       verbose=False, warm_start=True))

# #####################################################################################################
# clf_sbp.append(MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#        beta_2=0.999, early_stopping=False, epsilon=1e-08,
#        hidden_layer_sizes=(8,8,8), learning_rate='adaptive',
#        learning_rate_init=0.0001, max_iter=10000, momentum=0.9,
#        nesterovs_momentum=True, power_t=0.5, random_state=None,
#        shuffle=True, solver='adam', tol=1e-08, validation_fraction=0.1,
#        verbose=False, warm_start=True) )

# clf_dbp.append(MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#        beta_2=0.999, early_stopping=False, epsilon=1e-08,
#        hidden_layer_sizes=(8,8,8), learning_rate='adaptive',
#        learning_rate_init=0.0001, max_iter=10000, momentum=0.9,
#        nesterovs_momentum=True, power_t=0.5, random_state=None,
#        shuffle=True, solver='adam', tol=1e-08, validation_fraction=0.1,
#        verbose=False, warm_start=True))

# #####################################################################################################

# clf_sbp.append(MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#        beta_2=0.999, early_stopping=False, epsilon=1e-08,
#        hidden_layer_sizes=(4,4,4), learning_rate='adaptive',
#        learning_rate_init=0.0001, max_iter=50000, momentum=0.9,
#        nesterovs_momentum=True, power_t=0.5, random_state=None,
#        shuffle=True, solver='adam', tol=1e-08, validation_fraction=0.1,
#        verbose=False, warm_start=True) )

# clf_dbp.append(MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#        beta_2=0.999, early_stopping=False, epsilon=1e-08,
#        hidden_layer_sizes=(4,4,4), learning_rate='adaptive',
#        learning_rate_init=0.0001, max_iter=50000, momentum=0.9,
#        nesterovs_momentum=True, power_t=0.5, random_state=None,
#        shuffle=True, solver='adam', tol=1e-08, validation_fraction=0.1,
#        verbose=False, warm_start=True))


# #####################################################################################################
# clf_sbp.append(MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#        beta_2=0.999, early_stopping=False, epsilon=1e-08,
#        hidden_layer_sizes=(16,16,16), learning_rate='adaptive',
#        learning_rate_init=0.0001, max_iter=50000, momentum=0.9,
#        nesterovs_momentum=True, power_t=0.5, random_state=None,
#        shuffle=True, solver='adam', tol=1e-08, validation_fraction=0.1,
#        verbose=False, warm_start=True) )

# clf_dbp.append(MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#        beta_2=0.999, early_stopping=False, epsilon=1e-08,
#        hidden_layer_sizes=(16,16,16), learning_rate='adaptive',
#        learning_rate_init=0.0001, max_iter=50000, momentum=0.9,
#        nesterovs_momentum=True, power_t=0.5, random_state=None,
#        shuffle=True, solver='adam', tol=1e-08, validation_fraction=0.1,
#        verbose=False, warm_start=True))


# #####################################################################################################


clf_sbp.append(LinearRegression(copy_X=True, fit_intercept=True, n_jobs=4, normalize=True))
clf_dbp.append(LinearRegression(copy_X=True, fit_intercept=True, n_jobs=4, normalize=True))

clf_sbp.append(SVR(tol=1e-8))
clf_dbp.append(SVR(tol=1e-8))

clf_sbp.append(RandomForestRegressor(n_estimators=10000, n_jobs=4 ))
clf_dbp.append(RandomForestRegressor(n_estimators=10000, n_jobs=4))

clf_sbp.append(GradientBoostingRegressor(n_estimators=500))
clf_dbp.append(GradientBoostingRegressor(n_estimators=500))

X=np.empty((0, ECG_DATA_POINTS + PPG_DATA_POINTS))
y_s=[]
y_d=[]


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
	ecg_fft = np.abs(spectre[mask])[:ECG_DATA_POINTS]
	ecg_fft = (np.array(ecg_fft)-np.mean(ecg_fft, axis=0))/(np.max(ecg_fft)-np.min(ecg_fft))


	spectre = np.fft.fft(ppg)
	freq = np.fft.fftfreq(ppg.size, 1/rate)
	mask = freq > 0
	ppg_fft = np.abs(spectre[mask])[:PPG_DATA_POINTS]
	ppg_fft = (np.array(ppg_fft)-np.mean(ppg_fft, axis=0))/(np.max(ppg_fft)-np.min(ppg_fft))


	fft_local = np.append(ecg_fft, ppg_fft)

	# print(X.shape, fft_local.shape) 	# okay, checked

	X = np.vstack((X, fft_local))

	y_s.append(y_local[0])
	y_d.append(y_local[1])

	count += 1
	if(count % 100 == 0):
		print("completed {} files".format(count))
	if(count > MAX_FILE_COUNT):
		break

print("done accumulating data...")


# clf_final_sbp = RandomForestRegressor()
# clf_final_dbp = RandomForestRegressor()

# print("training done for the final clf")

# kf = KFold(n_splits = 5)
# for train_index, test_index in kf.split(X):
# 	#print("TRAIN:", train_index, "TEST:", test_index)
# 	X_train, X_test = X[train_index], X[test_index]
# 	y_s_train, y_s_test = y_s[train_index], y_s[test_index]
# 	y_d_train, y_d_test = y_d[train_index], y_d[test_index]
	
# for i in range(NUMBER_OF_CLF):
#     # if(i >= NUMBER_OF_CLF-2):
#     # 	clf_sbp[i].n_estimators += 5
#     # 	clf_dbp[i].n_estimators += 5
#     clf_sbp[i].fit(X, y_s)
#     clf_dbp[i].fit(X, y_d)

# 	y_s_pred = clf_sbp.predict(X_test)
# 	y_d_pred = clf_dbp.predict(X_test)


for i in range(NUMBER_OF_CLF):
	print("****************************************************************************")
	print(clf_sbp[i])
	score_s = cross_val_score(clf_sbp[i], X, y_s, cv=2, scoring='neg_mean_squared_error')
	score_d = cross_val_score(clf_dbp[i], X, y_d, cv=2, scoring='neg_mean_squared_error')
	print(score_s)
	print(score_d)

# mse = 0
# count = 0
# for file in os.listdir(path):
# 	data = np.loadtxt(path+'/'+file, delimiter=',', dtype='int')
# 	y_local = data[0]
#	X_local = data[DISCARD_FRONT:DISCARD_FRONT+DATA_POINTS].flatten()

# 	X_local = (np.array(X_local)-np.mean(X_local, axis=0))/(np.max(X_local)-np.min(X_local))  # normalization
# 	y_s = y_local[0]
# 	y_d = y_local[1]

# 	y_s_pred = np.mean([np.mean(clf_sbp[i].predict(X_local)) for i in range(NUMBER_OF_CLF)])
# 	y_d_pred = np.mean([np.mean(clf_dbp[i].predict(X_local)) for i in range(NUMBER_OF_CLF)])
	
# 	mse += (y_s_pred - y_s)**2 + 2*((y_d_pred - y_d)**2)


# 	count += 1
# 	if(count % 100 == 0):
# 		print("completed {} files".format(count))
# 	if(count > MAX_FILE_COUNT):
# 		break

# print("MSE:", mse/count)