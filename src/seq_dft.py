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
	print("Usage: python3 regression.py path_to_data [number_of_samples_to_use]")
	print("Example: python3 regression.py data/  [200]")
	exit(0)

MAX_FILE_COUNT = 1e8
DISCARD_FRONT = 4000
ECG_DATA_POINTS = 60
print("ECG_DATA_POINTS", ECG_DATA_POINTS)
PPG_DATA_POINTS = 20
# DISCARD_BACK = 5000


path = sys.argv[1]
if(len(sys.argv) > 2):
	MAX_FILE_COUNT = int(sys.argv[2])

random.seed(42)

clf_sbp = []
clf_dbp = []

# clf_sbp.append(MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#        beta_2=0.999, early_stopping=False, epsilon=1e-08,
#        hidden_layer_sizes=(100, 100, 100), learning_rate='adaptive',
#        learning_rate_init=0.0001, max_iter=10000, momentum=0.9,
#        nesterovs_momentum=True, power_t=0.5, random_state=None,
#        shuffle=True, solver='adam', tol=1e-08, validation_fraction=0.1,
#        verbose=False, warm_start=True) )

# clf_dbp.append(MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#        beta_2=0.999, early_stopping=False, epsilon=1e-08,
#        hidden_layer_sizes=(100, 100, 100), learning_rate='adaptive',
#        learning_rate_init=0.0001, max_iter=10000, momentum=0.9,
#        nesterovs_momentum=True, power_t=0.5, random_state=None,
#        shuffle=True, solver='adam', tol=1e-08, validation_fraction=0.1,
#        verbose=False, warm_start=True))

#####################################################################################################
# clf_sbp.append(MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#        beta_2=0.999, early_stopping=False, epsilon=1e-08,
#        hidden_layer_sizes=(64,64,64), learning_rate='adaptive',
#        learning_rate_init=0.0001, max_iter=10000, momentum=0.9,
#        nesterovs_momentum=True, power_t=0.5, random_state=None,
#        shuffle=True, solver='adam', tol=1e-08, validation_fraction=0.1,
#        verbose=False, warm_start=True) )

# clf_dbp.append(MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#        beta_2=0.999, early_stopping=False, epsilon=1e-08,
#        hidden_layer_sizes=(64,64,64), learning_rate='adaptive',
#        learning_rate_init=0.0001, max_iter=10000, momentum=0.9,
#        nesterovs_momentum=True, power_t=0.5, random_state=None,
#        shuffle=True, solver='adam', tol=1e-08, validation_fraction=0.1,
#        verbose=False, warm_start=True))

#####################################################################################################
# gc.collect()

# clf_sbp.append(MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#        beta_2=0.999, early_stopping=False, epsilon=1e-08,
#        hidden_layer_sizes=(32,32,32), learning_rate='adaptive',
#        learning_rate_init=0.0001, max_iter=50000, momentum=0.9,
#        nesterovs_momentum=True, power_t=0.5, random_state=None,
#        shuffle=True, solver='adam', tol=1e-08, validation_fraction=0.1,
#        verbose=False, warm_start=False) )

# clf_dbp.append(MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#        beta_2=0.999, early_stopping=False, epsilon=1e-08,
#        hidden_layer_sizes=(32,32,32), learning_rate='adaptive',
#        learning_rate_init=0.0001, max_iter=50000, momentum=0.9,
#        nesterovs_momentum=True, power_t=0.5, random_state=None,
#        shuffle=True, solver='adam', tol=1e-08, validation_fraction=0.1,
#        verbose=False, warm_start=False))


####################################################################################################
# gc.collect()
# clf_sbp.append(MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#        beta_2=0.999, early_stopping=False, epsilon=1e-08,
#        hidden_layer_sizes=(16,16), learning_rate='adaptive',
#        learning_rate_init=0.0001, max_iter=50000, momentum=0.9,
#        nesterovs_momentum=True, power_t=0.5, random_state=None,
#        shuffle=True, solver='adam', tol=1e-08, validation_fraction=0.1,
#        verbose=False, warm_start=False) )

# clf_dbp.append(MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#        beta_2=0.999, early_stopping=False, epsilon=1e-08,
#        hidden_layer_sizes=(16,16), learning_rate='adaptive',
#        learning_rate_init=0.0001, max_iter=50000, momentum=0.9,
#        nesterovs_momentum=True, power_t=0.5, random_state=None,
#        shuffle=True, solver='adam', tol=1e-08, validation_fraction=0.1,
#        verbose=False, warm_start=False))


#####################################################################################################
# gc.collect()
# clf_sbp.append(MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#        beta_2=0.999, early_stopping=False, epsilon=1e-08,
#        hidden_layer_sizes=(8,8), learning_rate='adaptive',
#        learning_rate_init=0.0001, max_iter=50000, momentum=0.9,
#        nesterovs_momentum=True, power_t=0.5, random_state=None,
#        shuffle=True, solver='adam', tol=1e-08, validation_fraction=0.1,
#        verbose=False, warm_start=False) )

# clf_dbp.append(MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#        beta_2=0.999, early_stopping=False, epsilon=1e-08,
#        hidden_layer_sizes=(8,8), learning_rate='adaptive',
#        learning_rate_init=0.0001, max_iter=50000, momentum=0.9,
#        nesterovs_momentum=True, power_t=0.5, random_state=None,
#        shuffle=True, solver='adam', tol=1e-08, validation_fraction=0.1,
#        verbose=False, warm_start=False))


#####################################################################################################

# clf_sbp.append(LinearRegression(copy_X=True, fit_intercept=True, n_jobs=4, normalize=True))
# clf_dbp.append(LinearRegression(copy_X=True, fit_intercept=True, n_jobs=4, normalize=True))

# clf_sbp.append(SVR(tol=1e-8))
# clf_dbp.append(SVR(tol=1e-8))
# gc.collect()
# clf_sbp.append(RandomForestRegressor(n_estimators=20000, n_jobs=4, max_depth=3))
# clf_dbp.append(RandomForestRegressor(n_estimators=20000, n_jobs=4, max_depth=3))

# gc.collect()
clf_sbp.append(RandomForestRegressor(n_estimators=5000, n_jobs=4, max_depth=5, max_features='sqrt'))
clf_dbp.append(RandomForestRegressor(n_estimators=5000, n_jobs=4, max_depth=5, max_features='sqrt'))

# gc.collect()
# clf_sbp.append(GradientBoostingRegressor(n_estimators=50))
# clf_dbp.append(GradientBoostingRegressor(n_estimators=50))

# gc.collect()
clf_sbp.append(KNNR(algorithm='auto', leaf_size=30, metric='minkowski',
          metric_params=None, n_jobs=4, n_neighbors=10, p=2,
          weights='uniform') )
clf_dbp.append(KNNR(algorithm='auto', leaf_size=30, metric='minkowski',
          metric_params=None, n_jobs=4, n_neighbors=10, p=2,
          weights='uniform'))


X=np.empty((0, ECG_DATA_POINTS + PPG_DATA_POINTS))
y_s=[]
y_d=[]


count = 0
NUMBER_OF_CLF = len(clf_sbp)
files = os.listdir(path)
random.shuffle(files)
for file in files:
	if(count > 400 and count < 500):
		count += 1
		continue

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


clf_final_sbp = RandomForestRegressor()
clf_final_dbp = RandomForestRegressor()

print("*************************** MEAN PREDICTION ***************************")
kf = KFold(n_splits = 2)
y_s = np.array(y_s)
y_d = np.array(y_d)


X_train = X
y_s_train = y_s
y_d_train = y_d

x_s_final_train = []
x_d_final_train = []

for i in range(NUMBER_OF_CLF):
    # if(i >= NUMBER_OF_CLF-2):
    # 	clf_sbp[i].n_estimators += 5
    # 	clf_dbp[i].n_estimators += 5
	clf_sbp[i].fit(X_train, y_s_train)
	clf_dbp[i].fit(X_train, y_d_train)

	####### pred on train data
	y_s_pred_train = clf_sbp[i].predict(X_train)
	y_d_pred_train = clf_dbp[i].predict(X_train)

	#prepare feature for final clf

	x_s_final_train = np.concatenate((x_s_final_train, y_s_pred_train))
	x_d_final_train = np.concatenate((x_d_final_train, y_d_pred_train))

	print("****************************************************************************\n")
	print(clf_sbp[i],"\n")
	print("clf", i)

	mse_s_train = np.mean((y_s_pred_train - y_s_train)**2)
	mse_d_train = np.mean((y_d_pred_train - y_d_train)**2)
	print("TRAINING: MSE (sbp) :", mse_s_train, " MSE (dbp):", mse_d_train)


###################################################################################################
x_s_final_train = np.reshape(x_s_final_train, (NUMBER_OF_CLF, -1))
x_s_final_train = x_s_final_train.transpose()
print("x_s_final_train.shape", x_s_final_train.shape)

x_d_final_train = np.reshape(x_d_final_train, (NUMBER_OF_CLF, -1))
x_d_final_train = x_d_final_train.transpose()



clf_final_sbp.fit(x_s_final_train, y_s_train)
clf_final_dbp.fit(x_d_final_train, y_d_train)

y_s_pred_train_final = clf_final_sbp.predict(x_s_final_train)
y_d_pred_train_final = clf_final_dbp.predict(x_d_final_train)

print("****************************************************************************\n")
print("FINAL CLF")
print(clf_final_sbp)

mse_s_train_final = np.mean((y_s_pred_train_final - y_s_train)**2)
mse_d_train_final = np.mean((y_d_pred_train_final - y_d_train)**2)
print("FINAL TRAINING: MSE (sbp) :", mse_s_train_final, " MSE (dbp):", mse_d_train_final)


for i in range(NUMBER_OF_CLF):
	joblib.dump(clf_sbp[i], 'clf_sbp'+str(i)+'.pkl', compress=9)
	joblib.dump(clf_dbp[i], 'clf_dbp'+str(i)+'.pkl', compress=9)

joblib.dump(clf_final_sbp, 'clf_final_sbp.pkl', compress=9)
joblib.dump(clf_final_dbp, 'clf_final_dbp.pkl', compress=9)