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

if(len(sys.argv) < 2):
	print("Usage: python3 regression.py path_to_data [number_of_samples_to_use]")
	print("Example: python3 regression.py data/  [200]")
	exit(0)

MAX_FILE_COUNT = 1e8

path = sys.argv[1]
if(len(sys.argv) > 2):
	MAX_FILE_COUNT = int(sys.argv[2])

random.seed(42)

clf_sbp = []
clf_dbp = []

clf_sbp.append(MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(32, 32, 32), learning_rate='adaptive',
       learning_rate_init=0.0001, max_iter=1000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=1e-08, validation_fraction=0.1,
       verbose=False, warm_start=True) )

clf_dbp.append(MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(32, 32, 32), learning_rate='adaptive',
       learning_rate_init=0.0001, max_iter=1000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=1e-08, validation_fraction=0.1,
       verbose=False, warm_start=True))


clf_sbp.append(LinearRegression(copy_X=True, fit_intercept=True, n_jobs=4, normalize=True))
clf_dbp.append(LinearRegression(copy_X=True, fit_intercept=True, n_jobs=4, normalize=True))

clf_sbp.append(SVR(tol=1e-8))
clf_dbp.append(SVR(tol=1e-8))

print(clf_sbp)

X=np.empty((0,2))
y_s=[]
y_d=[]

count = 0
NUMBER_OF_CLF = len(clf_sbp)
for file in os.listdir(path):
	data = np.loadtxt(path+'/'+file, delimiter=',', dtype='int')
	y_local = data[0]

	to_discard = 5000
	while (len(data) <= to_discard):
		to_discard /= 2
	X_local = data[to_discard:]

	X_local = (np.array(X_local)-np.mean(X_local, axis=0))/(np.max(X_local)-np.min(X_local))  # normalization

	# print(X_local) # okay, checked

	#X = np.vstack((X,X_local))
	y_s = [y_local[0]]*len(X_local)
	y_d = [y_local[1]]*len(X_local)

	#train on half of the data
	if(random.random() < 0.5):
		for i in range(NUMBER_OF_CLF):
			clf_sbp[i].fit(X_local, y_s)
			clf_dbp[i].fit(X_local, y_d)

	count += 1
	if(count % 100 == 0):
		print("completed {} files".format(count))
	if(count > MAX_FILE_COUNT):
		break

print("done training on the data on the intermediate clfs....")







# clf_final_sbp = RandomForestRegressor()
# clf_final_dbp = RandomForestRegressor()

# print("training done for the final clf")

# kf = KFold(n_split = 5)

# for train_index, test_index in kf.split(X):
# 	#print("TRAIN:", train_index, "TEST:", test_index)
# 	X_train, X_test = X[train_index], X[test_index]
# 	y_s_train, y_s_test = y_s[train_index], y_s[test_index]
# 	y_d_train, y_d_test = y_d[train_index], y_d[test_index]
	
# 	clf_sbp.fit(X_train, y_s)
# 	clf_dbp.fit(X_train, y_d)

# 	y_s_pred = clf_sbp.predict(X_test)
# 	y_d_pred = clf_dbp.predict(X_test)

# score_s = cross_val_score(clf_sbp, X, y_s, cv=4, scoring='neg_mean_squared_error')
# score_d = cross_val_score(clf_dbp, X, y_d, cv=4, scoring='neg_mean_squared_error')
# print(score_s.mean())
# print(score_d.mean())

mse = 0
count = 0
for file in os.listdir(path):
	data = np.loadtxt(path+'/'+file, delimiter=',', dtype='int')
	y_local = data[0]

	to_discard = 5000
	while (len(data) <= to_discard):
		to_discard /= 2
	X_local = data[to_discard:]

	X_local = (np.array(X_local)-np.mean(X_local, axis=0))/(np.max(X_local)-np.min(X_local))  # normalization
	y_s = y_local[0]
	y_d = y_local[1]

	y_s_pred = np.mean([np.mean(clf_sbp[i].predict(X_local)) for i in range(NUMBER_OF_CLF)])
	y_d_pred = np.mean([np.mean(clf_dbp[i].predict(X_local)) for i in range(NUMBER_OF_CLF)])


	
	mse += (y_s_pred - y_s)**2 + 2*((y_d_pred - y_d)**2)


	count += 1
	if(count % 100 == 0):
		print("completed {} files".format(count))
	if(count > MAX_FILE_COUNT):
		break

print("MSE:", mse/count)