import sys
import os
import random
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

if(len(sys.argv) < 2):
	print("Usage: python3 regression.py path_to_data [number_of_samples_to_use]")
	print("Example: python3 regression.py data/  [200]")
	exit(0)

MAX_FILE_COUNT = 1e8

path = sys.argv[1]
if(len(sys.argv) > 2):
	MAX_FILE_COUNT = int(sys.argv[2])

clf_sbp = MLPRegressor( hidden_layer_sizes=(8,8,8), warm_start=True)
clf_dbp = MLPRegressor( hidden_layer_sizes=(8,8,8), warm_start=True)

print(clf_sbp)

X=np.empty((0,2))
y_s=[]
y_d=[]

count = 0
for file in os.listdir(path):
	data = np.loadtxt(path+'/'+file, delimiter=',', dtype='int')
	y_local = data[0]
	X_local = data[1:]

	X_local = (np.array(X_local)-np.mean(X_local, axis=0))/np.std(X_local, axis=0)  # normalization

	# print(X_local) # okay, checked

	#X = np.vstack((X,X_local))
	y_s = [y_local[0]]*len(X_local)
	y_d = [y_local[1]]*len(X_local)

	#train on half of the data
	if(random.random() < 0.5):	
		clf_sbp.fit(X_local, y_s)
		clf_dbp.fit(X_local, y_d)

	count += 1
	if(count % 100 == 0):
		print("completed {} files".format(count))
	if(count > MAX_FILE_COUNT):
		break

print("done training on the data....")

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
total_datapoints = 0
for file in os.listdir(path):
	data = np.loadtxt(path+'/'+file, delimiter=',', dtype='int')
	y_local = data[0]
	X_local = data[1:]

	X_local = (np.array(X_local)-np.mean(X_local, axis=0))/np.std(X_local, axis=0)  # normalization
	y_s = [y_local[0]]*len(X_local)
	y_d = [y_local[1]]*len(X_local)

	y_s_pred = clf_sbp.predict(X_local)
	y_d_pred = clf_dbp.predict(X_local)

	mse += np.sum((y_s_pred - y_s)**2) + 2*np.sum((y_d_pred - y_d)**2)


	count += 1
	total_datapoints += len(X_local)
	if(count % 100 == 0):
		print("completed {} files".format(count))
	if(count > MAX_FILE_COUNT):
		break

print("MSE:", mse/total_datapoints)