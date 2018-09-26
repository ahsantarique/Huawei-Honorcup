import sys
import os
import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

if(len(sys.argv) < 2):
	print("Usage: python3 regression.py path_to_data")
	exit(0)

path = sys.argv[1]

clf_sbp = RandomForestRegressor()
clf_dbp = RandomForestRegressor()

print(clf_sbp)

X=np.empty((0,2))
y_s=[]
y_d=[]

count = 0
for file in os.listdir(path):
	data = np.loadtxt(path+'/'+file, delimiter=',', dtype='int')
	y_local = data[0]
	X_local = data[1:]

	X = np.vstack((X,X_local))
	y_s += [y_local[0]]*len(X_local)
	y_d += [y_local[1]]*len(X_local)

	count += 1
	if(count %50 == 0):
		print("completed {} files".format(count))
		break

print("done accumulating data....")

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

# 	clf.score



score_s = cross_val_score(clf_sbp, X, y_s, cv=2, scoring='neg_mean_squared_error')
score_d = cross_val_score(clf_dbp, X, y_d, cv=2, scoring='neg_mean_squared_error')

print(score_s.mean())
print(score_d.mean())


	


