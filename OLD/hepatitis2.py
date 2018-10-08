from sklearn.decomposition import PCA, NMF
from sklearn import datasets

#iris = datasets.load_iris()
#print(iris)

import csv

data = []
label = []

arr = []

with open('hepatitis.data', 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	for row in reader:
		# ~ arr.append(list(map(lambda x: int(x), row)))
		arr.append(row)

import numpy as np

print(np.array(arr)[:10,0])

from sklearn import preprocessing


arr = np.array(arr)

for i in range(0, np.array(arr).shape[1]):
	
	
	le = preprocessing.LabelEncoder()
	le.fit(np.array(arr)[:,i])
	
	arr[:,i] = le.transform(np.array(arr)[:,i])
	print(np.array(arr)[:,i])

print("==")
#cross-prediction
print(arr[:3])


arr = arr.tolist()

data = []
label = []
CLA = 0
feats = ['die/live','age','sex','steroid','antivirals','fatigue','malaise','anorexia','liver big','liver firm','spleen palpable','spiders','ascites','varices','bilirubin','alk phosphate','sgot','albumin','protime','histology']

# ~ print(feats[CLA],'linked with: ')
for i in range(0,len(arr)):
	
	label.append(arr[i][CLA])
	
	del arr[i][CLA]
	
	data.append(arr[i])

del feats[CLA]


#BEFORE PCA
from sklearn.model_selection import train_test_split
Xtr, Xte, Ytr, Yte = train_test_split(data, label, test_size=0.1, random_state=42)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=7, random_state=0)
clf.fit(Xtr, Ytr)
print(clf.feature_importances_)
print("RF: ", clf.score(Xte,Yte))



#AFTER PCA
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

chu = PCA(n_components=2)
# ~ chu = CCA(n_components=1)
# ~ chu = FastICA(n_components=2)
# ~ chu = LinearDiscriminantAnalysis(n_components=1)
chu = NMF() #cool!
data = chu.fit_transform(np.array(data).astype(np.float),np.array(label).astype(np.float))
print(data[:2])
# ~ data = chu.transform(data)
# ~ data = chu.fit_transform(data)

# ~ print("CHU",chu.explained_variance_ratio_)  
# ~ print("SV",chu.singular_values_)

import matplotlib.pyplot as plt
plt.scatter(data[:,0],data[:,1],c=label)
plt.show()


from sklearn.model_selection import train_test_split
Xtr, Xte, Ytr, Yte = train_test_split(data, label, test_size=0.1, random_state=42)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=7, random_state=0)
clf.fit(Xtr, Ytr)

# ~ for x in sorted(list(zip(feats,clf.feature_importances_)),key=lambda x: x[1],reverse=True):
	# ~ print(x)
	
print(clf.feature_importances_)
print("RF: ", clf.score(Xte,Yte))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
KNN = KNeighborsClassifier(3)
KNN.fit(Xtr,Ytr)
print("KNN: ",KNN.score(Xte,Yte))


import sys
sys.exit(0)
