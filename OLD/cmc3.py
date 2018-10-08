from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import pandas as pd


data = []
label = []

arr = []

dt = pd.read_csv('white.csv',sep=';')

print(dt)


print(dt.describe().stack().to_csv('sxxx.csv'))
print(dt.describe().unstack().to_csv('uxxx.csv'))

X = dt.iloc[:,:-1].values
Y = dt.iloc[:,-1].values
print(X)

from sklearn.model_selection import train_test_split
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.1, random_state=42)

X=preprocessing.scale(X)
Y=preprocessing.scale(Y)


from sklearn.svm import SVR

# ~ svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
# ~ svr_lin = SVR(kernel='linear', C=1e3)
# ~ svr_poly = SVR(kernel='poly', C=1e3, degree=2)

from sklearn.decomposition import PCA, KernelPCA
# ~ from sklearn.decomposition import PCA
# ~ pca = PCA(n_components=5)
# ~ pca = KernelPCA(kernel="rbf", gamma=1, n_components=4)
# ~ X = pca.fit_transform(X)
# ~ X = pca.fit_transform(X)
# ~ print("PCA done")

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# ~ plt.scatter(preprocessing.scale(dt['alcohol']), preprocessing.scale(dt['quality']))
# ~ plt.show()

# ~ sys.exit(0)



# ~ rfc = SGDClassifier(penalty=None)
rfc = RandomForestClassifier(n_estimators=200)
# ~ rfc = SVC(C=1000,gamma=0.05)
from sklearn import tree
# ~ rfc = tree.DecisionTreeClassifier(max_depth=4,n_estimators=200)


from sklearn.linear_model import Ridge
# ~ rfc = Ridge(alpha=1.7)
rfc.fit(Xtr, Ytr)
Ypr = rfc.predict(Xte)

for x in sorted(list(zip(dt.columns.values,rfc.feature_importances_)),key=lambda x: x[1],reverse=True):
	print(x)

print(Yte)
print(Ypr)
print(classification_report(Yte, np.floor(Ypr)))

# ~ https://www.kaggle.com/vishalyo990/prediction-of-quality-of-wine





sys.exit(0)

























from sklearn import preprocessing


print(arr[:3])
# ~ Attribute Information:

# ~ For more information, read [Cortez et al., 2009].
# ~ Input variables (based on physicochemical tests):
# ~ 1 - fixed acidity
# ~ 2 - volatile acidity
# ~ 3 - citric acid
# ~ 4 - residual sugar
# ~ 5 - chlorides
# ~ 6 - free sulfur dioxide
# ~ 7 - total sulfur dioxide
# ~ 8 - density
# ~ 9 - pH
# ~ 10 - sulphates
# ~ 11 - alcohol
# ~ Output variable (based on sensory data):
# ~ 12 - quality (score between 0 and 10)

# ~ import sys
# ~ sys.exit(0)

data = []
label = []
CLA = 6
# ~ feats = ['age','wedu','hedu','kids','islam','isworking','hwork','QoL','Media','Method']
# ~ arr = arr.tolist()
print(feats[CLA],'depends on: ')

label = arr[:,CLA].tolist()
# ~ del arr[:][CLA]
arr = np.delete(arr, CLA, 1)
data = arr.tolist()

print(label)

# ~ for i in range(0,len(arr)):
	
	# ~ label.append(arr[i][CLA])
	
	# ~ del arr[i][CLA]
	
	# ~ data.append(arr[i])

del feats[CLA]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(label)
label = le.transform(label)


from sklearn.model_selection import train_test_split
Xtr, Xte, Ytr, Yte = train_test_split(data, label, test_size=0.1, random_state=42)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=12, random_state=0)
clf.fit(Xtr, Ytr)
for x in list(zip(feats,clf.feature_importances_)):
	print(x)
print("RF: ", clf.score(Xte,Yte))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
KNN = KNeighborsClassifier(3)
KNN.fit(Xtr,Ytr)
print("KNN: ",KNN.score(Xte,Yte))

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

import numpy as np
gnb =GaussianNB()
gnb.fit(np.array(Xtr).astype(np.float), np.array(Ytr).astype(np.float))
print("GNB: ", gnb.score(np.array(Xte).astype(np.float),np.array(Yte).astype(np.float)))


from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(Xtr, Ytr)
print("DT: ", clf.score(Xte,Yte))

import sys
sys.exit(0)

