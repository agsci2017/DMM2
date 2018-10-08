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

# ~ fig = plt.figure()
# ~ ax = fig.add_subplot(111)
# ~ cax=ax.imshow(dt.corr(),cmap='bwr')
# ~ fig.colorbar(cax)
# ~ ticks = np.arange(0,len(dt.columns.values),1)
# ~ ax.set_xticks(ticks)
# ~ ax.set_yticks(ticks)
# ~ plt.xticks(rotation=90)
# ~ ax.set_xticklabels(dt.columns.values,fontsize=6)
# ~ ax.set_yticklabels(dt.columns.values,fontsize=6)

# ~ plt.show()

print(dt)


print(dt.describe().stack().to_csv('sxxx.csv'))
print(dt.describe().unstack().to_csv('uxxx.csv'))




X = dt.iloc[:,:-1].values
Y = dt.iloc[:,-1].values
print(X)

from sklearn.model_selection import train_test_split
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.1, random_state=42)


# ~ sc = preprocessing.StandardScaler()
# ~ Xtr = sc.fit_transform(Xtr)
# ~ Xte = sc.fit_transform(Xte)



# ~ min_max_scaler = preprocessing.MinMaxScaler()
# ~ x_scaled = min_max_scaler.fit_transform(X)
# ~ X = pd.DataFrame(x_scaled)

X=preprocessing.scale(X)
Y=preprocessing.scale(Y)


from sklearn.svm import SVR

# ~ svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
# ~ svr_lin = SVR(kernel='linear', C=1e3)
# ~ svr_poly = SVR(kernel='poly', C=1e3, degree=2)

from sklearn.decomposition import PCA, KernelPCA
# ~ from sklearn.decomposition import PCA
# ~ pca = PCA(n_components=5)
pca = KernelPCA(kernel="rbf", gamma=0.1, n_components=4)
X = pca.fit_transform(X)
# ~ X = pca.fit_transform(X)
print("PCA done")

# ~ X=preprocessing.scale(X)

# ~ from mpl_toolkits.mplot3d import Axes3D
# ~ fig = plt.figure()
# ~ ax = fig.add_subplot(111, projection='3d')

# ~ ax.scatter(X[:,0],X[:,1],X[:,2], c=Y)
# ~ plt.show()

# ~ from sklearn.preprocessing import StandardScaler
# ~ sc_x = StandardScaler()
# ~ sc_y = StandardScaler()
# ~ # Scale x and y (two scale objects)
# ~ x = sc_x.fit_transform(x)


# ~ svr = SVR(kernel='rbf', C=1000.0, epsilon=0.1)



# ~ from sklearn.linear_model import LinearRegression
# ~ svr = LinearRegression()
# ~ svr.fit(Xtr, Ytr)
# ~ Ypred = svr.predict(Xte)

# ~ from sklearn.metrics import mean_squared_error as mse
# ~ print("MSE: ",mse(Yte,Ypred))

# ~ plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')

# ~ print('fitted')
# ~ print("R2 score: ",svr.score(Xte, Yte))




# ~ import autosklearn.regression
# ~ automl = autosklearn.regression.AutoSklearnRegressor(
        # ~ time_left_for_this_task=2000,
        # ~ per_run_time_limit=350
    # ~ )
# ~ automl.fit(Xtr, Ytr)
# ~ print(automl.show_models())
# ~ Ypr = automl.predict(Xte)
# ~ from sklearn import metrics
# ~ print("R2 score:", metrics.r2_score(Yte, Ypr))



from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score


# ~ rfc = SGDClassifier(penalty=None)
rfc = RandomForestClassifier(n_estimators=200)
# ~ rfc = SVC(C=1000,gamma=0.05)
from sklearn.linear_model import Ridge
# ~ rfc = Ridge(alpha=1.7)
rfc.fit(Xtr, Ytr)
Ypr = rfc.predict(Xte)


# ~ for x in sorted(list(zip(dt.columns.values,rfc.feature_importances_)),key=lambda x: x[1],reverse=True):
	# ~ print(x)
#Let's see how our model performed

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

