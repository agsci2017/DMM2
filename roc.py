import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import copy
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys
import pandas as pd

dt = pd.read_csv('white.csv',sep=';')

X = dt.iloc[:,:-1].values
Y = dt.iloc[:,-1].values

# приводим значения label'а к виду 0 0 1 1 0
def myfunc(a):
	if a > 5:
		return 1
	else:
		return 0
		
vfunc = np.vectorize(myfunc)
Y=vfunc(Y)


#классифицируем
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.1, random_state=42)

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto')
clf.fit(Xtr, Ytr)

#получаем вероятности
Ypr = clf.predict_proba(Xte)[:,1]

#получаем значения FPR, TPR
fpr, tpr, thresholds = metrics.roc_curve(Yte, Ypr)


import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1],'r--', label='50% accuracy, bad classifier')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 'lower right')
plt.title('AUC = {:4.2f}'.format(metrics.auc(fpr,tpr)))
plt.plot(fpr,tpr)
plt.show()
