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

# Значимость признаков для Wine, на основе RandomForest

#сделать функция для оценки значимости на основе RF

dt = pd.read_csv('white.csv',sep=';')

#~ print(dt.head(3))


def importance(dt):
	X = dt.iloc[:,:-1].values
	Y = dt.iloc[:,-1].values
	
	#~ print(X[:3])
	#~ print(Y[:3])
	
	Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.1, random_state=42)
	
	clf = RandomForestClassifier(n_estimators=50, random_state=0)
	#~ clf = tree.DecisionTreeClassifier(max_depth=10)
	clf.fit(Xtr, Ytr)

	imp = clf.feature_importances_
	col = dt.columns.values
	
	for x in sorted(list(zip(col, imp)),key=lambda x: x[1],reverse=True):
		print(x)
	
	col = [a for a,b in list(sorted(list(zip(col, imp)),key=lambda x: x[1],reverse=False))]
	imp = [b for a,b in list(sorted(list(zip(col, imp)),key=lambda x: x[1],reverse=False))]
	
	plt.title('Feature Importances, RF, score='+str(clf.score(Xte,Yte)))
	plt.barh(col, imp, color='b', align='center')
	plt.xlabel('Relative Importance')
	plt.show()
	
	
def RFE_importance(dt):
	X = dt.iloc[:,:-1].values
	Y = dt.iloc[:,-1].values
	
	#~ print(X[:3])
	#~ print(Y[:3])
	
	Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.1, random_state=42)
	
	clf = RandomForestClassifier(n_estimators=50, random_state=0)
	#~ clf = tree.DecisionTreeClassifier(max_depth=10)
	#~ clf.fit(Xtr, Ytr)
	
	rfe = RFE(estimator=clf, n_features_to_select=1, step=1)
	rfe.fit(X, Y)
	col = dt.columns.values
	imp = (len(rfe.ranking_)+1-np.array(rfe.ranking_))/np.sum(rfe.ranking_)
	
	
	ncol = [a for a,b in list(sorted(list(zip(col, imp)),key=lambda x: x[1],reverse=False))]
	nimp = [b for a,b in list(sorted(list(zip(col, imp)),key=lambda x: x[1],reverse=False))]
	
	plt.title('Feature Importances, RFE')
	plt.barh(ncol, nimp, color='b', align='center')
	plt.xlabel('Relative Importance')
	plt.show()
	
	
def RFECV_importance(dt):
	X = dt.iloc[:,:-1].values
	Y = dt.iloc[:,-1].values
	
	#~ print(X[:3])
	#~ print(Y[:3])
	
	Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.1, random_state=42)
	
	clf = RandomForestClassifier(n_estimators=10, random_state=0)
	#~ clf = tree.DecisionTreeClassifier(max_depth=10)
	#~ clf.fit(Xtr, Ytr)
	rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(10), scoring='accuracy')
	rfecv.fit(X, Y)
	print("Optimal number of features : %d" % rfecv.n_features_)
	
	print(rfecv.support_)
	#~ [False  True False  True False  True  True  True  True False  True]
	
	print(rfecv.ranking_)
	#~ [5 1 4 1 3 1 1 1 1 2 1]
	
	#~ print(dt.drop(columns=['quality']).columns.values)
	
	print(dt.iloc[:, :-1].columns.values[rfecv.support_])
	
	plt.figure()
	plt.xlabel("Number of features selected")
	plt.ylabel("Cross validation score (nb of correct classifications)")
	plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
	plt.show()
	
	newcols = list(dt.iloc[:, :-1].columns.values[rfecv.support_])
	newcols.append(dt.columns.values[-1])
	
	return newcols
	
	#~ rfe = RFE(estimator=clf, n_features_to_select=1, step=1)
	#~ rfe.fit(X, Y)
	#~ col = dt.columns.values
	#~ imp = (len(rfe.ranking_)+1-np.array(rfe.ranking_))/np.sum(rfe.ranking_)
	
	
	#~ ncol = [a for a,b in list(sorted(list(zip(col, imp)),key=lambda x: x[1],reverse=False))]
	#~ nimp = [b for a,b in list(sorted(list(zip(col, imp)),key=lambda x: x[1],reverse=False))]
	
	#~ plt.title('Feature Importances, RFE')
	#~ plt.barh(ncol, nimp, color='b', align='center')
	#~ plt.xlabel('Relative Importance')
	#~ plt.show()


def classify(dt):
	X = dt.iloc[:,:-1].values
	Y = dt.iloc[:,-1].values
	Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.1, random_state=42)
	clf=None
	
	clas = []
	scores = []
	
	clf = RandomForestClassifier(n_estimators=50, random_state=0)
	clf.fit(Xtr, Ytr)
	clas.append("RandomForest")
	scores.append(clf.score(Xte,Yte))
	
	clf = tree.DecisionTreeClassifier(max_depth=5)
	clf.fit(Xtr, Ytr)
	clas.append("DecisionTree")
	scores.append(clf.score(Xte,Yte))
	
	clf = KNeighborsClassifier(1)
	clf.fit(Xtr, Ytr)
	clas.append("KNN(1)")
	scores.append(clf.score(Xte,Yte))
	
	clf = KNeighborsClassifier(2)
	clf.fit(Xtr, Ytr)
	clas.append("KNN(2)")
	scores.append(clf.score(Xte,Yte))
	
	clf = KNeighborsClassifier(3)
	clf.fit(Xtr, Ytr)
	clas.append("KNN(3)")
	scores.append(clf.score(Xte,Yte))
	
	clf = KNeighborsClassifier(5)
	clf.fit(Xtr, Ytr)
	clas.append("KNN(5)")
	scores.append(clf.score(Xte,Yte))
	
	clf = SVC(C=10.0)
	clf.fit(Xtr, Ytr)
	clas.append("SVC, C=10.0")
	scores.append(clf.score(Xte,Yte))
	
	clf = SVC(C=1.0)
	clf.fit(Xtr, Ytr)
	clas.append("SVC, C=1.0")
	scores.append(clf.score(Xte,Yte))
	
	clf = SVC(C=0.1)
	clf.fit(Xtr, Ytr)
	clas.append("SVC, C=0.1")
	scores.append(clf.score(Xte,Yte))
	
	clf = LogisticRegression(random_state=0, solver='lbfgs',max_iter=200, multi_class='auto')
	clf.fit(Xtr, Ytr)
	clas.append("LogisticRegression")
	scores.append(clf.score(Xte,Yte))
	
	clf = LogisticRegression(penalty='l1',random_state=0,max_iter=200, multi_class='auto')
	clf.fit(Xtr, Ytr)
	clas.append("LogisticRegression, L1")
	scores.append(clf.score(Xte,Yte))
	
	clf = LogisticRegression(penalty='l2',random_state=0, solver='lbfgs',max_iter=200, multi_class='auto')
	clf.fit(Xtr, Ytr)
	clas.append("LogisticRegression, L2")
	scores.append(clf.score(Xte,Yte))

	plt.title("Classification validation score")
	plt.barh(clas, scores, color='b', align='center')
	plt.xlabel('Relative Importance')
	plt.show()



def LR(dt):
	X = dt.iloc[:,:-1].values
	Y = dt.iloc[:,-1].values
	Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.1, random_state=42)
	clf=None
	clf = LogisticRegression(penalty='l2',C=1.0, random_state=0, solver='lbfgs',max_iter=100, multi_class='auto')
	clf.fit(Xtr, Ytr)
	print("feature importances(LR): ", clf.coef_)

def FFS(dt):

	col = dt.columns.values
	
	selected = []
	
	for k in range(0, len(col)-1):
		maxidx = -1
		maxscore = -1
		for i in range(0,len(col)-1):
			#~ print(col[i])
			if not (i in selected):
				newsel = copy.deepcopy(selected)
				newsel.append(i)
				#~ print(newsel)
				newsel.append(len(col)-1)
				
				
				st=dt[col[newsel]]
				
				#~ print(st.head(2))
				
				X = st.iloc[:,:-1].values
				Y = st.iloc[:,-1].values
				
				Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.1, random_state=42)
				
				rfe = RandomForestClassifier(n_estimators=50, random_state=0)
				#~ rfe = KNeighborsClassifier(5)
				rfe.fit(Xtr, Ytr)
				
				score = rfe.score(Xte,Yte)
				
				if score >= maxscore:
					#~ print(score,maxscore,i)
					maxscore=score
					maxidx=i
		
		#~ if not maxidx in selected:
		if not (maxidx in selected):
			selected.append(maxidx)
			print("RF score: {}, \n\tfeatures:{}, \n\ttheir names:{}\n".format(maxscore, selected, col[selected]))
	return selected

#~ LR(dt)

#~ FFS(dt)


#~ cols = RFECV_importance(dt)
#~ print(cols)
#~ dt=dt[cols]

#~ RFE_importance(dt)

#~ importance(dt)

#~ classify(dt)

#~ dt=dt.drop(columns=['fixed acidity','sulphates'])

#~ classify(dt)




import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits as load_data
import scikitplot as skplt


X, y = load_data(return_X_y=True)
nb = GaussianNB()


X = dt.iloc[:,:-1].values
Y = dt.iloc[:,-1].values
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.1, random_state=42)
clf=None
clf = LogisticRegression(penalty='l2',C=1.0, random_state=0, solver='lbfgs',max_iter=100, multi_class='auto')
clf.fit(Xtr, Ytr)
print("feature importances(LR): ", clf.coef_)


probas = clf.predict_proba(X)
Ypr = clf.predict(Xte)
#~ skplt.metrics.plot_confusion_matrix(Yte, Ypr, normalize=True)
skplt.metrics.plot_roc(y_true=Y, y_probas=probas, plot_macro=False)
pca = PCA(random_state=1)
pca.fit(X)
skplt.decomposition.plot_pca_2d_projection(pca, X, Y)

plt.show()



sys.exit(0)


















clf = KNeighborsClassifier(3)




print("RF without low features: ", clf.score(Xte,Yte))

dt = dt.drop(columns=['sulphates'])
#~ dt = dt.drop(columns=['fixed acidity'])
print(dt.head(3))

X = dt.iloc[:,:-1].values
Y = dt.iloc[:,-1].values

Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.1, random_state=42)

#~ clf = RandomForestClassifier(n_estimators=50, random_state=0)
clf = KNeighborsClassifier(3)
clf = tree.DecisionTreeClassifier(max_depth=5)

clf.fit(Xtr, Ytr)

print("RF w/o low features: ", clf.score(Xte,Yte))



features = dt.columns
importances = clf.feature_importances_
indices = np.argsort(importances)[:-1]  # top 10 features

plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.show()


clf = ExtraTreesClassifier()


print(dt.columns.values)
print(clf.feature_importances_)


clf.fit(Xtr, Ytr)

print("RF w/o low features: ", clf.score(Xte,Yte))


plt.figure()
plt.title("Feature importances")
plt.bar(range(len(imp)), imp,
       color="r", align="center")
plt.xticks(range(len(col)), col, rotation='vertical')
plt.show()

print("RF: ", clf.score(Xte,Yte))

