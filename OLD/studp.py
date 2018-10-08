from sklearn.decomposition import PCA, NMF
from sklearn import datasets
import sys

#iris = datasets.load_iris()
#print(iris)

import csv

data = []
label = []

arr = []

feats = []

with open('studp.csv', 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=';',quotechar='"')
	for row in reader:
		# ~ arr.append(list(map(lambda x: int(x), row)))
		arr.append(row)

feats = arr[0]
print(feats)

arr = arr[1:]

import numpy as np

print(np.array(arr)[:3,0])

# ~ sys.exit(0)

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


# ~ sys.exit(0)

arr = arr.tolist()

data = []
label = []
CLA = 33 -1

# ~ Attribute Information:

# ~ # Attributes for both student-mat.csv (Math course) and student-por.csv (Portuguese language course) datasets:
# ~ 1 school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
# ~ 2 sex - student's sex (binary: 'F' - female or 'M' - male)
# ~ 3 age - student's age (numeric: from 15 to 22)
# ~ 4 address - student's home address type (binary: 'U' - urban or 'R' - rural)
# ~ 5 famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)
# ~ 6 Pstatus - parent's cohabitation status (binary: 'T' - living together or 'A' - apart)
# ~ 7 Medu - mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education)
# ~ 8 Fedu - father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education)
# ~ 9 Mjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
# ~ 10 Fjob - father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
# ~ 11 reason - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')
# ~ 12 guardian - student's guardian (nominal: 'mother', 'father' or 'other')
# ~ 13 traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
# ~ 14 studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
# ~ 15 failures - number of past class failures (numeric: n if 1<=n<3, else 4)
# ~ 16 schoolsup - extra educational support (binary: yes or no)
# ~ 17 famsup - family educational support (binary: yes or no)
# ~ 18 paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
# ~ 19 activities - extra-curricular activities (binary: yes or no)
# ~ 20 nursery - attended nursery school (binary: yes or no)
# ~ 21 higher - wants to take higher education (binary: yes or no)
# ~ 22 internet - Internet access at home (binary: yes or no)
# ~ 23 romantic - with a romantic relationship (binary: yes or no)
# ~ 24 famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
# ~ 25 freetime - free time after school (numeric: from 1 - very low to 5 - very high)
# ~ 26 goout - going out with friends (numeric: from 1 - very low to 5 - very high)
# ~ 27 Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
# ~ 28 Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
# ~ 29 health - current health status (numeric: from 1 - very bad to 5 - very good)
# ~ 30 absences - number of school absences (numeric: from 0 to 93)

# ~ # these grades are related with the course subject, Math or Portuguese:
# ~ 31 G1 - first period grade (numeric: from 0 to 20)
# ~ 31 G2 - second period grade (numeric: from 0 to 20)
# ~ 32 G3 - final grade (numeric: from 0 to 20, output target)



print(feats[CLA],'can be predicted on: ')
for i in range(0,len(arr)):
	
	label.append(arr[i][CLA])
	
	del arr[i][CLA]
	
	data.append(arr[i])

del feats[CLA]


from sklearn.naive_bayes import GaussianNB


#BEFORE PCA
from sklearn.model_selection import train_test_split
Xtr, Xte, Ytr, Yte = train_test_split(data, label, test_size=0.1, random_state=42)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=7, random_state=0)
clf.fit(Xtr, Ytr)

for x in sorted(list(zip(feats,clf.feature_importances_)),key=lambda x: x[1],reverse=True):
	print(x)
print("RF: ", clf.score(Xte,Yte))

gnb = GaussianNB()
gnb.fit(np.array(Xtr).astype(np.float), np.array(Ytr).astype(np.float))
print("GNB: ", gnb.score(np.array(Xte).astype(np.float),np.array(Yte).astype(np.float)))


from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(Xtr, Ytr)
print("DT: ", clf.score(Xte, Yte))


le = preprocessing.LabelEncoder()
le.fit(np.array(label))
print(label[:10])
print(le.classes_)

import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feats, filled=True, rounded=True,class_names=le.classes_) 
graph = graphviz.Source(dot_data) 
graph.render("iris", view=True)





from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
KNN = KNeighborsClassifier(3)
KNN.fit(Xtr,Ytr)
print("KNN: ",KNN.score(Xte,Yte))


# ~ sys.exit(0)

print("AFTER PCA")
#AFTER PCA
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

chu = PCA(n_components=3)
# ~ chu = LinearDiscriminantAnalysis(n_components=1)
# ~ chu = NMF() #cool!
data = chu.fit_transform(np.array(data).astype(np.float),np.array(label).astype(np.float))
# ~ print(data[:2])

from sklearn.model_selection import train_test_split
Xtr, Xte, Ytr, Yte = train_test_split(data, label, test_size=0.1, random_state=42)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=7, random_state=0)
clf.fit(Xtr, Ytr)
print("RF: ", clf.score(Xte,Yte))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
KNN = KNeighborsClassifier(3)
KNN.fit(Xtr,Ytr)
print("KNN: ",KNN.score(Xte,Yte))


import sys
sys.exit(0)
