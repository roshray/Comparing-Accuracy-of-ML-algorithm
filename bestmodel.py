#load libraries

import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset 

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

#Dimension of dataset
#shape

print(dataset.shape)

#peek at the data
#head

print(dataset.head(20))

#description

print(dataset.describe())

#class distribution

print(dataset.groupby('class').size())

#DATA VISUALIZATION

""" Data need to extend  with some visualizations.

We are going to look at two types of plots:

1)Univariate plots to better understand each attribute.

2)Multivariate plots to better understand the relationships between attributes."""

#univariant plots
#box and whisker plots

dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
plt.show()

#histogram(for idea of distribution)

dataset.hist()
plt.show()

#multivariant plots
#scatter plot matrix

scatter_matrix(dataset)
plt.show()

#create a validation dataset
#Split-out validation dataset(80% & 20%)
#seed ????

array=dataset.values
X =array[:,0:4]
Y =array[:,4]
validation_size =0.20
seed = 7
X_train, X_validation, Y_train, Y_validation=cross_validation.train_test_split(X,Y,test_size=validation_size,random_state=seed)


#Test Harness
#Test options and evalution metric

num_folds =10
num_instances =len(X_train)
seed =7
scoring ='accuracy'

#Build Models
#simple linear (LR and LDA), nonlinear (KNN, CART, NB and SVM) algorithms
#evaluting 6 different algorithms
#Spot check algorithms

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))

#evaluate each model in turn
#??study about the cross validation

results = []
names = []

for name,model in models:
   Kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
   cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=Kfold, scoring=scoring)
   results.append(cv_results)
   names.append(name)
   msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())   #need to know
   print(msg)

#select best model
#from accuracy best algo is know (svm or knn)
#compare algorithms accuracy

fig = plt.figure()
fig.suptitle('Algorithm Comparision')
ax = fig.add_subplot(111) #?111
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#Make prediction on validation dataset

SVM = SVC()
SVM.fit(X_train, Y_train)
predictions = SVM.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

































































