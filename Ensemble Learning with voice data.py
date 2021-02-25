#!/usr/bin/env python
# coding: utf-8

# # Welcome!
# 
# In this lab section we will be utilizing the data from https://www.kaggle.com/primaryobjects/voicegender#
# 
# Do note that there is some tricky ethical stuff here considering that the voice of Philippe Jaroussky could be classified as Female (Countertenor) while the voice of Nathalie Stutzmann might be classified as Male (Contralto).
# 
# 
# For the exercise we will be following my framework TACT which is to Target, Arrange, Compose, and Transmit.
# 
# Read more about it here: tactmethod.com
# 

# # What is our target?
# 
# In general we want to achieve a highly precise classifier utilizing voice data (as close to 100% as possible).
# 
# How do we go about doing this? There are a few directions to go in this exercise either we do feature selection, feature transformation, or ensemble learning.
# 
# 

# # Arrange data
# 
# The first step is to load up the data
# 
# Also it's good to note what exactly is in it:
# 
# ## The Dataset
# The following acoustic properties of each voice are measured and included within the CSV:
# 
# * meanfreq: mean frequency (in kHz)
# * sd: standard deviation of frequency
# * median: median frequency (in kHz)
# * Q25: first quantile (in kHz)
# * Q75: third quantile (in kHz)
# * IQR: interquantile range (in kHz)
# * skew: skewness (see note in specprop description)
# * kurt: kurtosis (see note in specprop description)
# * sp.ent: spectral entropy
# * sfm: spectral flatness
# * mode: mode frequency
# * centroid: frequency centroid (see specprop)
# * peakf: peak frequency (frequency with highest energy)
# * meanfun: average of fundamental frequency measured across acoustic signal
# * minfun: minimum fundamental frequency measured across acoustic signal
# * maxfun: maximum fundamental frequency measured across acoustic signal
# * meandom: average of dominant frequency measured across acoustic signal
# * mindom: minimum of dominant frequency measured across acoustic signal
# * maxdom: maximum of dominant frequency measured across acoustic signal
# * dfrange: range of dominant frequency measured across acoustic signal
# * modindx: modulation index. Calculated as the accumulated absolute difference between adjacent measurements of fundamental frequencies divided by the frequency range
# * label: male or female (Note that we don't want to classify on this!)

# In[1]:


import pandas as pd

df = pd.read_csv('../data/voice.csv')

y = df['label']
X = df[set(df.columns) - set(['label'])]


# In[2]:


X


# # Strawman Model
# 
# Always always start with a baseline model. What is a decent enough model and what isn't?
# 
# KNN and Naive Bayesian models are pretty easy to utilize and generally get someone started
# 
# Can we make this better?

# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report

# K-Nearest Neighbors is usually where the introduction class leaves off
X_train, X_test, y_train, y_test = train_test_split(X, y)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
nb = BernoulliNB()
nb.fit(X_train, y_train)

print("KNN")
print(classification_report(knn.predict(X_test), y_test))
print("Naive Bayesian")
print(classification_report(nb.predict(X_test), y_test))


# # Thoughts?
# 
# What are your thoughts on the above classification report. Note that when we look at KNN the precision is around 70% which is ok, but not great. Can we do better by selecting better features? OR transforming features?

# # Feature Selection
# 
# Looking at the data what is a good feature?
# 
# In general there are a few directions to go here. We can look at Feature Importance by running this through a classifier, or we could look at variance thresholds

# In[4]:


from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

# If we take the data and fit it into a small support vector machine
# We can figure out what has the highest weighting and then turn it into a new transformer
# SelectFromModel does the heavy lifting

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
X_new.shape


# In[5]:


knn = KNeighborsClassifier()
knn.fit(model.transform(X_train), y_train)
nb = BernoulliNB()
nb.fit(model.transform(X_train), y_train)

print("KNN")
print(classification_report(knn.predict(model.transform(X_test)), y_test))
print("Naive Bayesian")
print(classification_report(nb.predict(model.transform(X_test)), y_test))


# In[7]:


from sklearn.ensemble import ExtraTreesClassifier

# Tree based models can also determine what is most indicative by looking at feature importance
# Feature importances are usually what are the attributes that yield the biggest change to the outcome

clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X, y)
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
X_new.shape


# In[ ]:





# In[8]:


knn = KNeighborsClassifier()
knn.fit(model.transform(X_train), y_train)
nb = BernoulliNB()
nb.fit(model.transform(X_train), y_train)

print("KNN")
print(classification_report(knn.predict(model.transform(X_test)), y_test))
print("Naive Bayesian")
print(classification_report(nb.predict(model.transform(X_test)), y_test))


# # What do you get from the above classification?
# 
# Note that having a precision of 97% purely based on selecting the right features is pretty good!
# 
# A tree based approach seems to yield some fruit.

# In[11]:


# Visualize the differences\
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

pd.Series(dict(zip(X.columns, clf.feature_importances_))).plot.bar()
plt.show()
pd.Series(dict(zip(X.columns, lsvc.coef_[0]))).plot.bar()
plt.show()
# view.plot.bar()


# # What do you learn from this?
# 
# Looking at the plots above what can you learn? What is the most important feature as it relates to gendered voice given tree classifications?
# 
# What is the most important coefficient for a linear model?

# # Feature Transformation
# 
# Above we selected features that are most likely to help us classify. But what if we don't want to give up on features? What if we want to keep it all?
# 
# There are generally a few directions we can go in.
# 
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
# 
# Here's a bunch of options ranked from simplistic to state of the art
# 
# * PCA (Principal Component Analysis)
# * ICA (Independent Component Analysis)
# * UMAP (Uniform Manifold Approximation and Projection for Dimension Reduction)

# In[12]:


# PCA (usually 2 components is a start)
from sklearn.decomposition import PCA

model = PCA(n_components=4)
model.fit(X)

knn = KNeighborsClassifier()
knn.fit(model.transform(X_train), y_train)
nb = BernoulliNB()
nb.fit(model.transform(X_train), y_train)

print("KNN")
print(classification_report(knn.predict(model.transform(X_test)), y_test))
print("Naive Bayesian")
print(classification_report(nb.predict(model.transform(X_test)), y_test))


# In[13]:


from sklearn.decomposition import FastICA

model = FastICA(4)
model.fit(X)

knn = KNeighborsClassifier()
knn.fit(model.transform(X_train), y_train)
nb = BernoulliNB()
nb.fit(model.transform(X_train), y_train)

print("KNN")
print(classification_report(knn.predict(model.transform(X_test)), y_test))
print("Naive Bayesian")
print(classification_report(nb.predict(model.transform(X_test)), y_test))


# In[14]:


import umap

model = umap.UMAP()
model.fit(X)

knn = KNeighborsClassifier()
knn.fit(model.transform(X_train), y_train)
nb = BernoulliNB()
nb.fit(model.transform(X_train), y_train)

print("KNN")
print(classification_report(knn.predict(model.transform(X_test)), y_test))
print("Naive Bayesian")
print(classification_report(nb.predict(model.transform(X_test)), y_test))


# # Not the best results but what did we learn?
# 
# In the above sections we tried out selecting better features and even transforming the existing features into a new matrix. The best approach from above was using KNN with Tree based selection. While that's true, can we utilize a better algorithm to get even better results?

# # Ensemble Learning
# 
# In general ensemble learning shows up either as bagging, or boosting, sometimes bayesian selection.
# 
# In this section we'll show the results of Bagging first and then Random Forests and then Boosting using XgBoost (state of the art).

# In[15]:


from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

model = BaggingClassifier()
model.fit(X_train, y_train)

print(classification_report(model.predict(X_test), y_test))


# # Great results! can we do better?

# In[12]:


from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('feature_selection', SelectFromModel(LinearSVC(C=0.01))),
    ('bagger', BaggingClassifier())])


pipe.fit(X_train, y_train)

print(classification_report(pipe.predict(X_test), y_test))


# In[13]:


from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('feature_transformer', FastICA(15)),
    ('bagger', BaggingClassifier())])


pipe.fit(X_train, y_train)

print(classification_report(pipe.predict(X_test), y_test))


# # There are endless possibilities
# 
# Note that this is an area of research: AutoML. Basically there's so many options, so many possibilities it'd take forever to explain them all.

# In[14]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

clf.fit(X_train, y_train)

print(classification_report(clf.predict(X_test), y_test))


# In[15]:


from xgboost import XGBClassifier

clf = XGBClassifier()
clf.fit(X_train, y_train)

print(classification_report(clf.predict(X_test), y_test))


# In[17]:


from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('transformer', PCA(10)),
    ('estimator', XGBClassifier())
])

pipe.fit(X_train, y_train)


print(classification_report(pipe.predict(X_test), y_test))

