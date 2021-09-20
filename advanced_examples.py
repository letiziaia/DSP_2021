"""
Some code snippets are taken from the tutorial of the Autumn 2019 course CS-C3160 Data Science
"""

##################################################################################
# NUMPY: See also PFDA CHAPTER 4                                                 #
##################################################################################

import numpy as np

# NUMPY examples

p_list = [1, 2, 3]  # this is a Python list
np_array = np.array(p_list)  # creating numpy array from a list

# Checking the difference of list and np.array by type
print('list type:', type(p_list), 'numpy array type:', type(np_array), '\n')

# Adding list+list vs numpy.array+numpy.array
print('List+list is:        ', p_list + p_list)  # Python concatenate lists
print('np.array+np.array is:', np_array + np_array)  # numpy performs element-wise addition on arrays

# Create a multidimensional numpy array
a = np.array([[1, 2, 3], [4, 5, 6]])

# Check the shape of a numpy array
print('Shape of the array:', a.shape)

# Check the number of dimensions
print('Number of dimensions:', a.ndim)

# Check the number of elements
print('Number of elements:', a.size)

# Reshaping
test_array = np.array([7, 8, 15])
print('Current shape:', test_array.shape)
vector = test_array.reshape(1, 3)  # from 1D -> 2D
print('New shape:', vector.shape)  # checking the shape is (1,3)

print('Array:', test_array)
print('Array transpose:', test_array.T)  # vectors can be transposed
print('Vector:', vector)
print('Vector transpose:\n', vector.T)

# Create 2x2 matrix
A = np.array([[1, 2], [3, 4]])
print(A)

# Create 3x2 matrix
B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)

# Linear algebra

# DO NOT MULTIPLY MATRICES!
print("Element-wise:\n", A * A)

# DO matrix multiplication!
print('A * A (correct matrix multiplication):\n', A @ A)

# Example of dot product
print('The result is the same as A @ A:\n', A.dot(A))

# If a matrix is invertible, np.linalg.inv(matrix) will compute the inverse
# Note: multiplying the inverse and the original matrix should yield the identity matrix,
# but there might be some floating-point error.
print('inv(A) * A:\n', np.linalg.inv(A) @ A)

##################################################################################
# PANDAS: See also PFDA CHAPTER 5                                                #
##################################################################################

import pandas as pd

# PANDAS examples

# Read a csv file
df = pd.read_csv("data/iris.csv")

# Print first 5 rows of the dataframe
print(df.head())

# Overview: summary statistics
print(df.describe(include="all"))

# Overview: which columns, how many values, memory usage
print(df.info())

# Columns of dataframe can be accessed by names or with dot notation (if there is no space in column name)
print(df['sepal.length'])
print(df.variety)

# Rows and columns can be accessed using .iloc (by numbers) or .loc (by labels).
print(df.iloc[0])

# A single value in the dataframe can be accessed the same way:
print('\n', df.iloc[0, 1])
print('\n', df.iloc[5]["variety"])

# Iterating by rows
for i, row in df.iterrows():
    print(f'Row {i}. Contents:\n{row}\n')

# unique values
print(df["variety"].unique())

# Group by
print(df.groupby("variety").count())

print(df.groupby("variety").groups.keys())

# In-built stats
print(df.median())

print(df.corr(method="kendall"))

# From one categorical column, get multiple columns as one-hot
print(pd.get_dummies(df["variety"]))

# Transform the type of a pandas column
# Here we change object to category
df["variety"] = df["variety"].astype(dtype="category")

# pandas can encode categories with numbers
print(df.variety.cat.codes)

##################################################################################
# PLOTTING: See also PFDA CHAPTER 9                                                   #
##################################################################################

# MATPLOTLIB

import matplotlib.pyplot as plt

# Plot directly from pandas
df.plot.scatter(x="sepal.width", y="sepal.length", c="petal.width", s="petal.length")
plt.show()

df.boxplot()
plt.show()

df.boxplot(by="variety")
plt.show()

df.plot.kde()
plt.show()

df.value_counts(subset=["variety"], normalize=True).plot(kind="pie")
plt.show()

df.value_counts(subset=["variety"], normalize=True).plot(kind="bar")
plt.show()

# From matplotlib
plt.scatter(x=df["sepal.width"], y=df["petal.width"])
plt.xlabel("sepal width")
plt.ylabel("petal width")
plt.show()


plt.scatter(x=df["sepal.width"], y=df["petal.width"], c=df.variety.cat.codes)
plt.xlabel("sepal width")
plt.ylabel("petal width")
plt.show()


plt.hist2d(x=df["sepal.width"], y=df["petal.width"])
plt.xlabel("sepal width")
plt.ylabel("petal width")
plt.show()

# SEABORN

import seaborn as sns

# From seaborn
sns.histplot(df, x=df.columns[0], hue="variety")
plt.show()

# From seaborn
sns.boxplot(x=df["sepal.length"], y=df["variety"])
plt.show()

# From seaborn
sns.FacetGrid(df, hue='variety').map(plt.scatter, 'sepal.length', 'sepal.width').add_legend()
plt.show()

# From seaborn
g = sns.PairGrid(df, hue="variety")
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.add_legend()
plt.show()

# From plotly

import plotly.express as px

fig = px.scatter(df, x=df.columns[0], y=df.columns[1], color=df.variety, size=df.columns[3])
fig.show()

fig = px.scatter_3d(df, x=df.columns[0], y=df.columns[1], z=df.columns[2], color=df.variety, size=df.columns[3])
fig.show()

##################################################################################
# MACHINE LEARNING, CLASSIFICATION: See also PFDA 13.4                           #
##################################################################################

# Sci-Kit Learn: Classification

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score

X_train, X_test, y_train, y_test = train_test_split(df[df.columns[:4]], df["variety"], test_size=0.33, random_state=42)

clf = SVC()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(balanced_accuracy_score(y_test, y_pred))

# Which classes does my classifier know?
print(clf.classes_)

# See parameters
print(clf.get_params())


# Do not expect to get accuracy 1 in real life!
# Let's see an example with some wrong predictions

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(balanced_accuracy_score(y_test, y_pred))

# class_prior is specific for GaussianNB
print(clf.class_prior_)

# Some classifiers compute the class label as probability
# In that case, you can ask your classifier to give you the probability of belonging to each class
y_pred_probabilities = clf.predict_proba(X_test)

# There are 3 classes, that's why the shape is:
print(y_pred_probabilities.shape)

##################################################################################
# MACHINE LEARNING, TIME SERIES: See also PFDA CHAPTER 11                        #
##################################################################################

df = pd.read_csv("data/airpassengers.csv")
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)
print(df.head())
print(df.info())

sns.lineplot(x=df.index, y=df["value"])
plt.show()

import statsmodels.api as sm

# detrend and plot the time series with no trend
sns.lineplot(x=df.index, y=sm.tsa.detrend(df["value"]))
plt.show()

# line plot with plotly
fig = px.line(x=df.index, y=df["value"])
fig.show()