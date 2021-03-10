
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
dataset = load_iris()
dataset

iris = pd.DataFrame(dataset.data, columns = dataset.feature_names)
#print the first few rows
print(iris.head())

# total no of rows and columns
print(iris.shape)

# check for null values
print(iris.isnull().sum())

X = dataset.data
y = dataset.target

# between sepal length and width

plt.scatter(X[y == 0, 0], X[y == 0, 1], label = 'setosa')
plt.scatter(X[y == 1, 0], X[y == 1, 1], label = 'versicolor')
plt.scatter(X[y == 2, 0], X[y == 2, 1], label = 'virginica')
plt.legend(loc='upper right') 
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()


# between petal length and width

plt.scatter(X[y == 0, 2], X[y == 0, 3], label = 'setosa')
plt.scatter(X[y == 1, 2], X[y == 1, 3], label = 'versicolor')
plt.scatter(X[y == 2, 2], X[y == 2, 3], label = 'virginica')
plt.legend(loc='lower right') 
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()


from sklearn.cluster import KMeans

# Experimenting with some values of 'K' to deduce the optimal value

wcv = [] # within cluster variation which helps to find the optimum no of clusters

for i in range(1, 16):
    km = KMeans(n_clusters = i)
    km.fit(X)
    wcv.append(km.inertia_) # calculates wcv

# Now plotting a graph which shows us the 'elbow' i.e. a point after the graph changes from exponential to linear
    
plt.plot(range(1, 16), wcv)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCV') # within cluster variation also known as within cluster sum of squares
plt.show()


# Creating the kmeans classifier
km = KMeans(n_clusters = 3)
y_pred = km.fit_predict(X)
y_pred

iris['clusters']=y_pred
iris.head()

plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], label = 'versicolor')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], label = 'setosa')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], label = 'virginica')

# Plotting the centroids.

plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:,1], label = 'Centroids')
plt.legend(loc='upper right') 
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

plt.scatter(X[y_pred == 0, 2], X[y_pred == 0, 3], label = 'versicolor')
plt.scatter(X[y_pred == 1, 2], X[y_pred == 1, 3], label = 'setosa')
plt.scatter(X[y_pred == 2, 2], X[y_pred == 2, 3], label = 'virginica')

# Plotting the centroids. This time we're going to use the cluster centres 

plt.scatter(km.cluster_centers_[:, 2], km.cluster_centers_[:,3], label = 'Centroids')
plt.legend(loc='lower right') 
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()
