import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt

dataset = pd.read_csv("C:\\Users\\swago\\Desktop\\Sparks Foundation\\scores.csv")
print("Data imported successfully")

#printing head( first few rows) of the dataset 
print("The first few values of the dataset are :")
print(dataset.head(5))

#the datatype of the columns
print("the dtypes of dataset :")
print(dataset.dtypes)

#the total no of rows and columns
print("The total no of rows and columns:")
print(dataset.shape)
#basic statisical details
print("basic statisical details")
print(dataset.describe())

#checking for any null values
print("checking for null values")
print(dataset.isnull().sum())

# Plotting the distribution of scores
dataset.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 1].values 

X.reshape(-1,1)
y.reshape(-1,1)

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y)
from sklearn.linear_model import LinearRegression  
lin_reg = LinearRegression() 

# to tell the algorithm also which data to work on we use the fit function
lin_reg.fit(X_train, y_train)
# Visualising the Training dataset 
plt.scatter(X_train,y_train)
plt.title('Training set')  
plt.plot(X_train,lin_reg.predict(X_train))
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score') 
plt.show()

# Accuracy of training set
lin_reg.score(X_train, y_train)

# Plotting the regression line
line = lin_reg.coef_*X + lin_reg.intercept_
plt.scatter(X, y)
plt.title('Regression Line') 
plt.plot(X, line)
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score') 
plt.show()

print(X_test) # Testing data - In Hours
y_pred = lin_reg.predict(X_test) # Predicting the scores


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df

hours = float(input('Enter number of hours a student is studying in a day '))
own_pred = lin_reg.predict([[hours]])
print("Predicted Score = {}".format((own_pred)[0]))

# Accuracy of test set
lin_reg.score(X_test, y_test)

from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))

# Visualising the test set 
plt.scatter(X_test,y_test)
plt.title('Test set')  
plt.plot(X_train,lin_reg.predict(X_train))
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score') 
plt.show()
