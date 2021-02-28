#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset=pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,0].values
Y = dataset.iloc[:,1].values

#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#train simple linear regression
from sklearn.linear_model import LinearRegression
linear_regression=LinearRegression()
linear_regression.fit(X_train.reshape(-1, 1),Y_train)

#predict
y_pred = linear_regression.predict(X_test.reshape(-1,1))

#visualise training set result and test set result
#training set
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,linear_regression.predict(X_train.reshape(-1,1)),color='blue')
plt.title('Salary Vs Experience (Training Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#test set
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_test,y_pred,color='blue')
plt.title('Salary Vs Experience (Test Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#print regressiong equation
print("Regressiong equation is:")
print("Salary = {} + {}*Years_of_Experience".format(linear_regression.intercept_,linear_regression.coef_))
