# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and read the dataframe.
2. Assign hours to X and scores to Y.
3. Implement training set and test set of the dataframe.
4. Plot the required graph both for test data and training data.
5. Find the values of MSE , MAE and RMSE.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Keerthika.N
RegisterNumber: 212221230049
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
#displaying the content in datafile
df.head()

df.tail()

#segregating data to variables
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y

#splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred

#displaying actual values
Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color="Blue")
plt.plot(X_train,regressor.predict(X_train),color="Pink")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(X_train,Y_train,color="Black")
plt.plot(X_train,regressor.predict(X_train),color="Pink")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
* df.head()
<img width="104" alt="1" src="https://user-images.githubusercontent.com/93427089/229064210-55c661ee-0b92-4cbe-b39f-19acb8bd0ee1.png">

* df.tail()
<img width="117" alt="2" src="https://user-images.githubusercontent.com/93427089/229064501-8b4193d0-6410-4492-90ac-c777250ab40d.png">

* Array value of X
<img width="416" alt="x" src="https://user-images.githubusercontent.com/93427089/229065017-f2e8aa6c-398e-4919-bd77-5a4e8e98f8cd.png">

* Array value of Y
<img width="440" alt="y" src="https://user-images.githubusercontent.com/93427089/229065059-cdb4d802-22f6-4120-bfe5-4a26b0153c8d.png">

* Values of Y prediction
<img width="437" alt="1" src="https://user-images.githubusercontent.com/93427089/229065535-e115c402-4065-4e9c-b786-20fe99d034cd.png">

* Array values of Y test
<img width="257" alt="2" src="https://user-images.githubusercontent.com/93427089/229065586-7bcda2a2-ef81-49e9-bf37-df98e98f87f8.png">

* Training Set Graph
<img width="464" alt="3" src="https://user-images.githubusercontent.com/93427089/229065845-bbea5f3e-f1cf-4a6b-886d-ee87815c8b53.png">

* Test Set Graph
<img width="438" alt="4" src="https://user-images.githubusercontent.com/93427089/229066029-706aba25-b311-40dd-ad91-fb7a75c799c4.png">

* Values of MSE, MAE and RMSE
<img width="165" alt="error" src="https://user-images.githubusercontent.com/93427089/229066337-612aab91-6440-4da2-ac09-28d23e2eb589.png">

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
