# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values.

## Program:
```python
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: MERCY A
RegisterNumber:  212223110027
*/

import pandas as pd
data = pd.read_csv("Salary.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
print(data.head())
x = data[["Position","Level"]]
y = data["Salary"]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(X_train,Y_train)
y_pred = dt.predict(X_test)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y_test,y_pred)
print("Mean Squared Error: ",mse)
r2 = metrics.r2_score(Y_test,y_pred)
print(r2)
dt.predict([[5,6]])

```

## Output:

![Screenshot 2024-10-16 114426](https://github.com/user-attachments/assets/afba271a-9984-415b-870c-4f0725347f60)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
