# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start the program.
Step 2: Import the necessary python packages
Step 3: Read the dataset.
Step 4: Define X and Y array.
Step 5: Define a function for costFunction,cost and gradient.
Step 6: Define a function to plot the decision boundary and predict the Regression value 
Step 7: Stop the program.
## Program:
```python
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: KISHORE.S
RegisterNumber: 212224230130  
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("Placement_Data.csv")
df

df=df.drop("sl_no",axis=1)
df=df.drop("salary",axis=1)
df

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["gender"]=df["gender"].astype('category')
df["ssc_b"]=df["ssc_b"].astype('category')
df["hsc_b"]=df["hsc_b"].astype('category')
df["hsc_s"]=df["hsc_s"].astype('category')
df["degree_t"]=df["degree_t"].astype('category')
df["workex"]=df["workex"].astype('category')
df["specialisation"]=df["specialisation"].astype('category')
df["status"]=df["status"].astype('category')
df.dtypes

df["gender"]=df["gender"].cat.codes
df["ssc_b"]=df["ssc_b"].cat.codes
df["hsc_b"]=df["hsc_b"].cat.codes
df["hsc_s"]=df["hsc_s"].cat.codes
df["degree_t"]=df["degree_t"].cat.codes
df["workex"]=df["workex"].cat.codes
df["specialisation"]=df["specialisation"].cat.codes
df["status"]=df["status"].cat.codes
df

X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values
Y

theta = np.random.random(X.shape[1]) # intitialise the model parameter
y=Y
# define the sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))

# define the loss function
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*log(1-h))

#define the gradient descent algorithm
def gradient_descent(theta, X,y, alpha, num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-= alpha*gradient
    return theta

#train the model
theta = gradient_descent(theta,X,y,alpha = 0.01, num_iterations = 1000)
# Make predictions
def predict(theta, X):
    h= sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,x)
y_pred

# evaluate the model
accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy",accuracy)

print(y_pred)

print(Y)

xnew= np.array([[0,87,0,95,0,2,0,0,1,0,0,0]])
y_prednew=predict(theta,xnew)
y_prednew
```

## Output:
DATAFRAME:

![Screenshot 2025-04-12 221604](https://github.com/user-attachments/assets/02755dcc-941b-4d35-9fed-a17debddb46d)

CLEANED DATA:

![Screenshot 2025-04-12 221630](https://github.com/user-attachments/assets/099f73dc-4640-42cc-9146-5012b6977200)

ENCODED DATA:

![Screenshot 2025-04-12 221641](https://github.com/user-attachments/assets/3790c2af-6473-4b59-8532-86851df8d7b4)

CATEGORISED VALUE:

![Screenshot 2025-04-12 221651](https://github.com/user-attachments/assets/379b7699-50fa-4a44-b324-94acf43c6895)

ACCURACY:

![Screenshot 2025-04-12 221703](https://github.com/user-attachments/assets/4bdc2816-4817-45fd-8def-8a4a2f8b81b0)

PREDICTED Y VALUE:

![Screenshot 2025-04-12 221717](https://github.com/user-attachments/assets/83946c18-486c-4244-b88e-a2dcf152b888)

Y VALUE:

![Screenshot 2025-04-12 221725](https://github.com/user-attachments/assets/2f4b6b08-7075-4a63-97d2-62129b78f766)

PREDICT THE NEW Y VALUE:

![Screenshot 2025-04-12 221731](https://github.com/user-attachments/assets/14b860db-9d8f-4ea8-9947-e8e82b556b83)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

