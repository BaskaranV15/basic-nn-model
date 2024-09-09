# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

This code loads data from Google Sheets into a Pandas DataFrame, preprocesses it by scaling the input features, and splits it into training and testing sets. It defines a simple neural network with TensorFlow/Keras, trains the model on the training data for 500 epochs, and then evaluates its performance on the test set. Finally, the code uses the trained model to make a prediction on a new scaled input value.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS
![image](https://github.com/user-attachments/assets/73ea739f-f6fc-4c3b-a83a-299cefa936fb)

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:V.BASKARAN
### Register Number:212222230020
```python


from tensorflow import keras
from keras import models

from keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd


auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)


worksheet = gc.open('DS').sheet1


rows = worksheet.get_all_values()


df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'INPUT':'float'})
df = df.astype({'OUTPUT':'float'})
df.head()

df

X = df.iloc[: , : -1].values
y = df.iloc[: , -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

scaler = MinMaxScaler()

scaler.fit(X_train.reshape(-1,1))

X_train1 = scaler.transform(X_train.reshape(-1,1))

n = models.Sequential([
    Dense(units=3,activation='relu',input_shape=[1]),
    Dense(units=3,activation='relu'),
    Dense(units=1)
])

n.summary()

n.compile(optimizer = 'rmsprop' , loss = 'mse')

n.fit(X_train1 , y_train , epochs = 500)

loss = pd.DataFrame(n.history.history)

loss.plot()

X_test1 = scaler.fit_transform(X_test)

n.evaluate(X_test1 , y_test)

i = [[30]]

i = scaler.fit_transform(i)

n.predict(i)



```
## Dataset Information

Include screenshot of the dataset


![image](https://github.com/user-attachments/assets/183d5529-ba95-4a7b-bf18-ae21c437de15)

## OUTPUT
### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/f3358dea-2902-485c-8946-19aef7b6b7de)


### Test Data Root Mean Squared Error

![image](https://github.com/user-attachments/assets/87ed951d-caef-459c-857c-e7de340563c3)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/bc3b633c-4285-4f86-812f-33af8f52a7e3)


## RESULT
Thus, the linear regressin network is built and implemented to predict the given input .
