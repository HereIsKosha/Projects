# Artificial Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('TelcoCustomerChurn.csv')
X = dataset.iloc[:, [1,2,5,6,7,8,13,14,15,18,19]].values
y = dataset.iloc[:, 20].values

#dataset = dataset.drop(columns = ['customerID'])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Below lines should be commented to avoid make_column_transformer - Future error at line 41
labelencoder_X_1 = LabelEncoder()
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])
labelencoder_X_2 = LabelEncoder()
X[:, 3] = labelencoder_X_2.fit_transform(X[:, 3])
labelencoder_X_3 = LabelEncoder()
X[:, 4] = labelencoder_X_3.fit_transform(X[:, 4])
labelencoder_X_4 = LabelEncoder()
X[:, 5] = labelencoder_X_4.fit_transform(X[:, 5])
labelencoder_X_5 = LabelEncoder()
X[:, 6] = labelencoder_X_5.fit_transform(X[:, 6])
labelencoder_X_6 = LabelEncoder()
X[:, 7] = labelencoder_X_6.fit_transform(X[:, 7])
labelencoder_X_7 = LabelEncoder()
X[:, 8] = labelencoder_X_7.fit_transform(X[:, 8])
labelencoder_X_8 = LabelEncoder()
X[:, 1] = labelencoder_X_8.fit_transform(X[:, 1])
labelencoder_X_9 = LabelEncoder()
X[:, 2] = labelencoder_X_9.fit_transform(X[:, 2])
labelencoder_X_10 = LabelEncoder()
X[:, 9] = labelencoder_X_10.fit_transform(X[:, 9])
labelencoder_X_11 = LabelEncoder()
X[:, 10] = labelencoder_X_11.fit_transform(X[:, 10])

labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
columnTransform = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [3])],remainder='passthrough')
X = np.array(columnTransform.fit_transform(X), dtype=np.float)
X = X[:, 1:]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# 0.78 % accuracy : Parameter Tunning can help improve accuracy   