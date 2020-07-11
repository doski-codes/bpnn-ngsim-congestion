# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
ngsim_us101 = pd.read_csv("NGSIM_US101.csv")

X = ngsim_us101.iloc[:, 0:18].values
y = ngsim_us101.iloc[:, 18].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 18))

# Adding the second hidden layer
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 200)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Part 4 - Graphs

# Plotting time graphs
from datetime import datetime
time_data = ngsim_us101.loc[:, 'Global_Time'].values

real_date = []
times = []

for date in time_data:
    real_date.append(datetime.fromtimestamp(date/1000).strftime("%A, %B %d, %Y %I:%M:%S"))
    times.append(datetime.fromtimestamp(date/1000).strftime("%I:%M:%S"))

real_date.sort()
real_date_data = pd.DataFrame(np.array(real_date))
real_time_data = pd.DataFrame(np.array(times))
