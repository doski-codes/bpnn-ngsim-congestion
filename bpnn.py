# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
ngsim_us101 = pd.read_csv("NGSIM_US101.csv")
ngsim_us101 = ngsim_us101.drop('Unnamed: 0', axis=1)

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

milli = []

for date in time_data:
    milli.append(date)

milli.sort()

beginning = milli[0]
begin = [milli[0],]
end = 0
indexes = [0]
for num in range(len(milli)):
    while num < (len(milli) - 1):
        if ((milli[num + 1] - beginning) < 300000):
            end = milli[num]
        else:
            begin.append(beginning)
            beginning = end
            indexes.append(num)
        break

X = []
Y = []
var = 0

for num in range(1, len(begin)):
    X.append(datetime.fromtimestamp(begin[num]/1000).strftime("%I:%M"))
    
for num in range(len(indexes) - 1):
    var = indexes[num + 1] - indexes[num]
    Y.append(var)

plt.figure(figsize=[20,10])
plt.title("Volume of cars on the US101 Highway on 15-June-2005", fontsize=24)
plt.plot(X, Y)
plt.yticks(np.arange(30000, 90000, step=5000))
plt.xlabel("Time of Vehicle's arrival", fontsize=14)
plt.ylabel("Number of vehilces", labelpad=10, fontsize=14)
plt.show()