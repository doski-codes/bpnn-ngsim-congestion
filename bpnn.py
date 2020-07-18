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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)

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

accuracy = (cm[0,0] + cm[1,1])/(cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1])
recall = cm[0,0]/(cm[0,0] + cm[1,0])
precision = cm[0,0]/(cm[0,0] + cm[0,1])

print("The accuracy of the BPNN model on this dataset is, %.5f" %accuracy)
print("The recall of the BPNN model on this dataset is, %.5f" %recall)
print("The precision of the BPNN model on this dataset is, %.5f" %precision)

#Part 4 - Graphs

# Plotting time graphs
# Importing the datetime library
from datetime import datetime
from pytz import timezone

# Using time data to plot the number of cars on the road at various times
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
    
begin.append(milli[len(milli) - 1])
indexes.append(len(milli))

X = []
Y = []
var = 0


for num in range(1, len(begin)):
    X.append(datetime.fromtimestamp(begin[num]/1000, tz=timezone('US/Pacific')).strftime("%I:%M"))
    
for num in range(len(indexes) - 1):
    var = indexes[num + 1] - indexes[num]
    Y.append(var)

plt.figure(figsize=[20,10])
plt.title("Volume of cars on the US101 Highway on 15-June-2005", fontsize=24)
plt.plot(X, Y)
plt.yticks(np.arange(3000, 90000, step=5000))
plt.xlabel("Time of Vehicle's arrival (in PDT)", fontsize=14)
plt.ylabel("Number of vehicles", labelpad=10, fontsize=14)
plt.show()

# Using congestion data to plot time/when congestion occurs
congestion = ngsim_us101['Congestion'] == 1
congestion = ngsim_us101[congestion]
congestion_time = congestion.loc[:, 'Global_Time'].values

milli = []

for date in congestion_time:
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
    
begin.append(milli[len(milli) - 1])
indexes.append(len(milli))

cong_X = []
cong_Y = []

for num in range(1, len(begin)):
    cong_X.append(datetime.fromtimestamp(begin[num]/1000, tz=timezone('US/Pacific')).strftime("%I:%M"))

for num in range(len(indexes) - 1):
    var = indexes[num + 1] - indexes[num]
    cong_Y.append(var)


plt.figure(figsize=[20,10])
plt.title("Traffic Congestion on the US101 Highway on 15-June-2005", fontsize=24)
plt.plot(cong_X, cong_Y)
plt.yticks(np.arange(3000, 90000, step=5000))
plt.xlabel("Time (in PDT)", fontsize=14)
plt.ylabel("Number of Vehicles involved in Congestion", labelpad=10, fontsize=14)
plt.show()


LR = pd.read_csv("LR_Prediction_TRAIN.csv")
ngsim_us101.insert(18, "Congestion", np.array(congestion), True)

# Bar chart based on vehicle types
v1 = ngsim_us101['v_Class'] == 1
v1 = ngsim_us101[v1]
v2 = ngsim_us101['v_Class'] == 2
v2 = ngsim_us101[v2]
v3 = ngsim_us101['v_Class'] == 3
v3 = ngsim_us101[v3]
total_vehicles = (len(v1) + len(v2) + len(v3))

vehicle_type = [(len(v1)/total_vehicles)*100, (len(v2)/total_vehicles)*100, (len(v3)/total_vehicles)*100]
vehicle_class = ['Motorcycles', 'Auto', 'Trucks']

colours = ['red', 'gold', 'green']
explode = (0.5, 0, 0)

plt.pie(vehicle_type, explode=explode, colors=colours, labels=vehicle_class, autopct='%1.1f%%', startangle=10,
        pctdistance=0.8, shadow=True, radius=2)


