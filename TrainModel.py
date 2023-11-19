import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers
import keras.models

import matplotlib.pyplot as plt

CERN_Data = pd.read_csv('CERN_DataSet.csv')

TOnumpy = CERN_Data.to_numpy()

DataForm1 = TOnumpy.reshape((TOnumpy.shape[0], 11))
DataForm2 = DataForm1[:, 2:]

DataTrain = DataForm2[:40000, :]
DataTest = DataForm2[40001:100000, :]

# Data normalization using Min-Max scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
DataTrain = scaler.fit_transform(DataTrain)
DataTest = scaler.transform(DataTest)

X_train = DataTrain.reshape(((DataTrain.shape[0], 9)))
Y_train = X_train[:, 0].reshape(-1, 1)

p = 16
# Create the model
modele = Sequential()
modele.add(Dense(p, input_dim=9, activation='relu'))

# Hidden layer with ReLU activation and dropout regularization
modele.add(Dense(p, activation='relu'))
modele.add(Dropout(0.2))
modele.add(Dense(p, activation='relu'))
modele.add(Dropout(0.2))

# Output layer with linear activation
modele.add(Dense(1, activation='linear'))

modele.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['accuracy'])

# Train the model
try:
    modele.fit(X_train, Y_train, batch_size=64, epochs=30)
except Exception as e:
    print(f"Error during training: {e}")
    exit(1)

def lossFunctionPlot():
    loss_per_epoch = modele.history.history['loss']

    # Convert the y_pred variable to a NumPy array
    y_pred = modele.predict(X_test)
    y_pred = np.array(y_pred)

    # Calculate the loss on the test set
    loss = modele.evaluate(X_test, Y_test) # Removed the [0]

    # Plot the loss
    plt.plot(loss_per_epoch)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss function of the model over the number of epochs')
    plt.show()

def PredictVSactual():
    y_pred = modele.predict(X_test)
    y_test_array = np.array(Y_test)
    plt.scatter(y_pred, y_test_array)
    plt.xlabel('Predicted values')
    plt.ylabel('Actual values')
    plt.title('Distribution of the predicted values versus the actual values')
    plt.show()

def accuracyPlot():
    accuracy_per_epoch = modele.history.history['accuracy']
    plt.plot(accuracy_per_epoch)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of the model over the number of epochs')
    plt.show()

# Evaluate the model on the test data
X_test = DataTest.reshape((DataTest.shape[0], 9))
Y_test = X_test[:, 0].reshape(-1, 1)

# Save the model
keras.models.save_model(modele, 'PHEmodel.h5')