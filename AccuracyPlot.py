import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt


df = pd.read_csv('CERN_DataSet.csv')
X_train, X_test, y_train, y_test = train_test_split(df[['pt', 'eta', 'phi', 'Q', 'chiSq', 'dxy', 'iso', 'MET', 'phiMET']], df['pt'], test_size=0.25, random_state=42)

# Create a DNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mae', loss_weights=[1.0, 0.01], metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10)

def accuracyPlot():
    accuracy_per_epoch = model.history.history['accuracy']
    plt.plot(accuracy_per_epoch)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of the model over the number of epochs')
    plt.show()
