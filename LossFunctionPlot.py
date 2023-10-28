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
model.compile(optimizer='adam', loss='mae', loss_weights=[1.0, 0.01])

# Train the model
model.fit(X_train, y_train, epochs=10)

def lossFunctionPlot():
    loss_per_epoch = model.history.history['loss']

    # Convert the y_pred variable to a NumPy array
    y_pred = model.predict(X_test)
    y_pred = np.array(y_pred)

    # Calculate the loss on the test set
    loss = model.evaluate(X_test, y_test) # Removed the [0]

    # Plot the loss
    plt.plot(loss_per_epoch)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss function of the model over the number of epochs')
    plt.show()
