import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def showModelWeights():
    # Load your model
    model = tf.keras.models.load_model('PHEmodel.h5')
    weights = model.get_weights()

    # Create a NumPy array to store the weights of all layers
    weights_array = np.empty((0))
    for i in range(len(weights)):
        weights_array = np.append(weights_array, weights[i])

    # Plot the weights
    plt.plot(weights_array)
    plt.xlabel("Weight Index")
    plt.ylabel("Weight Value")
    plt.title("Model Weights")
    plt.show()

