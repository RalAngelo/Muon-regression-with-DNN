import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv('CERN_DataSet.csv')

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df[['pt', 'eta', 'phi', 'Q', 'chiSq', 'dxy', 'iso', 'MET', 'phiMET']], df['pt'], test_size=0.25, random_state=42)

# Create a support vector machine model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = np.mean(abs(y_pred - y_test) / y_test) * 100

# Print the accuracy of the model
print('Accuracy:', accuracy)
