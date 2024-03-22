# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset from a CSV file
data = pd.read_csv("data.csv",
                   usecols=lambda column: column not in ["id", "Unnamed: 32"])
data.head()
# check for missing values
data.isnull().sum()
# Calculate and Print the correlation matrix
cols_to_include = [col for col in data.columns if col != 'diagnosis']
data_subset = data[cols_to_include]
corr_matrix = data_subset.corr()
print(corr_matrix)
# Separate the features and target variable
y = data['diagnosis'].map({'M': 1, 'B': 0}).values
X = data.drop(columns=["diagnosis", "perimeter_mean", "perimeter_worst"]).values
# We dropped the perimeter_mean and perimeter_worst columns because they are highly correlated

print('The number of features ( cell nucleus characteristics ) is', X.shape[1])
print('The number of target variable ( patients ) is', X.shape[0])
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split the testing into testing and validation sets
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.75, random_state=42)
# Standardize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Define the network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# Compile the model with gradient descent optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)  # Adjust the learning rate as needed
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
history = model.fit(X_train, y_train, epochs=800, batch_size=32, validation_data=(X_val, y_val))

# Get the training and validation loss and accuracy values from the history object
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(train_loss) + 1)

# Plot the training loss curve
plt.figure()
plt.plot(epochs, train_loss, 'bo', label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the validation loss curve
plt.figure()
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the training accuracy curve
plt.figure()
plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot the validation accuracy curve
plt.figure()
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# Use the trained model to make predictions on the test data
predictions = model.predict(X_test)
predicted_labels = np.round(predictions).flatten().astype(int)

# Convert the predicted labels and actual labels to 'M' and 'B' classes
predicted_classes = np.where(predicted_labels == 1, 'M', 'B')
actual_classes = np.where(y_test == 1, 'M', 'B')

# Print the predicted class and actual class for each row
for i in range(len(X_test)):
    print(f"Predicted: {predicted_classes[i]}  Actual: {actual_classes[i]}")
# Calculate evaluation metrics
accuracy = accuracy_score(actual_classes, predicted_classes)
precision = precision_score(actual_classes, predicted_classes, pos_label='M')
recall = recall_score(actual_classes, predicted_classes, pos_label='M')
f1 = f1_score(actual_classes, predicted_classes, pos_label='M')

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
values = [accuracy, precision, recall, f1]
colors = ['blue', 'green', 'orange', 'red']
plt.bar(metrics, values, color=colors)
plt.ylim([0.0, 1.0])  # set the y-axis limits to be between 0 and 1
plt.title('Evaluation Metrics')
plt.show()
# Use the trained model to make predictions on the test data
predictions = model.predict(X_test)
predicted_labels = np.round(predictions).flatten().astype(int)

# Convert the predicted labels and actual labels to 'M' and 'B' classes
predicted_classes = np.where(predicted_labels == 1, 'M', 'B')
actual_classes = np.where(y_test == 1, 'M', 'B')

# Print the predicted class and actual class for each row
for i in range(len(X_test)):
    print(f"Predicted: {predicted_classes[i]}  Actual: {actual_classes[i]}")
# Calculate evaluation metrics
accuracy = accuracy_score(actual_classes, predicted_classes)
precision = precision_score(actual_classes, predicted_classes, pos_label='M')
recall = recall_score(actual_classes, predicted_classes, pos_label='M')
f1 = f1_score(actual_classes, predicted_classes, pos_label='M')

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
values = [accuracy, precision, recall, f1]
colors = ['blue', 'green', 'orange', 'red']
plt.ylim([0.0, 1.0])  # set the y-axis limits to be between 0 and 1
plt.title('Evaluation Metrics')
plt.bar(metrics, values, color=colors)
# Make predictions on new data
# new_data = scaler.transform(new_data)  # Assuming 'new_data' is a new set of features
# predictions = model.predict(new_data)
