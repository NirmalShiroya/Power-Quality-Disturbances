# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from pyts.image import GramianAngularField
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Set TensorFlow to use float64 globally
tf.keras.backend.set_floatx('float64')

# Load and merge data
train_data = pd.read_csv('Data/Train_data.csv')
test_data = pd.read_csv('Data/Test_data.csv')merged_data = pd.concat([train_data, test_data], ignore_index=True)

# Function to extract patterns using only THD-V Max(%)
def extract_patterns(data, disturbance_column, thdv_column=['THD-V Max(%)'], window=5):
    patterns = []
    labels = []
    time_points = []
    for i, row in data.iterrows():
        if row[disturbance_column] != 0:  # Exclude non-disturbance events
            start_index = max(0, i - window)
            end_index = min(len(data) - 1, i + window)
            pattern = data.iloc[start_index:end_index + 1][thdv_column].values.flatten()
            if len(pattern) == len(thdv_column) * (2 * window + 1):  # Ensure consistent length
                patterns.append(pattern)
                labels.append(row[disturbance_column])
                time_points.append(data.iloc[i]['Start Date and Time'])
    return np.array(patterns), np.array(labels), time_points

# Extract patterns from the merged data
patterns, labels, time_points = extract_patterns(merged_data, 'Disturbance Type')

# Normalize the patterns
scaler = MinMaxScaler(feature_range=(-1, 1))
normalized_patterns = [scaler.fit_transform(p.reshape(-1, 1)).flatten() for p in patterns]

# Convert to images using Gramian Angular Field
image_size = min(50, max(len(p) for p in normalized_patterns))
gaf = GramianAngularField(image_size=image_size, method='summation')
image_patterns = gaf.fit_transform(normalized_patterns)

# Expand dimensions for CNN compatibility
X_images = np.expand_dims(image_patterns, axis=-1).astype('float64')

# Modify labels for binary classification (Event: Type 1 or No Event)
binary_labels = np.where(np.isin(labels, [4]), 1, 0)

# Split data into train (60%) and test (40%) subsets
X_train, X_test, y_train, y_test = train_test_split(
    X_images,
    binary_labels,
    test_size=0.4,  # Set test size to 40%
    random_state=42,
    stratify=binary_labels
)

# Further split train into training (80%) and validation (20%)
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,  # 20% of 60% (12% of total data) will be validation
    random_state=42,
    stratify=y_train
)

# Convert labels to categorical
y_train_final = to_categorical(y_train_final, num_classes=2).astype('float64')
y_val = to_categorical(y_val, num_classes=2).astype('float64')
y_test_binary = to_categorical(y_test, num_classes=2).astype('float64')

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(binary_labels), y=binary_labels)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Prepare TensorFlow datasets
batch_size = 32

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_final, y_train_final))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test_binary))
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Define a custom CNN model
def custom_cnn_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        Dense(2, activation='softmax')  # Binary classification output
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Initialize the custom CNN model
custom_cnn = custom_cnn_model((image_size, image_size, 1))

# Train the custom CNN model with validation data
custom_cnn_history = custom_cnn.fit(
    train_dataset,
    epochs=100,
    validation_data=val_dataset,
    class_weight=class_weights_dict
)

# Evaluate the model on validation data
val_loss, val_accuracy = custom_cnn.evaluate(val_dataset, verbose=1)
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Loss: {val_loss:.4f}")

# Evaluate the model on test data
test_loss, test_accuracy = custom_cnn.evaluate(test_dataset, verbose=1)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")


# Extract Training Accuracy and Loss from the training history
train_loss = custom_cnn_history.history['loss'][-1]  # Last epoch training loss
train_accuracy = custom_cnn_history.history['accuracy'][-1]  # Last epoch training accuracy

# Print Training, Validation, and Test Metrics
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Training Loss: {train_loss:.4f}")


# Predict using the custom CNN model
y_pred_probs_custom_cnn = custom_cnn.predict(X_test)
y_pred_custom_cnn = np.argmax(y_pred_probs_custom_cnn, axis=1)

# Classification Report
print(f"Custom CNN Classification Report:\n")
print(classification_report(
    np.argmax(y_test_binary, axis=1),  # Actual class labels
    y_pred_custom_cnn,                 # Predicted class labels
    target_names=['No Event', 'Event'] # Class names
))

# Confusion Matrix
binary_conf_matrix_custom = confusion_matrix(
    np.argmax(y_test_binary, axis=1),  # Actual class labels
    y_pred_custom_cnn                  # Predicted class labels
)

# Plot Confusion Matrix as Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(binary_conf_matrix_custom, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Event', 'Event'], yticklabels=['No Event', 'Event'])
plt.title("Custom CNN Binary Classification Confusion Matrix on Test Data")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# Define a style for plots

# Plot Train Accuracy vs Train Loss
plt.figure(figsize=(10, 6))
plt.plot(custom_cnn_history.history['accuracy'], label='Train Accuracy', color='blue', marker='o', linewidth=2)
plt.plot(custom_cnn_history.history['loss'], label='Train Loss', color='red', linestyle='--', marker='x', linewidth=2)
plt.title('Train Accuracy vs Train Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Value', fontsize=12, fontweight='bold')
plt.legend(loc='best', fontsize=10, frameon=True, shadow=True, borderpad=1)
plt.xticks(fontsize=10, fontweight='bold')
plt.yticks(fontsize=10, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Plot Validation Accuracy vs Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(custom_cnn_history.history['val_accuracy'], label='Validation Accuracy', color='green', marker='o', linewidth=2)
plt.plot(custom_cnn_history.history['val_loss'], label='Validation Loss', color='orange', linestyle='--', marker='x', linewidth=2)
plt.title('Validation Accuracy vs Validation Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Value', fontsize=12, fontweight='bold')
plt.legend(loc='best', fontsize=10, frameon=True, shadow=True, borderpad=1)
plt.xticks(fontsize=10, fontweight='bold')
plt.yticks(fontsize=10, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Plot Test Accuracy vs Test Loss
plt.figure(figsize=(10, 6))
plt.axhline(y=test_accuracy, color='blue', linestyle='-', linewidth=2, label=f'Test Accuracy: {test_accuracy:.4f}')
plt.axhline(y=test_loss, color='red', linestyle='--', linewidth=2, label=f'Test Loss: {test_loss:.4f}')
plt.title('Test Accuracy vs Test Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Value', fontsize=12, fontweight='bold')
plt.legend(loc='best', fontsize=10, frameon=True, shadow=True, borderpad=1)
plt.xticks(fontsize=10, fontweight='bold')
plt.yticks(fontsize=10, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()




import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Assuming y_pred_vgg, y_pred_lstm, y_pred_resnet, y_pred_cnn are predictions for VGG, LSTM, ResNet50, and CNN models
model_predictions = {
    
    'CNN': y_pred_custom_cnn  # Assuming this variable is defined as per your models
}

# Split normalized patterns into training and testing subsets
normalized_train_patterns = [normalized_patterns[i] for i in range(len(X_train))]
normalized_test_patterns = [normalized_patterns[i] for i in range(len(X_train), len(normalized_patterns))]

# Convert to numpy arrays for consistency
normalized_train_patterns = np.array(normalized_train_patterns)
normalized_test_patterns = np.array(normalized_test_patterns)

import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import numpy as np

# Function to find the closest training pattern using Euclidean distance
def find_closest_train_pattern(test_pattern, train_patterns):
    distances = cdist([test_pattern], train_patterns, metric='euclidean')
    closest_index = np.argmin(distances)
    return closest_index

# Step 1: Identify matched events for the CNN model
binary_test_labels = np.argmax(y_test_binary, axis=1)  # Convert one-hot encoded labels to binary labels
cnn_matched_indices = [
    i for i in range(len(binary_test_labels))
    if binary_test_labels[i] == 1 and y_pred_custom_cnn[i] == 1  # Both ground truth and prediction are "Event"
]

# Check if there are at least four matched indices
if len(cnn_matched_indices) < 4:
    print("Less than four matched events found for the CNN model.")
else:
    # Select the first four matched indices
    selected_indices = cnn_matched_indices[:4]

    # Step 2: Find closest training patterns for each selected matched test pattern
    closest_train_indices = {}
    for index in selected_indices:
        closest_train_index = find_closest_train_pattern(
            normalized_test_patterns[index],
            normalized_train_patterns
        )
        closest_train_indices[index] = closest_train_index
    
    # Step 3: Create subplots to plot each matched event with its closest training pattern
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()  # Flatten the axes array for easy indexing
    
    for ax, index in zip(axes, selected_indices):
        # Retrieve test pattern (voltage values)
        test_pattern = normalized_test_patterns[index]
        test_time_points = np.arange(len(test_pattern))  # Generate indices for x-axis (0, 1, 2, ...)
    
        # Retrieve closest training pattern (voltage values)
        closest_train_index = closest_train_indices[index]
        train_pattern = normalized_train_patterns[closest_train_index]
        train_time_points = np.arange(len(train_pattern))  # Generate indices for x-axis (0, 1, 2, ...)
    
        # Plot the test pattern
        ax.plot(
            test_time_points,
            test_pattern,
            label=f'Test Pattern (Event {index + 1})',
            linestyle='-',
            marker='o',
            color='black'
        )
    
        # Plot the closest training pattern
        ax.plot(
            train_time_points,
            train_pattern,
            label=f'Closest Train Pattern',
            linestyle='--',
            marker='x',
            color='blue'
        )
    
        # Set titles and labels
        ax.set_title(f'Test vs Closest Train Pattern (Event {index + 1})')
        ax.set_xlabel('Index (Relative Time)')  # Use relative index for x-axis
        ax.set_ylabel('Normalized THD-V (Index)')  # Use voltage values for y-axis
        ax.legend(loc='upper right', fontsize='small', framealpha=0.7)
        ax.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

