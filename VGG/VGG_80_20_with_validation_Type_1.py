import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from pyts.image import GramianAngularField
from skimage.transform import resize
from scipy.spatial.distance import cdist
import tensorflow as tf

# Set TensorFlow to use float32 globally
tf.keras.backend.set_floatx('float32')

# Load and merge data
train_data = pd.read_csv('Metadata/Train_data.csv')
test_data = pd.read_csv('Metadata/Test_data.csv')
merged_data = pd.concat([train_data, test_data], ignore_index=True)

# Function to extract patterns using 'Upstream Voltage Min(kV)'
def extract_patterns(data, disturbance_column, voltage_column=['Upstream Voltage Min(kV)'], window=5):
    patterns, labels = [], []
    for i, row in data.iterrows():
        if row[disturbance_column] != 0:  # Exclude non-disturbance events
            start_index = max(0, i - window)
            end_index = min(len(data) - 1, i + window)
            pattern = data.iloc[start_index:end_index + 1][voltage_column].values.flatten()
            if len(pattern) == len(voltage_column) * (2 * window + 1):
                patterns.append(pattern)
                labels.append(row[disturbance_column])
    return np.array(patterns), np.array(labels)

# Extract patterns and labels
patterns, labels = extract_patterns(merged_data, 'Disturbance Type')

# Normalize the patterns
scaler = MinMaxScaler(feature_range=(-1, 1))
normalized_patterns = scaler.fit_transform(np.vstack(patterns)).reshape(len(patterns), -1)

# Determine image size and transform patterns into GAF images
image_size = min(32, normalized_patterns.shape[1])
gaf = GramianAngularField(image_size=image_size, method='summation')
gaf_images = gaf.fit_transform(normalized_patterns)

# Resize GAF images for VGG16 compatibility
resized_images = np.array([resize(img, (32, 32), mode='reflect', anti_aliasing=True) for img in gaf_images])
X_images = np.expand_dims(resized_images, axis=-1).astype('float32')

# Modify labels for binary classification
binary_labels = np.where(np.isin(labels, [1]), 1, 0)

# Split data into train (80%) and test (20%) subsets
X_train, X_test, y_train, y_test = train_test_split(
    X_images, binary_labels, test_size=0.2, random_state=42, stratify=binary_labels
)

# Further split train into training (80%) and validation (20%)
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Convert labels to categorical
y_train_binary = to_categorical(y_train_final, num_classes=2).astype('float32')
y_val_binary = to_categorical(y_val, num_classes=2).astype('float32')
y_test_binary = to_categorical(y_test, num_classes=2).astype('float32')

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(binary_labels), y=binary_labels)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Define the VGG16 model
def vgg_model(input_shape):
    base_model = VGG16(include_top=False, weights=None, input_shape=input_shape)
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu', kernel_regularizer='l2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(2, activation='softmax')(x)  # Binary classification output
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Initialize the VGG16 model
vgg_advanced = vgg_model((32, 32, 1))

# Train the VGG16 model
history = vgg_advanced.fit(
    X_train_final, y_train_binary,
    validation_data=(X_val, y_val_binary),
    epochs=50,
    batch_size=32,
    class_weight=class_weights_dict,
    verbose=1
)

# Evaluate the VGG16 model
train_loss, train_accuracy = vgg_advanced.evaluate(X_train_final, y_train_binary, verbose=1)
val_loss, val_accuracy = vgg_advanced.evaluate(X_val, y_val_binary, verbose=1)
test_loss, test_accuracy = vgg_advanced.evaluate(X_test, y_test_binary, verbose=1)

# Print metrics
print(f"Training Accuracy: {train_accuracy:.4f}, Training Loss: {train_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")


# Classification Report
y_pred_probs_vgg = vgg_advanced.predict(X_test)
y_pred_vgg = np.argmax(y_pred_probs_vgg, axis=1)
print(classification_report(np.argmax(y_test_binary, axis=1), y_pred_vgg, target_names=['No Event', 'Event']))

# Confusion Matrix
conf_matrix = confusion_matrix(np.argmax(y_test_binary, axis=1), y_pred_vgg)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Event', 'Event'], yticklabels=['No Event', 'Event'])
plt.title("VGG16 Binary Classification Confusion Matrix on Test Data")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# Plot Train Accuracy vs Train Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue', marker='o', linewidth=2)
plt.plot(history.history['loss'], label='Train Loss', color='red', linestyle='--', marker='x', linewidth=2)
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
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='green', marker='o', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linestyle='--', marker='x', linewidth=2)
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
plt.scatter(test_loss, test_accuracy, color='blue', s=100, label=f'Test Accuracy vs Test Loss')
plt.title('Test Accuracy vs Test Loss', fontsize=14, fontweight='bold')
plt.xlabel('Test Loss', fontsize=12, fontweight='bold')
plt.ylabel('Test Accuracy', fontsize=12, fontweight='bold')
plt.xticks(fontsize=10, fontweight='bold')
plt.yticks(fontsize=10, fontweight='bold')
plt.legend(loc='best', fontsize=10, frameon=True, shadow=True, borderpad=1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



# Predict and generate classification report
y_pred_probs_vgg = vgg_advanced.predict(X_test)
y_pred_vgg = np.argmax(y_pred_probs_vgg, axis=1)
print(classification_report(np.argmax(y_test_binary, axis=1), y_pred_vgg, target_names=['No Event', 'Event']))

# Confusion Matrix
conf_matrix_vgg = confusion_matrix(np.argmax(y_test_binary, axis=1), y_pred_vgg)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_vgg, annot=True, fmt='d', cmap='Blues', xticklabels=['No Event', 'Event'], yticklabels=['No Event', 'Event'])
plt.title("VGG16 Binary Classification Confusion Matrix on Test Data")
plt.xlabel("Predicted", fontweight='bold')
plt.ylabel("Actual", fontweight='bold')
plt.show()

# Find closest training patterns for matched test events
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Assuming predictions for VGG16 are available as `y_pred_vgg`
model_predictions = {
    'VGG16': y_pred_vgg  # Using VGG16 predictions
}

# Split normalized patterns into training and testing subsets
normalized_train_patterns = [normalized_patterns[i] for i in range(len(X_train))]
normalized_test_patterns = [normalized_patterns[i] for i in range(len(X_train), len(normalized_patterns))]

# Convert to numpy arrays for consistency
normalized_train_patterns = np.array(normalized_train_patterns)
normalized_test_patterns = np.array(normalized_test_patterns)

# Function to find the closest training pattern using Euclidean distance
def find_closest_train_pattern(test_pattern, train_patterns):
    distances = cdist([test_pattern], train_patterns, metric='euclidean')
    closest_index = np.argmin(distances)
    return closest_index

# Step 1: Identify matched events for the VGG16 model
binary_test_labels = np.argmax(y_test_binary, axis=1)  # Convert one-hot encoded labels to binary labels
vgg_matched_indices = [
    i for i in range(len(binary_test_labels))
    if binary_test_labels[i] == 1 and y_pred_vgg[i] == 1  # Both ground truth and prediction are "Event"
]

# Check if there are at least four matched indices
if len(vgg_matched_indices) < 4:
    print("Less than four matched events found for the VGG16 model.")
else:
    # Select the first four matched indices
    selected_indices = vgg_matched_indices[:4]

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
        ax.set_title(f'Test vs Closest Train Pattern (Event {index + 1})', fontweight='bold', fontsize=12)
        ax.set_xlabel('Index (Relative Time)', fontweight='bold', fontsize=10)  # Use relative index for x-axis
        ax.set_ylabel('Normalized Upstream Voltage (Index)', fontweight='bold', fontsize=10)  # Use normalized voltage values for y-axis
        ax.legend(loc='upper right', fontsize=9, framealpha=0.7)
        ax.grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()
