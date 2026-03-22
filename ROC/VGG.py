import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import VGG16
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from pyts.image import GramianAngularField
from skimage.transform import resize
import tensorflow as tf

tf.keras.backend.set_floatx('float32')

# Load and merge data
train_data = pd.read_csv('Data/Train_Data.csv')
test_data = pd.read_csv('Data/Test_Data.csv')
merged_data = pd.concat([train_data, test_data], ignore_index=True)

# Function to extract patterns
def extract_patterns(data, disturbance_column, thdv_column=['Upstream Voltage Min(kV)'], window=5):  # Change the feature to [THD-V Max(%)] for the Label [4] Waveshape change disturbance 
    patterns, labels = [], []
    for i, row in data.iterrows():
        if row[disturbance_column] != 0:
            start_index = max(0, i - window)
            end_index = min(len(data) - 1, i + window)
            pattern = data.iloc[start_index:end_index + 1][thdv_column].values.flatten()
            if len(pattern) == len(thdv_column) * (2 * window + 1):
                patterns.append(pattern)
                labels.append(row[disturbance_column])
    return np.array(patterns), np.array(labels)

patterns, labels = extract_patterns(merged_data, 'Disturbance Type')
scaler = MinMaxScaler(feature_range=(-1, 1))
normalized_patterns = scaler.fit_transform(np.vstack(patterns)).reshape(len(patterns), -1)
image_size = min(32, normalized_patterns.shape[1])
gaf = GramianAngularField(image_size=image_size, method='summation')
gaf_images = gaf.fit_transform(normalized_patterns)
resized_images = np.array([resize(img, (32, 32), mode='reflect', anti_aliasing=True) for img in gaf_images])
X_images = np.expand_dims(resized_images, axis=-1).astype('float32')
binary_labels = np.where(np.isin(labels, [1]), 1, 0)  #Change the Label [1] to [4] for different disturbance type Waveshape change

# K-Fold Cross Validation
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
plt.figure()
for i, (train_index, val_index) in enumerate(kf.split(X_images, binary_labels)):
    X_train, X_val = X_images[train_index], X_images[val_index]
    y_train, y_val = binary_labels[train_index], binary_labels[val_index]
    y_train = to_categorical(y_train, num_classes=2).astype('float32')
    y_val = to_categorical(y_val, num_classes=2).astype('float32')
    
    def vgg16_model(input_shape):
        base_model = VGG16(include_top=False, weights=None, input_shape=input_shape)
        x = Flatten()(base_model.output)
        x = Dense(256, activation='relu', kernel_regularizer='l2')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        output = Dense(2, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    vgg16_advanced = vgg16_model((32, 32, 1))
    class_weights = compute_class_weight('balanced', classes=np.unique(binary_labels), y=binary_labels)
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
    vgg16_advanced.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), class_weight=class_weights_dict)
    
    y_pred_probs = vgg16_advanced.predict(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(np.argmax(y_val, axis=1), y_pred_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Fold {i+1} AUC = {roc_auc:.2f}')
    
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Receiver Operating Characteristic (ROC) Curve')

plt.legend(loc='lower right')
ply.tight_layout()
plt.savefig("C:/Users/nirma/Documents/paper_3_code/roc_auc_curve_Type_1_VGG16.png", dpi=300, bbox_inches='tight')

plt.show()
    
y_pred = np.argmax(vgg16_advanced.predict(X_val), axis=1)
y_true = np.argmax(y_val, axis=1)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['No Event', 'Event']))