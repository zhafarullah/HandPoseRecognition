import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

csv_filename = "hand_pose_dataset.csv"
data = pd.read_csv(csv_filename)

labels = data['label']
coordinates = data.iloc[:, 1:].applymap(lambda s: np.array(list(map(float, s.split(',')))) if isinstance(s, str) else s)

coordinates_flat = np.array([np.concatenate(coords) for coords in coordinates.values])

def augment_data(X, y):
    augmented_X = []
    augmented_y = []

    for i, landmarks in enumerate(X):
        landmarks_reshaped = landmarks.reshape(-1, 3) 

        augmented_X.append(landmarks)
        augmented_y.append(y[i])

        mirrored = landmarks_reshaped.copy()
        mirrored[:, 0] = -mirrored[:, 0]
        augmented_X.append(mirrored.flatten())
        augmented_y.append(y[i])

        mirrored_flipped_z = mirrored.copy()
        mirrored_flipped_z[:, 2] = -mirrored_flipped_z[:, 2]
        augmented_X.append(mirrored_flipped_z.flatten())
        augmented_y.append(y[i])

    return np.array(augmented_X), np.array(augmented_y)

coordinates_augmented, labels_augmented = augment_data(coordinates_flat, labels)

label_map = {label: idx for idx, label in enumerate(['fist', 'hifive', 'unknown'])}
y_encoded = np.array([label_map[label] if label in label_map else label_map['unknown'] for label in labels_augmented])

X_train, X_test, y_train, y_test = train_test_split(coordinates_augmented, y_encoded, test_size=0.2, random_state=42)

y_train = to_categorical(y_train, num_classes=len(label_map))
y_test = to_categorical(y_test, num_classes=len(label_map))

model = Sequential([
    Dense(128, activation='relu', input_shape=(coordinates_flat.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(label_map), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    validation_split=0.2,
                    epochs=50,
                    batch_size=16)


loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model Loss: {loss}")
print(f"Model Accuracy: {accuracy}")

model.save("hand_pose_dnn_model.h5")
print("Model saved as 'hand_pose_dnn_model.h5'")

print("\nTesting the model with a sample input:")
sample_idx = np.random.randint(0, X_test.shape[0])
sample_input = X_test[sample_idx].reshape(1, -1)
true_label = np.argmax(y_test[sample_idx])

predictions = model.predict(sample_input)
predicted_label = np.argmax(predictions)
prediction_confidence = np.max(predictions) * 100

reverse_label_map = {v: k for k, v in label_map.items()}
print(f"True Label: {reverse_label_map[true_label]}")
print(f"Predicted Label: {reverse_label_map[predicted_label]} ({prediction_confidence:.2f}% confidence)")

