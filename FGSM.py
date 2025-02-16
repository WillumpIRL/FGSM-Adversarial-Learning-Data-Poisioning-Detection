import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Load datasets
train_data = pd.read_csv("train.csv")  
test_data = pd.read_csv("test.csv")  

# Preprocessing: Separate features and labels
X_train = train_data.drop(columns=["index", "poisoned"]).values
y_train = train_data["poisoned"].values

X_test = test_data.drop(columns=["index"]).values  # Assuming no label in test
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a simple model
def build_model(input_shape):
    model = Sequential([
        Dense(128, input_dim=input_shape, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model(X_train.shape[1])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# FGSM attack
def generate_adversarial_example(model, X, epsilon=0.01):
    # We need the gradient of the loss w.r.t the input
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        predictions = model(X_tensor)
        loss = tf.keras.losses.binary_crossentropy(y_train, predictions)
    gradients = tape.gradient(loss, X_tensor)
    
    # Sign of the gradients to perturb the input
    perturbations = epsilon * tf.sign(gradients)
    adversarial_example = X_tensor + perturbations
    return tf.clip_by_value(adversarial_example, 0, 1)

# Generate adversarial examples for the training set
X_adversarial = generate_adversarial_example(model, X_train)

# Combine adversarial and clean examples (you can experiment with proportions)
X_combined = np.concatenate((X_train, X_adversarial), axis=0)
y_combined = np.concatenate((y_train, y_train), axis=0)  # Keeping same labels

# Train on combined dataset
model.fit(X_combined, y_combined, epochs=10, batch_size=32, verbose=1)

# Predict poisoned samples in training data
y_pred_train = model.predict(X_train)
poisoned_pred_train = (y_pred_train > 0.5).astype(int)

# Evaluate the model using AUC-ROC
auc_roc = roc_auc_score(y_train, poisoned_pred_train)
print(f"AUC-ROC Score on Training Data: {auc_roc}")

# Output predictions for submission
predictions = pd.DataFrame({
    "index": train_data["index"],
    "poisoned": poisoned_pred_train.flatten()
})
predictions.to_csv("submission.csv", index=False)
