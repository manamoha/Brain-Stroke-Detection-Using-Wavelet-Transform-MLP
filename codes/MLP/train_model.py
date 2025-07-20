import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import pickle

# Define the MLP model using Sequential API
def create_mlp_model(input_size=128, output_size=3):
    model = models.Sequential([
        layers.Input(shape=(input_size,)),  # Explicit Input layer
        layers.Dense(128, activation='relu', kernel_initializer='he_normal'),  # Hidden layer
        layers.BatchNormalization(),  # Add BatchNormalization
        layers.Dropout(0.3),  # Add Dropout (30%)
        layers.Dense(64, activation='relu', kernel_initializer='he_normal'),  # Hidden layer
        layers.BatchNormalization(),  # Add BatchNormalization
        layers.Dropout(0.1),  # Add Dropout (10%)
        layers.Dense(output_size, activation='softmax')  # Output layer for 3 classes
    ])
    return model

# Load the pre-saved datasets
with open('X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open('Y_train.pkl', 'rb') as f:
    Y_train = pickle.load(f)
with open('X_val.pkl', 'rb') as f:
    X_val = pickle.load(f)
with open('Y_val.pkl', 'rb') as f:
    Y_val = pickle.load(f)

# Create and compile the model
model = create_mlp_model(input_size=128, output_size=3)
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# Train the model with early stopping
num_epochs = 500
batch_size = 32
history = model.fit(X_train, Y_train,
                    validation_data=(X_val, Y_val),
                    epochs=num_epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping, reduce_lr],
                    verbose=1)

# Plot the losses and accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('mlp_wavelet_model_haar_level2.png', dpi=300, bbox_inches='tight')
plt.show()

# Save the model
model.save('mlp_wavelet_model_haar_level2.h5')