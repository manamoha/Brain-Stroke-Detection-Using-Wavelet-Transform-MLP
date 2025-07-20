import numpy as np
from tensorflow.keras import models
import pickle
from sklearn.metrics import classification_report

# Load the pre-saved test datasets
with open('X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open('Y_test.pkl', 'rb') as f:
    Y_test = pickle.load(f)

# Define the label mapping
label_mapping = {
    "haemorrhagic": [1, 0, 0],
    "ischemic": [0, 1, 0],
    "normal": [0, 0, 1]
}

# Load the trained model
model = models.load_model('mlp_wavelet_model_haar_level2.h5')

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Predict on the test set
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true_classes = np.argmax(Y_test, axis=1)

# Print classification report
report = classification_report(Y_true_classes, Y_pred_classes, target_names=label_mapping.keys())
print(report)

# Calculate the number of correct non-normal class
correct_non_normal = np.sum((Y_pred_classes != 2) & (Y_pred_classes == Y_true_classes))
total_non_normal = np.sum(Y_true_classes != 2)
print(f"Correct non-normal class: {correct_non_normal}/{total_non_normal} ({correct_non_normal/total_non_normal*100:.2f}%)")
