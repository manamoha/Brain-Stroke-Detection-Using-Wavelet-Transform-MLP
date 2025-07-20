import os
import numpy as np
from PIL import Image
import pywt
from torchvision import transforms
import pickle

# Define the root directory
root_dir = "cleaned_imgs"

# Define the label mapping
label_mapping = {
    "haemorrhagic": [1, 0, 0],
    "ischemic": [0, 1, 0],
    "normal": [0, 0, 1]
}

# Load and preprocess the image with augmentation
def load_and_preprocess_image(image_path, augment=False):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    if augment:
        transform = transforms.Compose([
            transforms.RandomRotation(10),  # Random rotation up to 10 degrees
            transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of horizontal flip
            transforms.ColorJitter(brightness=0.2),  # Random brightness adjustment
            transforms.Resize((256, 256)),  # Resize to 256x256
            transforms.ToTensor(),  # Convert to tensor
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to 256x256
            transforms.ToTensor(),  # Convert to tensor
        ])
    image = transform(image).squeeze(0)  # Remove batch dimension
    image = image / 255.0
    return image.numpy()

# Apply Wavelet Transform and reduce dimensionality
def wavelet_transform(image, wavelet='haar', level=2, keep_coefficients=128):
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
    flattened_coeffs = []
    for c in coeffs:
        if isinstance(c, tuple):
            for arr in c:
                flattened_coeffs.append(arr.flatten())
        else:
            flattened_coeffs.append(c.flatten())
    flattened_coeffs = np.concatenate(flattened_coeffs)
    sorted_indices = np.argsort(np.abs(flattened_coeffs))[::-1]
    reduced_coeffs = flattened_coeffs[sorted_indices[:keep_coefficients]]
    return reduced_coeffs

def process_directory(directory, augment=False):
    X = []
    Y = []
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            label = label_mapping.get(subdir, [0, 0, 0])
            for filename in os.listdir(subdir_path):
                if filename.endswith(".jpg"):
                    image_path = os.path.join(subdir_path, filename)
                    # Original image
                    image = load_and_preprocess_image(image_path, augment=False)
                    reduced_features = wavelet_transform(image, wavelet='haar', level=2, keep_coefficients=128)
                    X.append(reduced_features)
                    Y.append(label)
                    # Augmented images (if enabled)
                    if augment:
                        for _ in range(10):  # Generate 10 augmented versions per image
                            aug_image = load_and_preprocess_image(image_path, augment=True)
                            aug_features = wavelet_transform(aug_image, wavelet='haar', level=2, keep_coefficients=128)
                            X.append(aug_features)
                            Y.append(label)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

# Process data with augmentation
X, Y = process_directory(root_dir, augment=True)
print(f"X shape (with augmentation): {X.shape}")
print(f"Y shape (with augmentation): {Y.shape}")

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42, stratify=Y_temp)

print(f"Train set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Save the datasets
with open('X_train.pkl', 'wb') as f:
    pickle.dump(X_train, f)
with open('Y_train.pkl', 'wb') as f:
    pickle.dump(Y_train, f)
with open('X_val.pkl', 'wb') as f:
    pickle.dump(X_val, f)
with open('Y_val.pkl', 'wb') as f:
    pickle.dump(Y_val, f)
with open('X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)
with open('Y_test.pkl', 'wb') as f:
    pickle.dump(Y_test, f)