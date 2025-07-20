import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms
transform_train = transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Dataset
dataset = datasets.ImageFolder(root="cleaned_imgs", transform=transform_train)

# Train/Val/Test split
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Set test/val transforms to deterministic
val_dataset.dataset.transform = transform_test
test_dataset.dataset.transform = transform_test

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# test_loader is not needed if you’re using Keras evaluation separately

# CNN Model
class StrokeCNN(nn.Module):
    def __init__(self):
        super(StrokeCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> (32, 63, 63)
        x = self.pool(F.relu(self.conv2(x)))  # -> (64, 30, 30)
        x = self.pool(F.relu(self.conv3(x)))  # -> (128, 14, 14)
        x = x.view(x.size(0), -1)             # flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Model, Loss, Optimizer
model = StrokeCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss:.4f} | Train Accuracy: {acc:.2f}%")

# saving the model
torch.save(model.state_dict(), 'cnn_stroke_model.pth')

# Evaluate on validation set
def evaluate(model, dataloader, name="Validation"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"{name} Accuracy: {100 * correct / total:.2f}%")

# Make predictions on test set
def predict_and_evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Classification report
    target_names = ['haemorrhagic', 'ischemic', 'normal']
    print("🔎 Classification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("🧾 Confusion Matrix:")
    print(cm)

    # Stroke sensitivity = (stroke predicted as stroke) / (total stroke)
    # Classes: 0=haemorrhagic, 1=ischemic, 2=normal
    stroke_true = (all_labels == 0) | (all_labels == 1)
    stroke_pred = (all_preds == 0) | (all_preds == 1)
    stroke_correct = np.sum(stroke_true & stroke_pred)
    stroke_total = np.sum(stroke_true)
    stroke_sensitivity = stroke_correct / stroke_total * 100
    print(f"\n🧠 Overall Stroke Sensitivity: {stroke_correct}/{stroke_total} = {stroke_sensitivity:.2f}%")

    return all_labels, all_preds

# Train Accuracy
evaluate(model, train_loader, "Train")

# Validation Accuracy
evaluate(model, val_loader, "Validation")

# Test Accuracy + Detailed Report
_, _ = predict_and_evaluate(model, test_loader)
