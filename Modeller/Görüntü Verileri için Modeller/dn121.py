import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Veri setini yükleme ve ön işleme
data_folder = 'processed_kirpilmamis'
image_size = (224, 224)
num_classes = 4  # Sınıf sayısı

class CustomDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.images = []
        self.labels = []
        self.load_data()

    def load_data(self):
        for label in range(num_classes):
            label_folder = os.path.join(self.data_folder, f'label_{label}')
            for filename in os.listdir(label_folder):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(label_folder, filename)
                    self.images.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = CustomDataset(data_folder, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Modelin oluşturulması
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.densenet121(pretrained=True)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier.in_features, 1024),
    nn.ReLU(),
    nn.Linear(1024, num_classes)
)
model = model.to(device)

# Eğitme parametreleri
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
early_stopping_patience = 5

# Modelin eğitilmesi
best_loss = float('inf')
patience_counter = 0

for epoch in range(50):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    val_loss /= len(test_loader)
    print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}, Val Loss: {val_loss}')
    
    if val_loss < best_loss:
        best_loss = val_loss
        best_model_wts = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= early_stopping_patience:
        print('Early stopping')
        break

model.load_state_dict(best_model_wts)

# Test seti üzerinde tahminler yapma
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Classification report oluşturma
report = classification_report(all_labels, all_preds, target_names=['class_0', 'class_1', 'class_2', 'class_3'], digits=4)
print(report)
