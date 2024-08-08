import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet34
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
from PIL import Image

# Cihazı ayarlama (GPU varsa kullan)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Veri seti yükleme ve ön işleme
data_folder = 'processed'
image_size = (224, 224)

class CustomDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.images = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        for label in range(4):
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
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = CustomDataset(data_folder, transform=transform)
train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
train_set = torch.utils.data.Subset(dataset, train_idx)
test_set = torch.utils.data.Subset(dataset, test_idx)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# Model oluşturma
model = resnet34(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # 4 sınıf için çıktı katmanı

model = model.to(device)

# Eğitme parametreleri
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping parametreleri
patience = 3  # Performans iyileşmezse kaç epoch beklenmeli
best_model_wts = None
best_loss = float('inf')
counter = 0

# Eğitim döngüsü
num_epochs = 50  # Daha yüksek sayıda epoch belirlenmiş
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_set)
    epoch_acc = running_corrects.double() / len(train_set)

    print(f'Epoch {epoch}/{num_epochs - 1} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}')

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model_wts = model.state_dict()
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

# En iyi modeli yükle
if best_model_wts is not None:
    model.load_state_dict(best_model_wts)

# Test seti üzerinde tahminler yapma
model.eval()
y_true = []
y_pred = []
y_scores = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_scores.extend(outputs.cpu().numpy())

# Classification report oluşturma
report = classification_report(y_true, y_pred, target_names=['class_0', 'class_1', 'class_2', 'class_3'], digits=4)
print(report)

# Confusion matrix oluşturma
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# ROC AUC skorlarını hesaplama ve ROC eğrisi çizme
y_true_bin = np.eye(4)[y_true]  # One-hot encoding
y_scores = np.array(y_scores)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# ROC eğrisi çizme
plt.figure()
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'darkgreen'])
for i, color in zip(range(4), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {i} (area = {roc_auc[i]:.4f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
