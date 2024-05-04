import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.semi_supervised import LabelSpreading
from PIL import Image

from tqdm import tqdm

import warnings

warnings.filterwarnings('ignore')

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f'{device = }')

# 데이터셋을 위한 커스텀 데이터 로더
class CustomImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.data = []
        self.classes = set()
        self.label_map = {}
        self._load_dataset()

    def _load_dataset(self):
        idx = 0
        for label in tqdm(os.listdir(self.root)):
            label_dir = os.path.join(self.root, label)
            if os.path.isdir(label_dir):
                if label == "unknown":
                    label_id = -1
                else:
                    if label not in self.label_map:
                        self.label_map[label] = idx
                        idx += 1
                    label_id = self.label_map[label]
                for image_file in os.listdir(label_dir):
                    image_path = os.path.join(label_dir, image_file)
                    self.data.append((image_path, label_id))
                self.classes.add(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def _get_label_map(self):
        return self.label_map
    
    def update_labels(self, new_labels):
        self.labels = new_labels

    def _get_label(self, image_path):
        label = os.path.basename(os.path.dirname(image_path))
        return -1 if label == 'unknown' else self.label_map[label]
    
# 이미지 변환 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 데이터셋 로드
dataset = CustomImageDataset(root="./images", transform=transform)
label_map = dataset._get_label_map()

print(f'{label_map = }')

model = models.resnet18(pretrained=True)
# 마지막 레이어 제거 (fully connected layer)
model = torch.nn.Sequential(*list(model.children())[:-1])
# 모델을 GPU로 이동
model = model.to(device)

# 피처와 레이블 추출
features = []
labels = []

# DataLoader를 사용하여 데이터셋 로딩
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 배치 단위로 데이터 추출
for imgs, lbls in tqdm(dataloader):
    imgs, lbls = imgs.to(device), lbls.to(device)
    # 모델에 이미지 전달하여 피처 추출
    with torch.no_grad():
        output = model(imgs)
    features.append(output.cpu().squeeze())  # 피처를 CPU로 이동한 후 features 리스트에 추가
    labels.append(lbls.cpu())  # 레이블을 CPU로 이동한 후 labels 리스트에 추가

# 리스트를 텐서로 변환
features = torch.cat(features)
labels = torch.cat(labels).numpy()

# # 데이터셋에서 이미지와 레이블 추출
# features = []
# labels = []
# for img, label in tqdm(dataset):
#     features.append(torch.flatten(img))
#     labels.append(label)
# features = torch.stack(features)
# labels = np.array(labels)

# 레이블 전파
initial_fraction = (labels == -1).mean()
print(f'{initial_fraction = }')
label_spread = LabelSpreading(kernel='knn', n_neighbors=9)
# label_spread.fit(features, labels)
unlabeled_fraction = (labels == -1).mean()
former = 1.0
print(f'{unlabeled_fraction = }')

while unlabeled_fraction > 0.01:
    label_spread.fit(features.numpy(), labels)
    labels = label_spread.transduction_
    unlabeled_fraction = (labels == -1).mean()
    if former > unlabeled_fraction:
        print(f'{unlabeled_fraction = }')
    former = unlabeled_fraction

print('label spreading done.')

# 업데이트된 레이블을 데이터셋에 적용
new_labels = []
idx = 0
print('update dataset')
for img, _ in tqdm(dataset):
    label = dataset._get_label(dataset.data[idx][0])
    new_labels.append(labels[idx] if label != -1 else -1)
    idx += 1

dataset.update_labels(new_labels)

# -1 레이블을 가진 데이터를 제외하고 학습
filtered_indices = [idx for idx, (_, label) in enumerate(dataset) if label != -1]
filtered_dataset = Subset(dataset, filtered_indices)

# 트레이닝 및 테스트 데이터 분리
train_size = int(0.8 * len(filtered_dataset))
test_size = len(filtered_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(filtered_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 모델 설정
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)
model = model.to(device)

# 손실 함수와 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
6
# 학습 루프
def train_model(model, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    return model

# 모델 학습
trained_model = train_model(model, criterion, optimizer, num_epochs=10)

# 모델 저장
torch.save(trained_model.state_dict(), 'model.pth')

print("Model trained and saved as model.pth")
