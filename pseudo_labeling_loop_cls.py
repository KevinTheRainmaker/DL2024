import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import DBSCAN
from sklearn.semi_supervised import LabelPropagation
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os

from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class AnimalFacesDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = -1 if self.labels is None else self.labels[idx]
        return image, label

def load_images_from_folder(folder):
    images = []
    labels = []
    label_map = {}
    current_label = 0

    # 폴더 순회
    for label_folder in os.listdir(folder):
        label_path = os.path.join(folder, label_folder)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                if img_path.lower().endswith(('png', 'jpg', 'jpeg')):
                    images.append(img_path)
                    if label_folder == "unknown":  # 'unknown' 폴더 처리
                        labels.append(-1)
                    else:
                        if label_folder not in label_map:
                            label_map[label_folder] = current_label
                            current_label += 1
                        labels.append(label_map[label_folder])

    return images, labels, label_map

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

feature_extractor = resnet50(pretrained=True)
feature_extractor = nn.Sequential(*(list(feature_extractor.children())[:-1]))
feature_extractor.eval()

def extract_features(dataset):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    features = []
    labels = []
    with torch.no_grad():
        for images, lbls in loader:
            output = feature_extractor(images).flatten(start_dim=1)
            features.append(output)
            labels.extend(lbls)
    features = torch.cat(features, dim=0)
    return features.numpy(), np.array(labels)

def iterative_labeling(image_paths, initial_labels):
    dataset = AnimalFacesDataset(image_paths, labels=initial_labels, transform=transform)
    features, labels = extract_features(dataset)

    label_prop_model = LabelPropagation(kernel='rbf', gamma=20, n_neighbors=7)
    label_prop_model.fit(features, labels)
    new_labels = label_prop_model.transduction_

    pseudo_labels = []
    for i, label in enumerate(new_labels):
        if labels[i] == -1 and label_prop_model.label_distributions_[i].max() > 0.8:
            pseudo_labels.append((i, label))

    return pseudo_labels, new_labels

def train_final_model(dataset):
    model = resnet50(pretrained=False)
    model.fc = nn.Linear(2048, 3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    model.train()

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for epoch in range(10):  # 에포크 설정
        for images, labels in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    now = datetime.now()
    torch.save(model.state_dict(), f'model_checkpoint_{now.strftime('%m-%d-%H-%M')}.pth')
    print("Model checkpoint saved.")

def main(image_paths, known_labels):
    pseudo_labels, updated_labels = iterative_labeling(image_paths, known_labels)
    while len(pseudo_labels) / len(image_paths) < 0.7:
        _, updated_labels = iterative_labeling(image_paths, updated_labels)
        pseudo_labels.append(_)
        
    final_dataset = AnimalFacesDataset(image_paths, labels=updated_labels, transform=transform)
    train_final_model(final_dataset)

if __name__ == "__main__":
    image_paths, known_labels, label_map = load_images_from_folder('./images')
    print(f'{image_paths=}')
    print(f'{known_labels=}')
    print(f'{label_map=}')
    # dataset = AnimalFacesDataset(image_paths, known_labels, transform=transform)
    # train_final_model(dataset)
    # image_paths = ["./images/test.jpg"]
    # known_labels = [3]
    # dataset = AnimalFacesDataset(image_paths, labels=known_labels, transform=transform)
    # features, labels = extract_features(dataset)
    # print(f'{features = }')
    # print(f'{labels = }')
    # main(image_paths, known_labels)
    