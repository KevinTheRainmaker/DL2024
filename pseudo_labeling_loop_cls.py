import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import DBSCAN
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim

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

if __name__ == "__main__":
    image_paths = ["./images/test.jpg"]
    known_labels = [10]
    dataset = AnimalFacesDataset(image_paths, labels=known_labels, transform=transform)
    features, labels = extract_features(dataset)
    print(f'{features = }')
    print(f'{labels = }')
    