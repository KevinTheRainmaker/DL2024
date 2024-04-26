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

def main(image_paths, known_labels):
    pseudo_labels, updated_labels = iterative_labeling(image_paths, known_labels)
    while len(pseudo_labels) / len(image_paths) < 0.7:
        _, updated_labels = iterative_labeling(image_paths, updated_labels)
        pseudo_labels.append(_)

if __name__ == "__main__":
    image_paths = ["./images/test.jpg"]
    known_labels = [3]
    # dataset = AnimalFacesDataset(image_paths, labels=known_labels, transform=transform)
    # features, labels = extract_features(dataset)
    # print(f'{features = }')
    # print(f'{labels = }')
    main(image_paths, known_labels)
    