import os
import torch
import faiss
import numpy as np
from PIL import Image
from torchvision import models, transforms

from tqdm import tqdm

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import warnings

warnings.filterwarnings('ignore')

def load_model():
    model = models.resnet50(pretrained=True)  # 예시로 ResNet-50 모델 사용
    model.eval()  # 평가 모드로 설정
    return model

def image_to_tensor(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 이미지 크기 조정
        transforms.ToTensor(),  # 이미지를 텐서로 변환
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # 배치 차원 추가

def extract_features(model, image_tensor):
    with torch.no_grad():
        features = model(image_tensor)
    return features.numpy().flatten()  # 벡터로 변환

def build_faiss_index(feature_vectors):
    d = feature_vectors.shape[1]  # 벡터 차원
    index = faiss.IndexFlatL2(d)  # L2 거리를 사용하는 인덱스 생성
    index.add(feature_vectors)  # 인덱스에 벡터 추가
    return index

def save_faiss_index(index, path):
    faiss.write_index(index, path)

def load_faiss_index(path):
    return faiss.read_index(path)

def search_similar_images(index, query_vector, k=1):
    distances, indices = index.search(np.array([query_vector]), k)
    return distances, indices

def get_all_image_paths(root_folder):
    image_paths = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in tqdm(filenames):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(dirpath, filename))
    return image_paths

# 모델 로드
model = load_model()

# # 이미지 폴더 (루트)
image_folder = f'./images'
image_files = get_all_image_paths(image_folder)

# # 이미지 벡터 추출
# feature_vectors = np.array([extract_features(model, image_to_tensor(f)) for f in tqdm(image_files, desc="Extracting features")])

# # FAISS 인덱스 생성
# index = build_faiss_index(feature_vectors)

index_path = './faiss.index'
# save_faiss_index(index, index_path)

# 인덱스 로드
loaded_index = load_faiss_index(index_path)

# 쿼리 이미지
query_image_path = './cat_test.png'
query_image_tensor = image_to_tensor(query_image_path)
query_vector = extract_features(model, query_image_tensor)

# 가장 유사한 이미지 찾기
distances, indices = search_similar_images(loaded_index, query_vector, k=5)

# 결과 출력
label='cat'
for idx, distance in zip(indices[0], distances[0]):
    if image_files[idx].split('/')[2] == label:
        print(f"{image_files[idx]} with distance {distance}")
        break