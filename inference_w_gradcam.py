import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

import warnings

warnings.filterwarnings('ignore')

# Grad-CAM 클래스 정의
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradient = None
        self.activation = None
        self.model.eval()
        self.register_hooks()

    def register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0]

        def forward_hook(module, input, output):
            self.activation = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def forward(self, x):
        return self.model(x)

    def backward(self, gradients):
        self.model.zero_grad()
        gradients.backward(retain_graph=True)

    def generate(self, x):
        output = self.forward(x)
        output = output.max(1)[0]
        self.backward(output)
        pooled_gradients = torch.mean(self.gradient, dim=[0, 2, 3])
        activations = self.activation.squeeze(0)
        grad_cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, weight in enumerate(pooled_gradients):
            grad_cam += weight * activations[i, :, :]
        grad_cam = nn.functional.relu(grad_cam)
        grad_cam /= torch.max(grad_cam)
        return grad_cam.detach().numpy()

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# 모델 정의 (ResNet50)
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)

# 학습된 모델의 가중치 로드
model.load_state_dict(torch.load('model.pth'))
model.eval()

def predict_with_gradcam(model, image_path):
    # 이미지 불러오기
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    output = model(image_tensor)
    probs = torch.softmax(output, dim=1)  # 확률로 변환
    probs = probs.squeeze(0).detach().numpy() * 100 # 텐서를 넘파이 배열로 변환

    # 가장 높은 확률의 클래스 추출
    predicted_label = np.argmax(probs)
    probs = [int(round(prob)) for prob in probs]

    # Grad-CAM 계산
    grad_cam = GradCAM(model=model, target_layer=model.layer4)
    cam = grad_cam.generate(image_tensor)

    # 이미지와 CAM을 함께 시각화
    img = cv2.imread(image_path, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    cam = cv2.resize(cam, (224, 224))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam_img = heatmap + np.float32(img)
    cam_img = cam_img / np.max(cam_img)

    cv2.imwrite('cam.jpg', np.uint8(255 * cam_img))
    return predicted_label, probs

# 예측 실행
predicted_label, probs = predict_with_gradcam(model, 'wolf_test.png')
print("Predicted Label:", predicted_label)
print("Class Probabilities:", probs)