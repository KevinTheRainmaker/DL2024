import streamlit as st
from streamlit import session_state as ss
from streamlit_lottie import st_lottie_spinner
from streamlit_image_comparison import image_comparison
from st_files_connection import FilesConnection

import numpy as np
import json
import os
import csv

import gdown
import faiss

from PIL import Image
import cv2

import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import s3fs
import warnings

warnings.filterwarnings('ignore')

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

#레이아웃 설정
st.set_page_config(
    page_title="닮은 얼굴상 찾기",
    page_icon="🐶",
    layout="wide")

empty1,con0,empty2 = st.columns([0.5,0.5,0.5])
empty1,con1,con2,empty3 = st.columns([0.2,0.5,0.5,0.2])
empyt1,con3,con4,empty3 = st.columns([0.2,0.5,0.5,0.2])
empyt1,con5,empty2 = st.columns([0.2,1.0,0.2])
empyt1,con6,empty2 = st.columns([0.2,1.0,0.2])

conn = st.connection('s3', type=FilesConnection)
fs = s3fs.S3FileSystem(anon=False)

os.makedirs('./temp', exist_ok=True)
# os.makedirs('./faiss', exist_ok=True)

# def download_drive_file(output, url):
    # gdown.download(url, output, quiet=False)

# download_drive_file('faiss', 'https://drive.google.com/file/d/1QSzpvyfEqYM2dbzLPiwVnZdsExP0rMn3/view?usp=sharing')

# print(os.getcwd())
# print(os.listdir('./'))

# file_name = "faiss.zip"
# output_dir = "faiss"
# os.system("unzip "+file_name+" -d "+output_dir)
    
# print(os.getcwd())
# for root, dirs, files in os.walk("./faiss"):
#     path = root.split(os.sep)
#     print((len(path) - 1) * '---', os.path.basename(root))
#     for file in files:
#         print(len(path) * '---', file)
# print('################')
# JSON 파일 경로
file_path = 'asset/loading.json'
# 파일을 열고 JSON 데이터 읽기
with open(file_path, 'r') as file:
    lottie_animation = json.load(file)

#화면상태를 의미하는 세션 상태
if 'upload_file' not in ss: #파일 업로드 화면
    ss['upload_file'] = True

if 'process_img' not in ss:#이미지 처리 화면
    ss['process_img'] = False    

if 'show_result' not in ss:#결과 출력 화면
    ss['show_result'] = False

#모델 라벨 카테고리.
categories = np.array(['비글','보더콜리','여우','호랑이','사자','장모종 고양이','치타','단모종 고양이','도베르만','리트리버','늑대','시츄'])

class_map = {'0':'dog','1':'dog','8':'dog','9':'dog','11':'dog',
                '5':'cat','7':'cat','2':'fox','3':'tiger','4':'lion',
                '6':'cheetah','10':'wolf'}
                
animal_text = {
    '비글': '밝고 쾌활한 느낌, 큰 귀와 맑은 눈이 특징. 활발하고 호기심 많은 성격으로, 사람들과 쉽게 어울리는 사람.',
    '보더콜리': '영리하고 집중력 있는 인상, 날렵한 눈매와 날카로운 눈빛이 특징. 활동적이며 목표 지향적인 성격을 가진 사람.',
    '여우': '영리하고 교활한 이미지, 날카로운 눈매와 뾰족한 이목구비가 특징. 민첩하고 야무진 성격을 가진 사람.',
    '호랑이': '강렬하고 위엄 있는 인상, 날카로운 이목구비가 특징. 용맹하고 결단력 있는 성격을 가진 사람.',
    '사자': '위엄 있고 고귀한 느낌, 강렬한 눈빛이 특징. 리더십과 자신감을 상징하는 사람.',
    '장모종 고양이': '우아하고 부드러운 인상, 큰 눈이 특징. 독립적이고 고상한 성격을 가진 사람.',
    '치타': '날렵하고 빠른 이미지, 작은 얼굴이 특징. 민첩하고 활동적인 성격을 가진 사람.',
    '단모종 고양이': '깔끔하고 날렵한 인상, 날카로운 눈이 특징. 독립적이며 호기심 많은 성격을 가진 사람.',
    '도베르만': '강인하고 날카로운 느낌, 강렬한 눈빛이 특징. 용맹하고 보호 본능이 강한 성격을 가진 사람.',
    '리트리버': '친근하고 따뜻한 느낌, 밝은 눈빛이 특징. 사교적이고 충성스러운 성격을 가진 사람.',
    '늑대': '강렬하고 신비로운 인상, 날카로운 이목구비가 특징. 자유롭고 야성적인 성격을 가진 사람.',
    '시츄': '귀엽고 사랑스러운 느낌, 둥근 얼굴과 큰 눈이 특징. 애교 많고 온순한 성격을 가진 사람.'
}


st.markdown("""
            <style>
            h2 {
                color: #7340bf;
                justify-content: center;
            }
            </style>
            """, unsafe_allow_html=True
            )

st.markdown("""
            <style>
            h3 {
                color: #7340bf;
                justify-content: center;
            }
            </style>
            """, unsafe_allow_html=True
            )
    
def get_category_text(category):
    return animal_text[category]

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
    transforms.Resize((224, 224)),  # 이미지 크기 조정
    transforms.ToTensor(),  # 이미지를 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
])

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

def load_model(classes=12):
    # 모델 정의 (ResNet50)
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, classes)
    
    # 학습된 모델의 가중치 로드
    model.load_state_dict(torch.load('./model.pth', map_location=device))
    model.eval()
    return model

def load_extractor():
    model = models.resnet50(pretrained=True)  # 예시로 ResNet-50 모델 사용
    model.eval()  # 평가 모드로 설정
    return model

def predict_with_gradcam(model, PILimage):
    # 이미지 불러오기
    image = PILimage.convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    output = model(image_tensor)
    probs = torch.softmax(output, dim=1)  # 확률로 변환
    probs = probs.squeeze(0).detach().numpy() # 텐서를 넘파이 배열로 변환

    # 가장 높은 확률의 클래스 추출
    predicted_label = np.argmax(probs)
    # probs = [int(round(prob)) for prob in probs]

    # Grad-CAM 계산
    grad_cam = GradCAM(model=model, target_layer=model.layer4)
    cam = grad_cam.generate(image_tensor)

    # 이미지와 CAM을 함께 시각화
    img = np.array(PILimage.convert('RGB').resize((224, 224)))
    img = np.float32(img) / 255
    cam = cv2.resize(cam, (224, 224))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam_img = heatmap + np.float32(img)
    cam_img = cam_img / np.max(cam_img)
    cv2.imwrite('temp/cam.jpg', np.uint8(255 * cam_img))
    return predicted_label, probs

def load_faiss_index(path):
    return faiss.read_index(path)

def image_to_tensor(PILimage):

    image = PILimage.convert('RGB')
    return transform(image).unsqueeze(0)  # 배치 차원 추가

def extract_features(model, image_tensor):
    with torch.no_grad():
        features = model(image_tensor)
    return features.numpy().flatten()  # 벡터로 변환

def search_similar_images(index, query_vector, k=1):
    distances, indices = index.search(np.array([query_vector]), k)
    return distances, indices

def get_all_image_paths(root_folder):
    image_paths = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(dirpath, filename))
    return image_paths

def rerun_app():
    ss['process_img'] = False
    ss['show_result'] = False
    ss['upload_file'] = True
    ss.clear()
    st.rerun()
    # st.experimental_rerun()

def main():
    with empty1 :
        st.empty()
    with empty2 :
        st.empty()
    with con0:
        st.markdown("<h1 style='text-align: center;'>닮은 동물상 찾기 🐶</h1>", unsafe_allow_html=True)
        

    if ss['upload_file']:
        
        with con0:
            st.subheader("이미지를 업로드하세요.")
            uploaded_file = st.file_uploader(label=" ", type=["jpg", "jpeg", "png"], key="file_uploader")
        if uploaded_file is not None:
            ss['upload_file'] = False
            ss['process_img'] = True
            ss['image'] = uploaded_file  # backup the file
            st.rerun()
    if ss['process_img']:        
        # PIL Image로 변환
        upload_img = Image.open(ss['image'])
        upload_img_gray = upload_img.convert('L')
        logger.debug("Image loaded successfully.")
        logger.debug(f"Image size: {upload_img.size}")
        logger.debug(f"Image size: {upload_img_gray.size}")

        ss['face_img'] = upload_img
        with con0:
            with st_lottie_spinner(lottie_animation, key="download"):     
                model = load_model(12)

                ss['predictions'], ss['probs'] = predict_with_gradcam(model, upload_img_gray)
                
                cv2.imwrite('temp/cam.jpg', cv2.resize(cv2.imread('temp/cam.jpg'), upload_img.size))
                ss['grad_cam'] = Image.open('temp/cam.jpg')
                
                extractor = load_extractor()
                
                lcategory = class_map[str(ss['predictions'])]
                loaded_index = load_faiss_index(f'faiss/faiss_{lcategory}.index')
                logger.debug("faiss loaded.")
                query_image_tensor = image_to_tensor(upload_img)
                query_vector = extract_features(extractor, query_image_tensor)
                logger.debug("features extracted")
                distances, indices = search_similar_images(loaded_index, query_vector, k=5)
                logger.debug("similar images searched")
                df = conn.read(f"dl2024-bucket/list/{lcategory}_list.csv", input_format="csv", ttl=600)
                image_files = df.iloc[:, 0].tolist()
                for idx, distance in zip(indices[0], distances[0]):
                    if image_files[idx].split('/')[2] == lcategory:
                        # ss['closest_img'] = Image.open(image_files[idx])
                        ss['closest_img'] = Image.open(fs.open(f'dl2024-bucket{image_files[idx]}', mode='rb').read())
                        ss['closest_dist'] = distance
                        break
                             
    if ss['show_result']: 
        result_category = categories[ss['predictions']]
        probability = ss['probs'][ss['predictions']]

        with con1:
            # _, col, _ = st.columns([1, 3, 1])
            # with col:
            st.markdown("<h3 style='text-align: center;'>원본 사진</h3>", unsafe_allow_html=True)
            # st.image(ss['face_img'], use_column_width='auto')
            st.image(ss['face_img'], width=500)
                
        with con2:
            st.markdown("<h3 style='text-align: center;'>가장 비슷한 동물상</h3>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            col1.metric("",result_category)
            col2.metric("", round(probability*100,2))
                        # Inject custom CSS for each progress bar

            sorted_indices = np.argsort(ss['probs'])[::-1]
            ss['probs'] = ss['probs'][sorted_indices]
            sorted_categories = categories[sorted_indices]

            top_k = 0
            for category, prob in zip(sorted_categories, ss['probs']):
                prob = int(round(prob*100))

                col5, col6 = st.columns(2)
                with col5:
                    st.write(category)
                with col6:
                    st.progress(prob)
                top_k += 1
                if top_k == 10:
                    break
        with con3:
            st.markdown("<h3 style='text-align: center;'>Grad-CAM Visualization</h3>", unsafe_allow_html=True)
            image_comparison(
                img2=ss['face_img'],
                img1=ss['grad_cam'],
                width=500,
            )
            
        with con4:
            st.markdown("<h3 style='text-align: center;'>비슷한 동물 사진</h3>", unsafe_allow_html=True)
            image_comparison(
                img2=ss['face_img'],
                img1=ss['closest_img'],
                width=500,
            )

        with con5:
            text = get_category_text(result_category)
            cat_text = f'<h2 style="text-align: center;">{text}</h2>'
            st.markdown(cat_text, unsafe_allow_html=True) 

        with con6:           
            st.markdown(
                """
            <style>
            button {
                height: 60px;
                font-weight: bold;
                font-style: italic
                font-size: 54px; !important
            }
            </style>
            """,
                unsafe_allow_html=True,
            )
            rerun_text = '다시 시도하기'
            if st.button(rerun_text, use_container_width=True, type="primary"):
                rerun_app()
                    
if __name__ == '__main__':
    main()