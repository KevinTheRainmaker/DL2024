import streamlit as st
import time
import numpy as np
import json
from PIL import Image
from streamlit import session_state as ss
from streamlit_lottie import st_lottie_spinner
from streamlit_image_comparison import image_comparison
import json

# JSON 파일 경로
file_path = 'asset/loading.json'
# 파일을 열고 JSON 데이터 읽기
with open(file_path, 'r') as file:
    lottie_animation = json.load(file)

#레이아웃 설정
st.set_page_config(
    page_title="닮은 얼굴상 찾기",
    page_icon="🐶",
    layout="wide")
empty1,con1,empty2 = st.columns([0.5,1.0,0.5])
empyt1,con2,con3,empty2 = st.columns([0.3,0.5,0.5,0.3])
# empyt1,con4,con5,empty2 = st.columns([0.5,0.5,0.5,0.5])
empyt1,con4,empty2 = st.columns([0.4,1.2,0.4])
empyt1,con5,empty2 = st.columns([0.4,1.2,0.4])

#화면상태를 의미하는 세션 상태
if 'upload_file' not in ss: #파일 업로드 화면
    ss['upload_file'] = True

if 'process_img' not in ss:#이미지 처리 화면
    ss['process_img'] = False    

if 'show_result' not in ss:#결과 출력 화면
    ss['show_result'] = False

#모델 라벨 카테고리.
categories = np.array(['강아지', '고양이', '토끼', '공룡', '곰', '사슴', '여우'])

animal_text = {
    '강아지': '밝고 친근한 느낌, 둥근 얼굴과 큰 눈이 특징. 사람들과 쉽게 친해지며, 순수하고 선한 인상을 줌.',
    '고양이': '날카로운 눈매와 작은 코, 우아하고 신비로운 매력이 특징. 독립적이고 자신감 있는 태도를 가짐.',
    '토끼': '긴 눈과 볼륨감 있는 볼, 부드러운 인상이 특징. 순수하고 상냥한 성격을 가진 것으로 인식됨.',
    '공룡': '강인하고 드물게 보는 특이한 얼굴 특징, 강렬하고 인상적인 모습을 가짐.',
    '곰': '큰 얼굴과 뚜렷한 이목구비, 포근하고 따뜻한 느낌을 줌. 보호 본능과 강인함을 연상시킴.',
    '사슴': '긴 속눈썹과 큰 눈, 우아하고 섬세한 인상. 조용하고 차분한 성격을 표현하는데 적합.',
    '여우': '영리하고 교활한 이미지, 날카로운 눈매와 뾰족한 이목구비가 특징. 민첩하고 야무진 성격을 연상시킴.'
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

def main():
    with empty1 :
       st.empty()
    with empty2 :
        st.empty()
    with con1:
        st.title("닮은 동물상 찾기 🐶")

    if ss['upload_file']:
        with con1:
            st.subheader("이미지를 업로드하세요.")
            uploaded_file = st.file_uploader(label="", type=["jpg", "jpeg", "png"], key="file_uploader")
        if uploaded_file is not None:
            ss['upload_file'] = False
            ss['process_img'] = True
            ss['image'] = uploaded_file  # backup the file
            st.rerun()
                
    if ss['process_img']:        
        # PIL Image로 변환
        upload_img = Image.open(ss['image'])

        #face_img = process_image(upload_img)
        ss['face_img'] = upload_img
        with con1:
    	    with st_lottie_spinner(lottie_animation, key="download"):
                #로딩 화면 테스트용 더미 시간
                time.sleep(2)
                
                #prediction = get_prediction(face_img)
                ss['predictions'] = np.random.rand(7)

                #grad_cam = get_gradcam(face_img)
                ss['grad_cam'] = upload_img
            
                #closet_img, closet_dist = get_closet(face_img)
                ss['closet_img'] = Image.open("asset/testresult2.jpg")
                ss['closet_dist'] = np.random.rand(1)
        ss['process_img'] = False
        ss['show_result'] = True
        st.rerun()
            
    if ss['show_result']: 
        sorted_indices = np.argsort(ss['predictions'])[::-1]
        # predictions와 categories를 같은 인덱스로 정렬
        sorted_predictions = ss['predictions'][sorted_indices]
        sorted_categories = categories[sorted_indices]

        with con1:
            _, col, _ = st.columns([1, 3, 1])
            with col:
                st.markdown("<h3>크롭된 얼굴 사진</h3>", unsafe_allow_html=True)
                st.image(ss['face_img'], use_column_width=True)
                
        with con2:
            st.markdown("<h3>Grad-CAM Visualization</h3>", unsafe_allow_html=True)
            image_comparison(
                img2=ss['face_img'],
                img1=ss['grad_cam'],
                width=350,
            )
            
        with con3:
            st.markdown("<h3>비슷한 동물 사진</h3>", unsafe_allow_html=True)
            image_comparison(
                img2=ss['face_img'],
                img1=ss['closet_img'],
                width=350,
            )

        with con4:
            text = get_category_text(sorted_categories[0])
            cat_text = f'<h2>{text}</h2>'
            st.markdown(cat_text, unsafe_allow_html=True) 
            
            #이미지 결과 출력
            col1, col2 = st.columns(2)
            st.markdown("""
                        <style>
                        div[data-testid="metric-container"] {
                        background-color: rgba(28, 131, 225, 0.1);
                        border: 1px solid rgba(28, 131, 225, 0.1);
                        padding: 5% 5% 5% 10%;
                        border-radius: 5px;
                        color: rgb(30, 103, 119);
                        overflow-wrap: break-word;
                        }
                        </style>
                        """
                        , unsafe_allow_html=True)
            col1.metric("가장 비슷한 동물상",sorted_categories[0])
            col2.metric(sorted_categories[0]+"상인 정도", sorted_predictions[0])

        with con5:
            #st.write("가장 비슷한 동물상은 **{}** 입니다!".format(categories[np.argmax(ss['predictions'])]))
			#닮은 동물과의 거리
            col3, col4 = st.columns(2)
            with col3:
                st.write("닮은 동물 사진과의 거리")
            with col4:
                st.progress(ss['closet_dist'][0])
            

            # Inject custom CSS for each progress bar
            for category, prob in zip(sorted_categories, sorted_predictions):
                col5, col6 = st.columns(2)
                with col5:
                    st.write(category)
                with col6:
                    st.progress(prob)


        with con5:           
            # Add a button to reset the state
            if st.button('다시 시도하기', use_container_width=True):
                ss['process_img'] = False
                ss['show_result'] = False
                ss['upload_file'] = True
                ss.clear()  # Optionally clear all session state
                st.rerun()

if __name__ == '__main__':
    main()