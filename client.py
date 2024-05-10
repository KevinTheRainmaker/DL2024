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


st.set_page_config(layout="wide")
empty1,con1,empty2 = st.columns([0.3,1.0,0.3])
empyt1,con2,con3,empty2 = st.columns([0.3,0.5,0.5,0.3])
empyt1,con4,con5,empty2 = st.columns([0.3,0.5,0.5,0.3])
empyt1,con6,empty2 = st.columns([0.3,1.0,0.3])
empyt1,con7,empty2 = st.columns([0.3,1.0,0.3])


if 'upload_file' not in ss:
    ss['upload_file'] = True

if 'process_img' not in ss:
    ss['process_img'] = False    

if 'show_result' not in ss:
    ss['show_result'] = False
    
categories = np.array(['강아지', '고양이', '토끼', '공룡', '곰', '사슴', '여우'])

def get_category_text(category):
    return "고양이 얼굴상은 인간의 얼굴이 고양이와 유사한 특징을 가질 때 사용되는 표현입니다. 이러한 얼굴상을 가진 사람들은 특히 크고 또렷한 눈, 날카로운 눈매, 작고 잘 정의된 코, 부드러운 턱선과 같은 특징을 갖고 있습니다. 또한, 고양이처럼 섬세하고 우아한 분위기를 풍기며, 신비로운 매력이나 도도한 인상을 줄 수도 있습니다."


def main():
    with empty1 :
       st.empty()
    with empty2 :
        st.empty()
    with con1:
        st.title("닮은 동물상 찾기")

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
                st.subheader("크롭된 얼굴 사진")
                st.image(ss['face_img'], use_column_width=True)
                    
        #이미지 결과 출력
        with con2:
            st.subheader("Grad-CAM Visualization")
            st.image(ss['grad_cam'], use_column_width=True)
        with con3:
            st.subheader("사진 비교")
            image_comparison(
                img1=ss['grad_cam'],
                img2=ss['face_img'],
            )
        with con4:
            st.subheader("비슷한 동물 사진")
            st.image(ss['closet_img'], use_column_width=True)                        
            # Display the prediction results as progress bars
        with con5:
            st.subheader("사진 비교")
            image_comparison(
                img1=ss['closet_img'],
                img2=ss['face_img'],
            )
        with con6:
            col1, col2 = st.columns(2)
            
            text = get_category_text(sorted_categories[0])
            st.markdown(text) 

            col1.metric("가장 비슷한 동물상", sorted_categories[0])
            col2.metric(sorted_categories[0]+"와의 유사도", sorted_predictions[0])

            #st.write("가장 비슷한 동물상은 **{}** 입니다!".format(categories[np.argmax(ss['predictions'])]))
			#닮은 동물과의 거리
            col1, col2 = st.columns([1, 3])
            with col1:
                st.write("닮은 동물 사진과의 거리")
            with col2:
                st.progress(ss['closet_dist'][0])
            

            # Inject custom CSS for each progress bar
            for category, prob in zip(sorted_categories, sorted_predictions):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.write(category)
                with col2:
                    st.progress(prob)
            # Add a button to reset the state
        with con7:           
            if st.button('다시 시도하기'):
                ss['process_img'] = False
                ss['show_result'] = False
                ss['upload_file'] = True
                ss.clear()  # Optionally clear all session state
                st.rerun()

if __name__ == '__main__':
    main()