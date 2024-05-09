import streamlit as st
import time
import numpy as np
from PIL import Image

st.set_page_config(layout="wide")
empty1,con1,empty2 = st.columns([0.3,1.0,0.3])
empyt1,con2,con3,empty2 = st.columns([0.3,0.5,0.5,0.3])
empyt1,con4,empty2 = st.columns([0.3,1.0,0.3])

categories = ['강아지', '고양이', '토끼', '공룡', '곰', '사슴', '여우']

def main():
    with empty1 :
       st.empty()
    with con1:
        st.title("닮은 동물상 찾기")
        st.subheader("이미지를 업로드하세요.")
        uploaded_file = st.file_uploader(label="", type=["jpg", "jpeg", "png"], key="file_uploader")

        if uploaded_file is not None:
            # PIL Image로 변환
            upload_img = Image.open(uploaded_file)
            #업로드한 이미지
            #face_img = process_image(upload_img)
            face_img = Image.open(uploaded_file)
            #강아지,고양이,토끼,공룡,곰,사슴,여우
            with st.spinner('사진을 분류중입니다.'):
                result = time.sleep(2)
                #prediction = get_prediction(face_img)
                predictions = np.random.rand(7)
                #grad_cam = get_gradcam(face_img)
                grad_cam = Image.open(uploaded_file)
                #closet_img = get_closet(face_img)
                closet_img = Image.open(uploaded_file)
                
            with con1:
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.image(face_img, caption='크롭된 얼굴 사진', use_column_width=True)
            #이미지 결과 출력
            with con2:
                st.image(grad_cam, caption='Grad-CAM Visualization', use_column_width=True)
            with con3:
                st.image(closet_img, caption='가장 비슷한 동물', use_column_width=True)
            with con4:
                # Display the prediction results as progress bars
                st.write("가장 비슷한 동물상은 **{}** 입니다!".format(categories[np.argmax(predictions)]))
    
                # Inject custom CSS for each progress bar
                for category, prob in zip(categories, predictions):
                    with st.container():
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.write(category)
                        with col2:
                            st.progress(prob)
    with empty2 :
        st.empty()

if __name__ == '__main__':
    main()