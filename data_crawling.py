from google_images_search import GoogleImagesSearch
from PIL import Image
import os
import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
GOOGLE_CUSTOM_SEARCH = os.environ.get('GOOGLE_CUSTOM_SEARCH')
SEARCH_ENGINE_ID = os.environ.get('SEARCH_ENGINE_ID')

gis = GoogleImagesSearch(GOOGLE_CUSTOM_SEARCH, SEARCH_ENGINE_ID)

def download_and_label_images(search_term, download_path, number_images=5):
    _search_params = {
        'q': search_term,
        'num': number_images,
        'fileType': 'jpg|png',
        'imgType': 'face',
    }

    gis.search(search_params=_search_params)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    image_data = []
    
    for idx, image in enumerate(gis.results()):
        try:
            raw_image_data = image.get_raw_data()
            image_np = np.asarray(bytearray(raw_image_data), dtype="uint8")
            image_np = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))

            print(f"Number of faces detected: {len(faces)}")  # 로그 추가

            for i in range(len(faces)):
                x, y, w, h = faces[i]
                cropped_face = image_np[y:y+h, x:x+w]
                
                gray_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
                cropped = face_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                
                if len(cropped) == 1:
                    file_name = f"{search_term.replace(' ', '_')}_{idx}_{i}.jpg"
                    cv2.imwrite(os.path.join(download_path, file_name), cropped_face)
                    print(f"Cropped face saved: {file_name}")
                    image_data.append({'filename': file_name, 'label': search_term})
                else:
                    print(len(cropped))
        except Exception as e:
            print(f"Failed to process image: {e}")

    if image_data:
        df = pd.DataFrame(image_data)
        df.to_csv(os.path.join(download_path, f'{search_term}.csv'), index=False)
        print(f"Data saved to '{search_term}.csv'.")
    else:
        print("No images were processed for labeling.")

if __name__=='__main__':
    queries = ['강아지상 남자 연예인']
    download_path = os.path.join('.','images')
    os.makedirs(download_path, exist_ok=True)
    
    for query in queries:
        download_subpath = os.path.join(download_path, query)

        os.makedirs(download_subpath, exist_ok=True)
        
        download_and_label_images(query, download_path=download_subpath, number_images=5)
