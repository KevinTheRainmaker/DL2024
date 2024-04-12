from google_images_search import GoogleImagesSearch
from PIL import Image
import os
import cv2
import numpy as np
import pandas as pd

gis = GoogleImagesSearch('AIzaSyBhdR13ROTgI9IVUucO3aDOABRWvvi5ImE', 'b4bac6bfbdc2d4828')



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
    
    for index, image in enumerate(gis.results()):
        try:
            raw_image_data = image.get_raw_data()
            image_np = np.asarray(bytearray(raw_image_data), dtype="uint8")
            image_np = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

            print(f"Number of faces detected: {len(faces)}")  # 로그 추가

            if len(faces) == 1:
                x, y, w, h = faces[0]
                cropped_face = image_np[y:y+h, x:x+w]
                if cropped_face.size > 0:
                    file_name = f"{search_term.replace(' ', '_')}_{index}.jpg"
                    cv2.imwrite(os.path.join(download_path, file_name), cropped_face)
                    print(f"Cropped face saved: {file_name}")
                    image_data.append({'filename': file_name, 'label': search_term})
            else:
                print("No single front face found or multiple faces detected.")
        except Exception as e:
            print(f"Failed to process image: {e}")

    if image_data:
        df = pd.DataFrame(image_data)
        df.to_csv(os.path.join(download_path, 'labeled_images.csv'), index=False)
        print("Data saved to 'labeled_images.csv'.")
    else:
        print("No images were processed for labeling.")

if __name__=='__main__':
    queries = ["여우상 여자 연예인", '여우상 남자 연예인']
    for query in queries:
        download_path = os.path.join('.','images')
        download_path = os.path.join(download_path, query)

        os.makedirs(download_path, exist_ok=True)
        
        download_and_label_images(query, number_images=3, download_path=download_path)
