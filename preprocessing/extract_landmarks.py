import mediapipe as mp
import numpy as np
import cv2
import tqdm

# Инициализация FaceMesh
mp_face_mesh = mp.solutions.face_mesh

def extract_landmarks_batch(images, image_size=(48, 48)):
    all_landmarks = []

    print(f"[•] Извлечение лицевых признаков из {len(images)} изображений...")

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        for img in tqdm.tqdm(images):
            # Подготовка изображения
            img_uint8 = (img.squeeze() * 255).astype(np.uint8)
            img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)

            # Обработка через FaceMesh
            results = face_mesh.process(img_rgb)

            if results.multi_face_landmarks:
                landmarks = []
                for lm in results.multi_face_landmarks[0].landmark:
                    landmarks.extend([lm.x, lm.y])
                all_landmarks.append(landmarks)
            else:
                # Если лицо не найдено — сохраняем нули
                all_landmarks.append([0.0] * 936)

    return np.array(all_landmarks, dtype=np.float32)
