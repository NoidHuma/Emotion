import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

# Эмоции и их индексы
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
label_to_index = {label: idx for idx, label in enumerate(emotion_labels)}


def load_images_from_folder(folder_path, image_size=(48, 48)):
    images = []
    labels = []

    print(f"Загрузка изображений из: {folder_path}")

    for label in emotion_labels:
        class_folder = os.path.join(folder_path, label)
        if not os.path.exists(class_folder):
            print(f"[!] Пропущено: {class_folder} не существует")
            continue

        label_idx = label_to_index[label]
        files = os.listdir(class_folder)
        print(f" -> Класс '{label}': {len(files)} изображений")

        for filename in files:
            img_path = os.path.join(class_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, image_size)
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=-1)

            images.append(img)
            labels.append(label_idx)

    X = np.array(images)
    y = to_categorical(labels, num_classes=7)
    print(f"[✓] Загружено: {X.shape[0]} изображений\n")
    return X, y
