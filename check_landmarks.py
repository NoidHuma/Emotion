import numpy as np
import cv2
import matplotlib.pyplot as plt

# Загружаем 1 изображение и его лендмарки
X = np.load("data/X_train.npy")        # форма: (N, 48, 48, 1)
landmarks = np.load("data/landmarks_train.npy")  # форма: (N, 936)

# Выбираем случайное или определённое изображение
index = 10470  # можно выбрать любой
img = X[index].squeeze()  # (48, 48)
landmark_vec = landmarks[index]  # (936,)

# Преобразуем в пары (x, y)
points = np.array(landmark_vec).reshape(-1, 2)
points_pixel = (points * 48).astype(int)  # приведение к пикселям

# Отображаем изображение и точки
img_color = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

for (x, y) in points_pixel:
    if 0 <= x < 48 and 0 <= y < 48:
        cv2.circle(img_color, (x, y), radius=1, color=(0, 255, 0), thickness=-1)

# Показываем с помощью matplotlib
plt.figure(figsize=(3, 3))
plt.imshow(img_color)
plt.title(f"Лицевые признаки для изображения {index}")
plt.axis("off")
plt.show()
