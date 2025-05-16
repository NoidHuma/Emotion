import numpy as np
from preprocessing.extract_landmarks import extract_landmarks_batch
import os

# Загрузка изображений
X_train = np.load("data/X_train.npy")
X_test = np.load("data/X_test.npy")

# Признаки
landmarks_train = extract_landmarks_batch(X_train)
landmarks_test = extract_landmarks_batch(X_test)

# Сохранение
np.save("data/landmarks_train.npy", landmarks_train)
np.save("data/landmarks_test.npy", landmarks_test)

print("[✓] Признаки лица сохранены в папку 'data/'")
