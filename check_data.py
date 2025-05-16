from preprocessing.load_data import load_images_from_folder
import numpy as np
import os

# Пути к папкам
train_path = "data/fer2013/train"
test_path = "data/fer2013/test"

# Загружаем
X_train, y_train = load_images_from_folder(train_path)
X_test, y_test = load_images_from_folder(test_path)

# Проверка
print("Формы массивов:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# Сохраняем для дальнейшего использования
os.makedirs("data", exist_ok=True)
np.save("data/X_train.npy", X_train)
np.save("data/y_train.npy", y_train)
np.save("data/X_test.npy", X_test)
np.save("data/y_test.npy", y_test)

print("[✓] Данные сохранены в папку 'data/'")
