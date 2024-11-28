import cv2
import numpy as np
import matplotlib.pyplot as plt
import rasterio

# Завантаження чорно-білого зображення
def load_grayscale_image(file_path):
    with rasterio.open(file_path) as src:
        band1 = src.read(1)  # Читання першого каналу
        return band1

# Завантаження зображень
image_before = load_grayscale_image("T36UYA_20190318T083701_B01.jp2")
image_after = load_grayscale_image("T36UYA_20190825T083601_B01.jp2")

# Перетворення в формат OpenCV
image_before_opencv = np.uint8(image_before)
image_after_opencv = np.uint8(image_after)

# Ініціалізація SIFT
sift = cv2.SIFT_create()

# Пошук ключових точок і дескрипторів
kp_before, desc_before = sift.detectAndCompute(image_before_opencv, None)
kp_after, desc_after = sift.detectAndCompute(image_after_opencv, None)

# Візуалізація ключових точок
img_before_kp = cv2.drawKeypoints(image_before_opencv, kp_before, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_after_kp = cv2.drawKeypoints(image_after_opencv, kp_after, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Відображення зображень з ключовими точками
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.imshow(img_before_kp, cmap='gray')
plt.title("Keypoints Before Deforestation")

plt.subplot(1, 2, 2)
plt.imshow(img_after_kp, cmap='gray')
plt.title("Keypoints After Deforestation")

plt.show()

# Використання Brute Force Matcher для порівняння дескрипторів
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Зіставлення ключових точок
matches = bf.match(desc_before, desc_after)

# Сортуємо матчі за відстанню
matches = sorted(matches, key=lambda x: x.distance)

# Візуалізація співпадінь
img_matches = cv2.drawMatches(image_before_opencv, kp_before, image_after_opencv, kp_after, matches[:50], None, flags=2)

# Відображення співпадінь
plt.figure(figsize=(20, 10))
plt.imshow(img_matches)
plt.title("Keypoint Matches Between Images")
plt.show()

# Кількість відповідних ключових точок
print(f"Number of Matches: {len(matches)}")
