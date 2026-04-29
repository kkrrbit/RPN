import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображения
img = cv2.imread('images/3.jpg')

# Преобразование в градации серого
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Размытие изображения для уменьшения шума
gray_blurred = cv2.medianBlur(gray, 23)

# Параметры для функции HoughCircles
dp = 0.7                # Параметр разрешения аккумуляторного массива
minDist = 13            # Минимальное расстояние между центрами обнаруженных окружностей
param1 = 31             # Порог для метода Кэнни
param2 = 23            # Порог для функции HoughCircles (чем меньше, тем больше ложных кругов)
minRadius = 5        # Минимальный радиус круга
maxRadius = 40          # Максимальный радиус круга

# Обнаружение кругов
circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp, minDist,
                           param1=param1, param2=param2,
                           minRadius=minRadius, maxRadius=maxRadius)
# Создание копии изображения для отображения результатов
img_circles = img.copy()

# Проверка, найдены ли круги
if circles is not None:
    # Округление координат центров и радиусов до целых чисел
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        center = (circle[0], circle[1])  # Координаты центра
        radius = circle[2]              # Радиус
        # Рисование контура круга
        cv2.circle(img_circles, center, radius, (0, 255, 0), 2)
        # Рисование центра круга
        cv2.circle(img_circles, center, 2, (0, 0, 255), 3)
else:
    print("Круги не найдены")

# Отображение результатов
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
plt.title('Размытое изображение')
plt.imshow(gray_blurred, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Обнаруженные круги (Преобразование Хафа)')
plt.imshow(cv2.cvtColor(img_circles, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()