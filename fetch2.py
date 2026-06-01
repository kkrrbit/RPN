import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузим изображение
img = cv2.imread('images/1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Применим пороговую обработку
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Найдем контуры
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Скопируем изображение для отрисовки контуров
img_contours = img.copy()

# Нарисуем контуры
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)

# Отобразим результаты
plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Исходное изображение'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
plt.title('Контуры'), plt.xticks([]), plt.yticks([])
plt.show()