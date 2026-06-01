import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def check_in(c, x, y, R):
    if ((c[0].item() - x.item())**2 + (c[1].item() - y.item())**2) <= R.item()**2:
        return True
    return False

# Параметры для функции HoughCircles
dp = 0.7                # Параметр разрешения аккумуляторного массива
minDist = 13            # Минимальное расстояние между центрами обнаруженных окружностей
param1 = 31             # Порог для метода Кэнни
param2 = 23            # Порог для функции HoughCircles (чем меньше, тем больше ложных кругов)
minRadius = 5        # Минимальный радиус круга
maxRadius = 40          # Максимальный радиус круга

input_im = 'images/'
imgs = [f for f in os.listdir(input_im)]

for ims in imgs:
    img = cv2.imread(input_im+ims)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.medianBlur(gray, 23)



    circles1 = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp, minDist,
                               param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)

    circlesSNC = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp+0.4, minDist+500,
                               param1=param1+10, param2=param2+30,
                               minRadius=minRadius+150, maxRadius=maxRadius+450)


    img_circles = img.copy()

    mainSNCx, mainSNCy = 0, 0
    mainRad = 0
    if circlesSNC is not None:
        circlesSNC = np.uint16(np.around(circlesSNC))
        # print(circlesSNC)
        # for circles in circlesSNC[0, :]:
        #     center = (circles[0], circles[1])
        #     radius = circles[2]
        #     cv2.circle(img_circles, center, radius, (0, 255, 0), 2)
        center = (circlesSNC[0][0][0], circlesSNC[0][0][1])
        radius = circlesSNC[0][0][2]
        mainSNCx = center[0]
        mainSNCy = center[1]
        mainRad = radius
        cv2.circle(img_circles, center, radius, (0, 255, 0), 2)
    else:
        print("Кабель не найден")

    if circles1 is not None:
        # Округление координат центров и радиусов до целых чисел
        circles = np.uint16(np.around(circles1))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            if not (check_in(center, mainSNCx, mainSNCy, mainRad)):
                continue
            radius = circle[2]
            # Рисование контура круга
            cv2.circle(img_circles, center, radius, (0, 255, 0), 2)
            # Рисование центра круга
            cv2.circle(img_circles, center, 2, (0, 0, 255), 3)
    else:
        print("Круги не найдены")

    cv2.imwrite('out_images/'+ims, img_circles)



# # Отображение результатов
# plt.figure(figsize=(15, 10))
# plt.subplot(1, 2, 1)
# plt.title('Размытое изображение')
# plt.imshow(gray_blurred, cmap='gray')
# plt.axis('off')
#
# plt.subplot(1, 2, 2)
# plt.title('Обнаруженные круги (Преобразование Хафа)')
# plt.imshow(cv2.cvtColor(img_circles, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()