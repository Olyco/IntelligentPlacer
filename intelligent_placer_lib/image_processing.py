import os
import cv2
from imageio import imread
import numpy as np
from imageio import imread, imsave
from matplotlib import pyplot as plt

from skimage.measure import regionprops
from skimage.measure import label as sk_measure_label
from scipy.ndimage import binary_fill_holes

def get_largest_component_by_num(mask, num):
    labels = sk_measure_label(mask) # разбиение маски на компоненты связности
    props = regionprops(labels) # нахождение свойств каждой области
    areas = np.array([prop.area for prop in props]) # площади компонент связности

    if num == 0:
        largest_comp_id = np.array(areas).argmax() # находим номер компоненты с максимальной площадью
        return labels == (largest_comp_id + 1) # области нумеруются с 1, поэтому надо прибавить 1 к индексу

    areas_copy = areas.copy()
    sorted_indexes = np.array([])
    for i in range(areas.size):
        largest_comp_id = np.array(areas_copy).argmax()
        sorted_indexes = np.append(sorted_indexes, largest_comp_id)
        areas_copy[largest_comp_id] = 0 # зануляем площадь наибольшей компоненты во вспомогательном массиве
        if i == num: # если дошли до нужной по величине компоненты, останавливаемся
            break

    return labels == (sorted_indexes[num] + 1)

def get_largest_perimeter_component_by_num(mask, num):
    labels = sk_measure_label(mask) # компоненты связности
    props = regionprops(labels) # свойства каждой области
    perimeters = np.array([prop.perimeter for prop in props]) # площади компонент связности
    #     print(perimeters)
    if num == 0:
        largest_comp_id = np.array(perimeters).argmax() # находим номер компоненты с максимальной площадью
        return labels == (largest_comp_id + 1) # области нумеруются с 1, поэтому надо прибавить 1 к индексу

    perimeters_copy = perimeters.copy()
    sorted_indexes = np.array([])

    for i in range(perimeters.size):
        largest_comp_id = np.array(perimeters_copy).argmax() # номер компоненты с максимальной площадью
        sorted_indexes = np.append(sorted_indexes, largest_comp_id)
        perimeters_copy[largest_comp_id] = 0
        if i == num:
            break

    largest_comp_id = np.array(perimeters).argmax() # находим номер компоненты с максимальной площадью

    return labels == (sorted_indexes[num] + 1) # области нумеруются с 1, поэтому надо прибавить 1 к индексу

def bradley_roth_threshold(image):
    img = np.array(image).astype(float)
    width = image.shape[1]
    height = image.shape[0]

    # Будем разбивать изображение на прямоугольники со стороной s
    s = np.round(width/8)
    s = s + np.mod(s,2) # если s - нечетное:

    # Среднее значение интенсивности в каждом прямоугольнике будем изменять на величину t(%)
    t = 15.0

    # Интегральное изображение
    integral_image = cv2.integral(image)

    # Строим сетку пикселей
    (X,Y) = np.meshgrid(np.arange(width), np.arange(height))
    X = X.ravel()
    Y = Y.ravel()

    # Получаем все координаты соседних фрагментов
    x1 = X - s/2
    x2 = X + s/2
    y1 = Y - s/2
    y2 = Y + s/2

    # Поправляем координаты, вышедшие за границы изображения
    x1[x1 < 0] = 0
    x2[x2 >= width] = width - 1
    y1[y1 < 0] = 0
    y2[y2 >= height] = height - 1

    x1 = x1.astype(int)
    x2 = x2.astype(int)
    y1 = y1.astype(int)
    y2 = y2.astype(int)

    # Находим количество пикселей в каждой области
    count = (x2 - x1) * (y2 - y1)

    # Правый нижний фрагмент
    f1_x = x2
    f1_y = y2

    # Правый верхний
    f2_x = x2
    f2_y = y1 - 1
    f2_y[f2_y < 0] = 0 # если вышли за границу изображения

    # Левый нижний
    f3_x = x1-1
    f3_x[f3_x < 0] = 0 # если вышли за границу изображения
    f3_y = y2

    # Левый верхний
    f4_x = f3_x
    f4_y = f2_y

    # Вычисляем сумму значений яркостей по областям
    sums = integral_image[f1_y, f1_x] - integral_image[f2_y, f2_x] - integral_image[f3_y, f3_x] + integral_image[f4_y, f4_x]

    res = np.ones(height * width, dtype=bool)
    res[img.ravel() * count <= sums * (100.0 - t)/100.0] = False

    res = 255 * np.reshape(res, (height, width)).astype(np.uint8)

    return res

def get_bound_boxes(mask):
    labels = sk_measure_label(mask) # разбиение маски на компоненты связности
    props = regionprops(labels) # нахождение свойств каждой области
    bboxes = [prop.bbox for prop in props] # ограничивающие прямоугольники компонент связности

    return bboxes

def rectangular_crop_image(image, rec_points: tuple, eps):
    upper = rec_points[0]
    lower = rec_points[2]
    left = rec_points[1]
    right = rec_points[3]
    return image[upper - eps:lower + eps, left - eps:right + eps]