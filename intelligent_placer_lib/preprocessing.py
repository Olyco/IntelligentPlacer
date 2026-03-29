import os
import cv2
from imageio import imread
import numpy as np
from imageio import imread, imsave
from matplotlib import pyplot as plt

from skimage.color import rgb2gray
from skimage.morphology import binary_opening, binary_erosion, binary_dilation, binary_closing, remove_small_objects
from skimage.filters import threshold_local, try_all_threshold, threshold_otsu, gaussian
from scipy.ndimage import binary_fill_holes

from intelligent_placer_lib.image_processing import bradley_roth_threshold, get_largest_component_by_num, \
    get_bound_boxes, rectangular_crop_image

TEMPLATE_DIR = 'input\objects'
TEMPLATE_MASK_DIR = 'input\objects\masks'
TEST_DIR = 'input\\tests'

COMPRESSION_COEF = 0.8

def compress_image(image: np.array) -> np.array:
    return cv2.resize(image, [int(image.shape[1] * (1 - COMPRESSION_COEF)),\
                              int(image.shape[0] * (1 - COMPRESSION_COEF))], cv2.INTER_AREA)

def get_template_mask(template_path: str):
    image_path = os.path.join(TEMPLATE_DIR, template_path)
    image = imread(image_path)

    # Сожмем изображение для ускорения работы с ним
    compressed_image = compress_image(image)
    filtered_image = gaussian(compressed_image, sigma=1, channel_axis=2)
    gray_image = rgb2gray(filtered_image)

    # Используем алгоритм определения порога Брэдли-Рота
    bradley_roth_res = cv2.bitwise_not(bradley_roth_threshold(gray_image))

    bradley_roth_res_enclosed_inside = binary_erosion(binary_dilation(bradley_roth_res))
    bradley_roth_res_enclosed = binary_opening(bradley_roth_res_enclosed_inside, footprint=np.ones((5, 5)))

    if template_path.startswith('9'):
        object_mask = get_largest_component_by_num(bradley_roth_res_enclosed, 1) # убираем лишнее вне предмета
    else:
        object_mask = get_largest_component_by_num(bradley_roth_res_enclosed, 0) # убираем лишнее вне предмета
    result_mask = binary_fill_holes(object_mask) # убираем лишнее внутри предмета

    masked_image = cv2.bitwise_and(compressed_image, compressed_image, mask=result_mask.astype("uint8"))

    # Находим ограничивающий прямоугольник
    bound_box = get_bound_boxes(result_mask)[0]

    # Обрезаем шаблонное изображение и маску
    cropped_mask = rectangular_crop_image(result_mask, bound_box, 0)
    cropped_masked_image = rectangular_crop_image(masked_image, bound_box, 0)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cropped_masked_image, cmap='gray')

    # Сохранение полученного изображения в файл
    # binarization_result = os.path.join(TEMPLATE_MASK_DIR, "1_1_mask.png")
    # imsave(TEMPLATE_MASK_DIR + '\\mask_' + template_path, cropped_mask.astype(np.uint8))
    # imsave(TEMPLATE_MASK_DIR + '\\masked_' + template_path, cropped_masked_image.astype(np.uint8))

    return cropped_mask, cropped_masked_image


def preprocess_image(image):
    '''
    Возвращает список словарей
    '''
    compressed_image = compress_image(image)
    filtered_image = gaussian(compressed_image, sigma=1, channel_axis=2)
    gray_image = rgb2gray(filtered_image)

    mean_input = gray_image >= threshold_local(gray_image, 63, method='mean')
    threshold_res_enclosed = binary_opening(mean_input, footprint=np.ones((7, 7)))

    # Используя модифицированную функцию, выбираем две наибольшие по площади компоненты связности
    first_sheet = dict()
    second_sheet = dict()

    sheets_masks = [get_largest_component_by_num(threshold_res_enclosed, 0),
                    get_largest_component_by_num(threshold_res_enclosed, 1)]

    masks_bound_boxes = [get_bound_boxes(sheets_masks[0])[0], get_bound_boxes(sheets_masks[1])[0]]

    sheets_by_masks = []
    for i in range(2):
        sheets_masks[i] = binary_fill_holes(sheets_masks[i]).astype("uint8")
        sheets_by_masks.append(cv2.bitwise_and(compressed_image,compressed_image, mask=sheets_masks[i]))

        # Обрезаем маску и полученное по ней изображение листа
        sheets_masks[i] = rectangular_crop_image(sheets_masks[i], masks_bound_boxes[i], 0)
        sheets_by_masks[i] = rectangular_crop_image(sheets_by_masks[i], masks_bound_boxes[i], 0)

    first_sheet['mask'] = sheets_masks[0]
    second_sheet['mask'] = sheets_masks[1]

    first_sheet['masked'] = sheets_by_masks[0]
    second_sheet['masked'] = sheets_by_masks[1]

    # Будем сравнивать среднеквадратическое отклонение
    std = np.array([])
    for i in range(2):
        # Используем алгоритм определения порога Брэдли-Рота для нахождения объекта/объектов
        bradley_roth_res = cv2.bitwise_not(bradley_roth_threshold(rgb2gray(sheets_by_masks[i])))
        bradley_roth_res_enclosed = binary_opening(bradley_roth_res, footprint=np.ones((1, 1)))

        input_mask = (1 - bradley_roth_res_enclosed).astype("uint8")
        input_by_mask = rgb2gray(cv2.bitwise_and(sheets_by_masks[i], sheets_by_masks[i],mask = input_mask))

        std = np.append(std, np.std(input_by_mask.flatten()))
        # ax[i].imshow(sheets_by_masks[i], cmap='gray')

    if (std[0] > std[1]):
        first_sheet['object(s)'] = 'items'
        second_sheet['object(s)'] = 'polygon'
    else:
        first_sheet['object(s)'] = 'polygon'
        second_sheet['object(s)'] = 'items'

    return [first_sheet, second_sheet]

def preprocess_all_images(image):
    '''
    Возвращает список словарей
    '''
    input_path = os.path.join(TEST_DIR, image)
    input_img = imread(input_path)

    compressed_image = compress_image(input_img)
    filtered_image = gaussian(compressed_image, sigma=1, channel_axis=2)
    gray_image = rgb2gray(filtered_image)


    mean_input = gray_image >= threshold_local(gray_image, 63, method='mean')
    threshold_res_enclosed = binary_opening(mean_input, footprint=np.ones((7, 7)))

    # Используя модифицированную функцию, выбираем две наибольшие по площади компоненты связности
    first_sheet = dict()
    second_sheet = dict()

    sheets_masks = [get_largest_component_by_num(threshold_res_enclosed, 0),
                    get_largest_component_by_num(threshold_res_enclosed, 1)]

    masks_bound_boxes = [get_bound_boxes(sheets_masks[0])[0], get_bound_boxes(sheets_masks[1])[0]]

    sheets_by_masks = []
    for i in range(2):
        sheets_masks[i] = binary_fill_holes(sheets_masks[i]).astype("uint8")
        sheets_by_masks.append(cv2.bitwise_and(compressed_image,compressed_image, mask=sheets_masks[i]))

        # Обрезаем маску и полученное по ней изображение листа
        sheets_masks[i] = rectangular_crop_image(sheets_masks[i], masks_bound_boxes[i], 0)
        sheets_by_masks[i] = rectangular_crop_image(sheets_by_masks[i], masks_bound_boxes[i], 0)

    # fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    # ax[0].imshow(sheets_masks[0], cmap='gray')
    # ax[1].imshow(sheets_masks[1], cmap='gray')

    first_sheet['mask'] = sheets_masks[0]
    second_sheet['mask'] = sheets_masks[1]

    first_sheet['masked'] = sheets_by_masks[0]
    second_sheet['masked'] = sheets_by_masks[1]

    # Будем сравнивать среднеквадратическое отклонение
    std = np.array([])
    for i in range(2):
        # Используем алгоритм определения порога Брэдли-Рота для нахождения объекта/объектов
        bradley_roth_res = cv2.bitwise_not(bradley_roth_threshold(rgb2gray(sheets_by_masks[i])))
        bradley_roth_res_enclosed = binary_opening(bradley_roth_res, footprint=np.ones((1, 1)))

        input_mask = (1 - bradley_roth_res_enclosed).astype("uint8")
        input_by_mask = rgb2gray(cv2.bitwise_and(sheets_by_masks[i], sheets_by_masks[i],mask = input_mask))

        std = np.append(std, np.std(input_by_mask.flatten()))
        # ax[i].imshow(sheets_by_masks[i], cmap='gray')

    if (std[0] > std[1]):
        first_sheet['object(s)'] = 'items'
        second_sheet['object(s)'] = 'polygon'
    else:
        first_sheet['object(s)'] = 'polygon'
        second_sheet['object(s)'] = 'items'

    fig1, ax1 = plt.subplots(1, 2, figsize=(10, 10))
    ax1[0].imshow(first_sheet['masked'])
    ax1[0].set_title(first_sheet['object(s)'] + " " + image[:-5])
    ax1[1].imshow(second_sheet['masked'])
    ax1[1].set_title(second_sheet['object(s)'] + " " + image[:-5])

    return [first_sheet, second_sheet]

