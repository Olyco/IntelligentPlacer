import cv2
import numpy as np
from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage.morphology import binary_opening, remove_small_objects
from scipy.ndimage import binary_fill_holes


from intelligent_placer_lib.image_processing import get_largest_perimeter_component_by_num, bradley_roth_threshold, \
    get_bound_boxes, rectangular_crop_image, get_largest_component_by_num

BOUND_BOX_EPS = 2

def detect_polygon(polygon_sheet):
    # Для получения замкнутого многоугольника применим фильтр
    polygon_sheet = gaussian(polygon_sheet, sigma=0.5, channel_axis=2)

    # Будем искать вторую по периметру компоненту свзяности, потому что первая всегда - лист.
    polygon = get_largest_perimeter_component_by_num(bradley_roth_threshold(rgb2gray(polygon_sheet)), 1)

    # можно заменить на props.image
    bbox = get_bound_boxes(polygon)[0]
    polygon = rectangular_crop_image(polygon, bbox, BOUND_BOX_EPS)

    return polygon # возвращаем обрезанную маску с одной компонентой связности

def detect_objects(items_sheet):
    # Используем алгоритм определения порога Брэдли-Рота
    bradley_roth_res = bradley_roth_threshold(rgb2gray(items_sheet['masked']))
    bradley_roth_res_enclosed = binary_opening(bradley_roth_res, footprint=np.ones((3, 3))) # 3 (5?)

    cropped_table = 1 - items_sheet['mask']
    or_items = cv2.bitwise_or(cropped_table, bradley_roth_res_enclosed.astype(np.uint8))
    items = binary_fill_holes(1 - or_items)
    threshold_res_enclosed = remove_small_objects(binary_opening(items, footprint=np.ones((3, 3))), 100) # 4

    # Находим ограничивающие прямоугольники всех предметов
    bboxes = get_bound_boxes(threshold_res_enclosed)
    #     print(bboxes)

    # Сохраним распознанные предметы и их маски в список
    items_list = list()

    for bbox in bboxes:
        item = dict()
        # Применяем маску и обрезаем по прямоугольнику
        item_cropped = rectangular_crop_image(items_sheet['masked'], bbox, BOUND_BOX_EPS)
        item_mask_cropped = get_largest_component_by_num(rectangular_crop_image(threshold_res_enclosed, bbox, BOUND_BOX_EPS), 0)
        item['mask'] = item_mask_cropped
        item['masked'] = cv2.bitwise_and(item_cropped, item_cropped, mask=item_mask_cropped.astype(np.uint8))
        items_list.append(item)

    return items_list