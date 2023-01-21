import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

from skimage.measure import regionprops
from skimage.measure import label as sk_measure_label
from skimage.transform import warp, SimilarityTransform#, rotate
from scipy.ndimage import rotate
from intelligent_placer_lib.image_processing import get_largest_component_by_num

SHIFT = 10
ANGLE = 10
INDENT = 90

def polygon_area_less_sum_of_items_areas(polygon_props, item_props_list) -> bool:
    if polygon_props.area < sum(x.area for x in item_props_list):
        return True
    return False

def can_be_placed(cur_placement, item_full_mask) -> bool:
    # fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    # ax[0].imshow(item_full_mask)
    no_cur_placement = cv2.bitwise_not(cur_placement)
    # ax[1].imshow(no_cur_placement)
    check = cv2.bitwise_and(no_cur_placement, item_full_mask.astype("uint8"))
    # ax[2].imshow(check)
    if True in check:
        return False
    return True

def move_item(item_full_mask, axis, start):
    item_props = regionprops(get_largest_component_by_num(item_full_mask, 0).astype(np.uint8))[0]
    # print("num of labels: ", len(props))
    item_bbox = item_props.bbox
    # print("left border: ", item_bbox[1], "upper border: ", item_bbox[0])
    # print("right border: ", item_bbox[3], "lower border: ", item_bbox[2])
    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax.imshow(item_full_mask)

    if not start: # пока не начали движение, проверяем, что при повороте ббокс предмета не вышел за границы рабочей области
        if item_bbox[1] < INDENT:
            if item_bbox[3] + INDENT - item_bbox[1] > item_full_mask.shape[1] - INDENT or item_bbox[3] > item_full_mask.shape[1] - INDENT: # ббокс предмета вышел за пределы ббокса многоугольника
                # fig, ax = plt.subplots(figsize=(12, 5))
                # ax.imshow(item_full_mask)
                return False, item_full_mask
            else:
                tform = SimilarityTransform(translation=(-(INDENT - item_bbox[1]), 0))
                item_full_mask = warp(item_full_mask, tform).astype(bool)
                # fig, ax = plt.subplots(figsize=(12, 5))
                # ax.imshow(item_full_mask)
                return True, item_full_mask

        if item_bbox[0] < INDENT:
            if item_bbox[2] + INDENT - item_bbox[0] > item_full_mask.shape[0] - INDENT or item_bbox[2] > item_full_mask.shape[0] - INDENT:
                # fig, ax = plt.subplots(figsize=(12, 5))
                # ax.imshow(item_full_mask)
                return False, item_full_mask
            else:
                tform = SimilarityTransform(translation=(0, -(INDENT - item_bbox[0])))
                item_full_mask = warp(item_full_mask, tform).astype(bool)
                # fig, ax = plt.subplots(figsize=(12, 5))
                # ax.imshow(item_full_mask)
                return True, item_full_mask

    if axis == 1: # двигаем по х
        # если правая граница ббокса выходит за пределы рабочей области
        if item_bbox[3] + SHIFT > item_full_mask.shape[1] - INDENT: #>=
            tform = SimilarityTransform(translation=(item_bbox[1] - INDENT, 0))
            item_full_mask = warp(item_full_mask, tform).astype(bool)
            # fig, ax = plt.subplots(figsize=(12, 5))
            #             # ax.imshow(item_full_mask)

            return False, item_full_mask

        tform = SimilarityTransform(translation=(-SHIFT, 0))
        item_full_mask = warp(item_full_mask, tform).astype(bool)
        # fig, ax = plt.subplots(figsize=(12, 5))
        # ax.imshow(item_full_mask)
        return True, item_full_mask

    if axis == 0: # двигаем по у
        # если нижняя граница ббокса выходит за пределы рабочей области
        if item_bbox[2] + SHIFT > item_full_mask.shape[0] - INDENT: # >=
            tform = SimilarityTransform(translation=(0, item_bbox[0] - INDENT))
            item_full_mask = warp(item_full_mask, tform).astype(bool)
            return False, item_full_mask

        tform = SimilarityTransform(translation=(0, -SHIFT))
        item_full_mask = warp(item_full_mask, tform).astype(bool)
        return True, item_full_mask

def try_to_place(unplaced_items_num, cur_placement, items_list, time):
    if unplaced_items_num == 0:
        return True, cur_placement

    # if (time.time() - time)/60.0 > 5.0:
    #     return False, None

    start_placement = cur_placement
    item = items_list[len(items_list) - unplaced_items_num]
    # создаем вспомогательную маску размером с рабочее поле
    cur_item_full_mask = np.zeros_like(cur_placement, dtype=np.uint8)
    cur_item_full_mask[INDENT:item.shape[0] + INDENT, INDENT:item.shape[1] + INDENT] = item

    y_movable, x_movable = True, True
    for angle in range(0, 360 - ANGLE, ANGLE):
        # print("Текущий угол: ", angle)
        cur_item_full_mask_ = cur_item_full_mask.astype(int)
        cur_item_full_mask = rotate(cur_item_full_mask_, -angle, reshape=False).astype("uint8")
        move_start = False
        move_start, cur_item_full_mask = move_item(cur_item_full_mask, 1, move_start)
        if not move_start: # меняем угол без движения по осям
            continue

        while y_movable:
            while x_movable:
                if can_be_placed(cur_placement, cur_item_full_mask):
                    # fig, ax = plt.subplots(figsize=(12, 5))
                    # ax.imshow(cv2.bitwise_xor(cur_placement, cur_item_full_mask.astype("uint8")))

                    # размещаем предмет в многоугольник
                    cur_placement = cv2.bitwise_xor(cur_placement, cur_item_full_mask.astype("uint8"))
                    # print("Предмет под номером ", len(items_list) - unplaced_items_num, " упакован")
                    # unplaced_items_num = 0
                    unplaced_items_num -= 1
                    answer, cur_placement = try_to_place(unplaced_items_num, cur_placement, items_list, time)
                    if answer == False:
                        cur_placement = start_placement
                        unplaced_items_num += 1
                        # print("Предмет под номером ", len(items_list) - unplaced_items_num, " меняет положение")
                        # continue
                    else:
                        return True, cur_placement
                # else: # продолжаем двигать по х
                x_movable, cur_item_full_mask = move_item(cur_item_full_mask, 1, move_start)
                if not x_movable:
                    x_movable = True
                    break
            y_movable, cur_item_full_mask = move_item(cur_item_full_mask, 0, move_start)
            if not y_movable:
                y_movable = True
                break

    # fig, ax = plt.subplots(figsize=(12, 5))
    # ax.imshow(cur_placement)
    return False, None

def place_items(polygon, items_dict_list):
    item_list_labels = [] # содержит списки компонент(длиной 1) всех масок предметов
    for item in items_dict_list:
        item_list_labels.append(sk_measure_label(item['mask']))

    # для каждого списка находим свойства компонент и выбираем первую, соответствующую предмету
    item_props_list = [regionprops(label)[0] for label in item_list_labels]
    # сортируем по убыванию площадей
    item_props_list = sorted(item_props_list, key=lambda p: p.area, reverse=True,)

    # находим свойства компоненты многоугольника
    polygon_props = regionprops(sk_measure_label(polygon))[0]

    if not polygon_area_less_sum_of_items_areas(polygon_props, item_props_list):
        # находим размеры рабочего поля по максимальным размерам предметов и ббокса многоугольника
        max_item_x = max([item.bbox[2] - item.bbox[0] for item in item_props_list])
        max_item_y = max([item.bbox[3] - item.bbox[1] for item in item_props_list])
        max_field_x = max(max_item_x, polygon_props.bbox[2] - polygon_props.bbox[0])
        max_field_y = max(max_item_y, polygon_props.bbox[3] - polygon_props.bbox[1])

        # print(max_field_x, max_field_y)

        cur_placement = np.zeros([max_field_x + INDENT * 2, max_field_y + INDENT * 2], dtype=np.uint8) # добавить по 2 с каждой стороны и сделать отступ 1 от границы?
        cur_placement[INDENT:polygon_props.image.shape[0] + INDENT, INDENT:polygon_props.image.shape[1] + INDENT] = polygon_props.image.astype(np.uint8)
        # print("placement shape:", cur_placement.shape)

        unplaced_items_num = len(item_props_list)

        time_start = time.time()
        items_list = [item_props_list[len(item_props_list) - i].image for i in range(len(item_props_list), 0, -1)]
        answer, cur_placemen = try_to_place(unplaced_items_num, cur_placement, items_list, time)
        return answer, cur_placemen

    return False, None