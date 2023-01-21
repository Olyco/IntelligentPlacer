import os
from imageio import imread, imsave
from matplotlib import pyplot as plt

from intelligent_placer_lib.preprocessing import preprocess_image
from intelligent_placer_lib.detection import detect_polygon, detect_objects
from intelligent_placer_lib.placer import place_items
TEST_DIR = 'input\\tests'

def check_image(input_name: str) -> bool:
    input_path = os.path.join(TEST_DIR, input_name)
    input_img = imread(input_path)

    sheets_list = preprocess_image(input_img)

    # polygon, items_list = None
    for sheet in sheets_list:
        if sheet['object(s)'] == 'polygon':
            polygon = detect_polygon(sheet['masked'])
        else:
            items_list = detect_objects(sheet)

    placement = None
    answer, placement = place_items(polygon, items_list)
    # print("Предметы могут быть помещены в многоугольник?", answer)

    return answer, placement

if __name__ == '__main__':
    pass