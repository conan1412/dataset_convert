# # -*- coding: utf-8 -*-
import json
from collections import defaultdict
import cv2
from tqdm import tqdm

"""hyper parameters"""
json_file_path = '/Users/videopls/Desktop/原始6类及抛洒物数据集/val2017_sixcoco/annotations/val2017_sixcoco535_1019.json'
output_path = '/Users/videopls/Desktop/原始6类及抛洒物数据集/val2017_sixcoco/annotations/test_val.txt'


"""load json file"""
name_box_id = defaultdict(list)
id_name = dict()
with open(json_file_path, encoding='utf-8') as f:
    data = json.load(f)
    annotations = data['annotations']
    images = data['images']

f = open(output_path, 'w')
for image in tqdm(images):
    img_name = image['file_name']
    f.write(img_name)
    height, width = image['height'], image['width']
    for annotation in annotations:
        image_id = annotation['image_id']
        # image_name = images[image_id]['file_name']  # for own coco
        image_name = str(image_id).zfill(12) + '.jpg'  # for official coco
        if img_name == image_name:
            # id = str(ant['image_id'])
            # name = '%s/%s.jpg' % (images_dir_path, id.zfill(12))
            cat = annotation['category_id']

            # for原始coco数据集
            if cat >= 1 and cat <= 11:
                cat = cat - 1
            elif cat >= 13 and cat <= 25:
                cat = cat - 2
            elif cat >= 27 and cat <= 28:
                cat = cat - 3
            elif cat >= 31 and cat <= 44:
                cat = cat - 5
            elif cat >= 46 and cat <= 65:
                cat = cat - 6
            elif cat == 67:
                cat = cat - 7
            elif cat == 70:
                cat = cat - 9
            elif cat >= 72 and cat <= 82:
                cat = cat - 10
            elif cat >= 84 and cat <= 90:
                cat = cat - 11

            info = annotation['bbox']
            xmin = int(info[0])
            ymin = int(info[1])
            w = int(info[2])
            h = int(info[3])
            xmax = xmin + w
            ymax = ymin + h
            # 标注人员可能标注的方向是从右下到左上，所以需要判断，哪个点是左上角
            xmin_tmp, ymin_tmp, xmax_tmp, ymax_tmp = \
                min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax)
            xmin, ymin, xmax, ymax = xmin_tmp, ymin_tmp, xmax_tmp, ymax_tmp
            if (w / width) >= (3 / 4) or (h / height) >= (3 / 4):
                continue
            box_info = " %d,%d,%d,%d,%d" % (
                xmin, ymin, xmax, ymax, cat)
            f.write(box_info)
    f.write('\n')

