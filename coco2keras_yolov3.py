# # -*- coding: utf-8 -*-
import json
from collections import defaultdict
import cv2
from tqdm import tqdm

"""hyper parameters"""
# json_file_path = '/Users/videopls/Desktop/原始6类及抛洒物数据集/train2017_sixcoco/annotations/train2017_sixcoco12251_1019.json'
# images_dir_path = '/Users/videopls/Desktop/原始6类及抛洒物数据集/train2017_sixcoco/train2017_sixcoco'
# output_path = '/Users/videopls/Desktop/原始6类及抛洒物数据集/train2017_sixcoco/annotations/train2017_sixcoco12251_1019.txt'

json_file_path = '/ai/223/person/huheng/code/detection/CenterNet-HarDNet/data/coco/new_anno/annotations/train2017_sixcoco70082_1019.json'
images_dir_path = '/ai/223/person/huheng/code/detection/CenterNet-HarDNet/data/coco/train2017'
output_path = '/ai/223/person/huheng/code/detection/CenterNet-HarDNet/data/coco/new_anno/annotations/train2017_sixcoco70082_1019.txt'

# json_file_path = '/Users/videopls/Desktop/工业检测/初赛数据/train.json'
# json_file_path = '/Users/videopls/Desktop/原始6类及抛洒物数据集/val2017_sixcoco/annotations/val2017_sixcoco535_1019.json'
# images_dir_path = '/Users/videopls/Desktop/原始6类及抛洒物数据集/val2017_sixcoco/val2017_sixcoco'
# output_path = '/Users/videopls/Desktop/原始6类及抛洒物数据集/val2017_sixcoco/annotations/val2017_sixcoco.txt'

"""load json file"""
name_box_id = defaultdict(list)
id_name = dict()
with open(json_file_path, encoding='utf-8') as f:
    data = json.load(f)
    annotations = data['annotations']


for ant in annotations:

    id = str(ant['image_id'])
    name = '%s/%s.jpg' % (images_dir_path, id.zfill(12))
    cat = ant['category_id']

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

    name_box_id[name].append([ant['bbox'], cat])

"""write to txt"""
with open(output_path, 'w') as f:
    for key in tqdm(name_box_id.keys()):
        height, width, _ = cv2.imread(key).shape
        f.write(key)
        box_infos = name_box_id[key]
        for info in box_infos:
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            w = int(info[0][2])
            h = int(info[0][3])
            x_max = x_min + w
            y_max = y_min + h
            if (w/width) >= (3/4) or (h/height) >= (3/4):
                continue
            box_info = " %d,%d,%d,%d,%d" % (
                x_min, y_min, x_max, y_max, int(info[1]))
            f.write(box_info)
        f.write('\n')
