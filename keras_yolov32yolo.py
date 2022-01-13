import os
import json
import cv2
import random
import time

yolo_format_annotation_path = '/Users/videopls/Desktop/paosawu_all/add_sixcoco+ownimg+paosawu.txt'
yolotxt_save_path = '/Users/videopls/Desktop/paosawu_all/add_yolotxt_sixcoco+ownimg+paosawu/'
with open(yolo_format_annotation_path, 'r') as f2:
    lines2 = f2.readlines()

for i, line in enumerate(lines2):
    line = line.split(' ')
    name = os.path.basename(line[0])[:-4]

    img_path = line[0].rstrip()
    height, width = cv2.imread(img_path).shape[:2]
    with open(yolotxt_save_path + "%s.txt" % name,"w" ) as ff:
        for box in line[1:]:
            box = box.split(',')
            xmin, ymin, xmax, ymax = float(box[0]), float(box[1]), float(box[2]), float(box[3]),
            x, y, w, h = (xmax+xmin)/(2*width), (ymax+ymin)/(2*height), (xmax-xmin)/width, (ymax-ymin)/height
            ff.write(box[4].rstrip() + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n')


