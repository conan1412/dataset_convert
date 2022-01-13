'''import json
import os
import cv2

# ------------用os提取images文件夹中的图片名称，并且将BBox都读进去------------
# 根路径，里面包含images(图片文件夹)，annos.txt(bbox标注)，classes.txt(类别标签),
# 以及annotations文件夹(如果没有则会自动创建，用于保存最后的json)
root_path = r'G:\data\cell_phone_samples\correct_images_and_labels\cellphone_labels_cut_person_and_cellphone_total\labels_coco_format'
# 用于创建训练集或验证集
phase = 'train'  # 需要修正

# dataset用于保存所有数据的图片信息和标注信息
dataset = {'categories': [], 'annotations': [], 'images': []}

# # 打开类别标签
# with open(os.path.join(root_path, 'classes.txt')) as f:
#     classes = f.read().strip().split()
#
# # 建立类别标签和数字id的对应关系
# for i, cls in enumerate(classes, 1):
#     dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

dataset['categories'] = [{'supercategory': 'person', 'id': 1, 'name': 'person'},
                            {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'},
                            {'supercategory': 'vehicle', 'id': 3, 'name': 'car'},
                            {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'},
                            {'supercategory': 'vehicle', 'id': 5, 'name': 'bus'},
                            {'supercategory': 'vehicle', 'id': 6, 'name': 'truck'}]

# 读取images文件夹的图片名称
indexes = os.listdir(os.path.join(root_path, 'images'))

# 统计处理图片的数量
global count
count = 0

# 读取Bbox信息
with open(os.path.join(root_path, 'annos.txt')) as tr:
    annos = tr.readlines()

    # ---------------接着将，以上数据转换为COCO所需要的格式---------------
    for k, index in enumerate(indexes):
        count += 1
        # 用opencv读取图片，得到图像的宽和高
        im = cv2.imread(os.path.join(root_path, 'images/') + index)
        height, width, _ = im.shape

        # 添加图像的信息到dataset中
        dataset['images'].append({'file_name': index,
                                  'id': k,
                                  'width': width,
                                  'height': height})

        for ii, anno in enumerate(annos):
            parts = anno.strip().split()

            # 如果图像的名称和标记的名称对上，则添加标记
            if parts[0] == index:
                # 类别
                cls_id = parts[1]
                # x_min
                x1 = float(parts[2])
                # y_min
                y1 = float(parts[3])
                # x_max
                x2 = float(parts[4])
                # y_max
                y2 = float(parts[5])
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                dataset['annotations'].append({
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': int(cls_id),
                    'id': i,
                    'image_id': k,
                    'iscrowd': 0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })

        print('{} images handled'.format(count))

# 保存结果的文件夹
folder = os.path.join(root_path, 'annotations')
if not os.path.exists(folder):
    os.makedirs(folder)
json_name = os.path.join(root_path, 'annotations/{}.json'.format(phase))
with open(json_name, 'w') as f:
    json.dump(dataset, f)'''

# 说明：
'''
1)military_object.names文件内容如下：
label1
label2
label3
2)此训练针对矩形框的标注
3）代码中很多键值都是自己造的，因为训练用不到这些(比如mask的训练)，仅仅只是为了跟coco格式保持一致
'''

import os
import json
import cv2
import random
import time
from tqdm import tqdm

# coco格式最后储存位置
# coco_format_save_path = '/Users/videopls/Desktop/原始6类及抛洒物数据集/val_own+sixcoco.json'
# coco_format_save_path = '/Users/videopls/Desktop/AIWIN轮胎检测/train+test500/train_ori+flip+mosaic.json'
# coco_format_save_path = '/Users/videopls/Desktop/原始6类及抛洒物数据集/7类_211105(head摩托车包括人)/own_images7_train_1105.json'
coco_format_save_path = '/Users/videopls/Desktop/原始6类及抛洒物数据集/7类_211105(head摩托车包括人)/own_images7_val_1105.json'
# coco_format_save_path = '/ai/223/person/huheng/code/detection/CenterNet-HarDNet/data/coco/new_anno/annotations/train_own+sixcoco70082.json'
# coco_format_save_path = '/Users/videopls/Desktop/原始6类及抛洒物数据集/train2017_sixcoco/annotations/train2017_sixcoco.json'
# coco_format_save_path = '/Users/videopls/Desktop/原始6类及抛洒物数据集/val2017_sixcoco/annotations/test_val.json'

# 类别文件，一行一个类
# yolo_format_classes_path = '../datasets/military/annotations/military_object.names'
# yolo格式的注释文件
# yolo_format_annotation_path = '/Users/videopls/Desktop/原始6类及抛洒物数据集/val_own+sixcoco.txt'
# yolo_format_annotation_path = '/Users/videopls/Desktop/AIWIN轮胎检测/train+test500/train_ori+flip+mosaic.txt'
# yolo_format_annotation_path = '/Users/videopls/Desktop/原始6类及抛洒物数据集/7类_211105(head摩托车包括人)/own_train.txt'
yolo_format_annotation_path = '/Users/videopls/Desktop/原始6类及抛洒物数据集/7类_211105(head摩托车包括人)/own_val.txt'
# yolo_format_annotation_path = '/ai/223/person/huheng/code/detection/CenterNet-HarDNet/data/coco/new_anno/annotations/train_own+sixcoco70082.txt'
# yolo_format_annotation_path = '/Users/videopls/Desktop/原始6类及抛洒物数据集/train2017_sixcoco/annotations/train2017_sixcoco12251_1019.txt'
# yolo_format_annotation_path = '/Users/videopls/Desktop/原始6类及抛洒物数据集/val2017_sixcoco/annotations/test_val.txt'

# 前面都造好了基础(为了和coco一致，其实很多都用不到的)，现在开始解析自己的数据
with open(yolo_format_annotation_path, 'r') as f1:
    lines1 = f1.readlines()
# with open(yolo_format_annotation_path2, 'r') as f2:
#     lines2 = f2.readlines()
# with open(yolo_format_annotation_path3, 'r') as f3:
#     lines3 = f3.readlines()
# with open(yolo_format_annotation_path4, 'r') as f4:
#     lines4 = f4.readlines()
# with open(yolo_format_annotation_path5, 'r') as f5:
#     lines5 = f5.readlines()
# with open(yolo_format_annotation_path6, 'r') as f6:
#     lines6 = f6.readlines()

# lines1 = lines1 + lines2
# lines1 = lines1 + lines2 + lines3 + lines4 + lines5
# lines1 = lines1 + lines2 + lines3 + lines4 + lines5 + lines6

# yolo_format_annotation_path2 = '/Users/videopls/Desktop/ownimg_6coco/all_img/add_txt/addimg_train.txt'
# yolo_format_annotation_path2 = '/Users/videopls/Desktop/ownimg_6coco/own_labelme_img_yolo3.txt'
# 根据自己的数据集写类别。举个例子:
# categories_dict = [{'supercategory': 'None', 'id': 1, 'name': 'w3'},{'supercategory': 'None', 'id': 2, 'name': 'h3'}]

# # 我有类别文件,本着能用代码绝不手写的原则
# with open(yolo_format_classes_path, 'r') as f1:
#     lines1 = f1.readlines()
# categories = []
# for j, label in enumerate(lines1):
#     label = label.strip()
#     categories.append({'id': j + 1, 'name': label, 'supercategory': 'None'})
"""========================================后文就正式开始了========================================="""
write_json_context = dict()
# write_json_context['info'] = {'description': '', 'url': '', 'version': '', 'year': 2020, 'contributor': '',
#                               'date_created': '2020-06-16 11:00:08.5'}
# write_json_context['licenses'] = [{'id': 1, 'name': None, 'url': None}]
write_json_context['categories'] = [{'supercategory': 'person', 'id': 1, 'name': 'person'},
                            {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'},
                            {'supercategory': 'vehicle', 'id': 3, 'name': 'car'},
                            {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'},
                            {'supercategory': 'vehicle', 'id': 5, 'name': 'bus'},
                            {'supercategory': 'vehicle', 'id': 6, 'name': 'truck'},
                            {'supercategory': 'vehicle', 'id': 7, 'name': 'head'},
                                    ]
# write_json_context['categories'] = [{'supercategory': 'vehicle', 'id': 1, 'name': 'defect'},]

# 'aeroplane': 1, 'bicycle' : 2, 'bird':3, 'boat':4, 'bottle':5,
#                 'bus' : 6, 'car' : 7, 'cat' : 8, 'chair' : 9, 'cow' : 10
#                 , 'diningtable' : 11, 'dog' : 12, 'horse' : 13, 'motorbike' : 14, 'person' : 15
#                 , 'pottedplant' : 16, 'sheep' : 17, 'sofa' : 18, 'train' : 19, 'tvmonitor' : 20

# write_json_context['categories'] = [{'supercategory': 'vehicle', 'id': 1, 'name': 'aeroplane'},
#                             {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'},
#                             {'supercategory': 'vehicle', 'id': 3, 'name': 'bird'},
#                             {'supercategory': 'vehicle', 'id': 4, 'name': 'boat'},
#                             {'supercategory': 'vehicle', 'id': 5, 'name': 'bottle'},
#                             {'supercategory': 'vehicle', 'id': 6, 'name': 'bus'},
#                             {'supercategory': 'vehicle', 'id': 7, 'name': 'car'},
#                             {'supercategory': 'vehicle', 'id': 8, 'name': 'cat'},
#                             {'supercategory': 'vehicle', 'id': 9, 'name': 'chair'},
#                             {'supercategory': 'vehicle', 'id': 10, 'name': 'cow'},
#                             {'supercategory': 'vehicle', 'id': 11, 'name': 'diningtable'},
#                             {'supercategory': 'vehicle', 'id': 12, 'name': 'dog'},
#                             {'supercategory': 'vehicle', 'id': 13, 'name': 'horse'},
#                             {'supercategory': 'vehicle', 'id': 14, 'name': 'motorbike'},
#                             {'supercategory': 'vehicle', 'id': 15, 'name': 'person'},
#                             {'supercategory': 'vehicle', 'id': 16, 'name': 'pottedplant'},
#                             {'supercategory': 'vehicle', 'id': 17, 'name': 'sheep'},
#                             {'supercategory': 'vehicle', 'id': 18, 'name': 'sofa'},
#                             {'supercategory': 'vehicle', 'id': 19, 'name': 'train'},
#                             {'supercategory': 'vehicle', 'id': 20, 'name': 'tvmonitor'},
#                                     ]


# write_json_context['categories'] = [{'supercategory': 'vehicle', 'id': 1, 'name': 'car'},
#                                     ]

write_json_context['images'] = []
write_json_context['annotations'] = []

# 第n个标注框计数,也就是annotations中的id
num_bboxes = 0
# 每一行就是一张图片的标注信息。
# 其他格式也很好做，这个遍历就是图片文件路径，后面bboxes的遍历就是解析当前图片的标注信息
# 如果xml或者json的,我的csdn中也有labelme标注的xml和json的解析代码(很久很久以前的代码,其实有更简洁的解析,但是我懒得写,哈哈)
# 看懂我的代码,改起来不要太简单
for i, line in enumerate(tqdm(lines1)):

    img_context = {}
    # 我的数据以空格分隔的,具体查看上面的截图
    line = line.split(' ')
    # 我在想:如果这张图没有任何标注,要不要保留呢,我选择保留负样本，当然你也可以打开下面这段代码舍弃
    # if len(line) < 2:
    #     continue
    # 文件名最好不要有空格，这是一种习惯
    img_path = line[0].rstrip()  # for own coco
    # img_path = '/Users/videopls/Desktop/原始6类及抛洒物数据集/val2017_sixcoco/val2017_sixcoco/'+line[0].rstrip()  # for train official coco
    img_name = os.path.basename(img_path)
    # 因为需要width和height,而我得yolo文件里没有，所以我还得读图片，很烦
    # 我就用opencv读取了,当然用其他库也可以
    # 别把图片路径搞错了，绝对路径和相对路径分清楚！
    height, width = cv2.imread(img_path).shape[:2]
    img_context['file_name'] = img_name
    img_context['height'] = height
    img_context['width'] = width
    # img_context['date_captured'] = '2020-06-16 11:00:08.5'



    # 这么多id搞得我头都懵了,我猜这是第几张图序号吧,每行一张图，那当然就是第i张了
    # img_context['id'] = i
    img_context['id'] = i + 700000  # for val
    # img_context['id'] = i + 581922



    # img_context['license'] = 1
    # img_context['coco_url'] = ''
    # img_context['flickr_url'] = ''
    write_json_context['images'].append(img_context)
    # 这个地方如果有标注框继续，没有的话跳过，跟上面的区别在于是否将图片加入到images中
    # 如果images中有这张图但是没有标注信息，那就是负样本，反之亦然
    if len(line) < 2:
        continue
    for bbox in line[1:]:
        bbox_dict = {}
        xmin, ymin, xmax, ymax, class_id = bbox.strip().split(',')
        # 我就有时候int和str不注意各种报错
        xmin, ymin, xmax, ymax, class_id = float(xmin), float(ymin), float(xmax), float(ymax), int(class_id)

        # '''只有6类时，把第7类去掉'''
        # if class_id == 6:
        #     continue
        # '''#############'''

        bbox_dict['id'] = num_bboxes

        # bbox_dict['image_id'] = i
        bbox_dict['image_id'] = i + 700000  # for val
        # bbox_dict['image_id'] = i + 581922





        bbox_dict['category_id'] = class_id + 1
        bbox_dict['iscrowd'] = 0  # 前面有解释
        h, w = abs(ymax - ymin), abs(xmax - xmin)
        bbox_dict['area'] = h * w

        # 标注人员可能标注的方向是从右下到左上，所以需要判断，哪个点是左上角
        xmin_tmp, ymin_tmp, xmax_tmp, ymax_tmp = \
            min(xmin, xmax),min(ymin, ymax),max(xmin, xmax),max(ymin, ymax)
        xmin, ymin, xmax, ymax = xmin_tmp, ymin_tmp, xmax_tmp, ymax_tmp
        # if xmin > xmax:
        #     xmin_tmp, ymin_tmp, xmax_tmp, ymax_tmp = xmin, ymin, xmax, ymax
        #     xmin, ymin, xmax, ymax = xmax_tmp, ymax_tmp, xmin_tmp, ymin_tmp

        bbox_dict['bbox'] = [xmin, ymin, w, h]
        bbox_dict['segmentation'] = [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]]
        write_json_context['annotations'].append(bbox_dict)
        num_bboxes += 1
    # i += 1

# 终于搞定了,保存！
with open(coco_format_save_path, 'w') as fw:
    json.dump(write_json_context, fw)
