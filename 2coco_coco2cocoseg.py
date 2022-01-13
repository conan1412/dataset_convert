import json
import os
import cv2
import shutil

def coco2():
    json_file1 = '/Users/videopls/Desktop/ownimg_6coco/train_sixcoco_70000.json'
    json_file2 = '/Users/videopls/Desktop/paosawu_all/all_paosawu.json'

    save_json = '/Users/videopls/Desktop/paosawu_all/all_paosawu+sixcoco7w.json'
    # json_file = 'annotations/instances_val2017.json'
    with open(json_file1) as annos1:
        annotations1 = json.load(annos1)
    with open(json_file2) as annos2:
        annotations2 = json.load(annos2)

    annotations1['images'] += annotations2['images']
    annotations1['annotations'] += annotations2['annotations']
    annotations1['categories'] = annotations2['categories']

    # id = []
    # for image in annotations2['images']:
    #     id.append(image['id'])
    # id.sort()

    with open(save_json, 'w') as annos3:
        json.dump(annotations1, annos3)


def coco2cocoseg():
    # json_ = '/Users/videopls/Desktop/初赛数据集/train/a_annotations'
    # json_ = '/Users/videopls/Desktop/初赛数据集/train/b_annotations'
    # json_ = '/Users/videopls/Desktop/初赛数据集/test/a_annotations'
    json_ = '/Users/videopls/Desktop/初赛数据集/test/b_annotations'
    json_file1 = json_ + '.json'
    save_json = json_ + '_seg.json'
    with open(json_file1, 'r') as annos1:
        annotations1 = json.load(annos1)
    annotations =  annotations1['annotations']
    for i, annotation in enumerate(annotations):
        [xmin, ymin, w, h] = annotation['bbox']
        # h, w = abs(ymax - ymin), abs(xmax - xmin)
        ymax, xmax = ymin+h, xmin+w
        segmentation = [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]]
        annotations[i]['segmentation'] = segmentation
    with open(save_json, 'w') as annos3:
        json.dump(annotations1, annos3)

def cocoaddcoco_idplus():
    json1 = '/Users/videopls/Desktop/初赛数据集/训练用数据集/train/a_annotations_seg.json'
    json2 = '/Users/videopls/Desktop/初赛数据集/训练用数据集/train/b_annotations_seg.json'
    save_json = '/Users/videopls/Desktop/初赛数据集/训练用数据集/train/train_annotations_seg.json'
    with open(json1, 'r') as annos1:
        annotations1 = json.load(annos1)
    with open(json2, 'r') as annos2:
        annotations2 = json.load(annos2)

    anno_imgs1 =  annotations1['images']
    anno_annos1 =  annotations1['annotations']
    anno_cats1 =  annotations1['categories']

    anno_imgs2 = annotations2['images']
    anno_annos2 = annotations2['annotations']
    anno_cats2 = annotations2['categories']

    #  改变1 id号
    for i, img in enumerate(anno_imgs1):
        id = anno_imgs1[i]['id']
        id += 10000
        anno_imgs1[i]['id'] = id
    for j, anno in enumerate(anno_annos1):
        image_id = anno_annos1[j]['image_id']
        image_id += 10000
        anno_annos1[j]['image_id'] = image_id

        id = anno_annos1[j]['id']
        id += 10000
        anno_annos1[j]['id'] = id

    anno_imgs2 += anno_imgs1
    anno_annos2 += anno_annos1
    with open(save_json, 'w') as annos3:
        json.dump({'images':anno_imgs2, 'annotations':anno_annos2, 'categories':anno_cats1}, annos3)


    # print()

def coco_info():
    train_json_a = '/Users/videopls/Desktop/初赛数据集/初赛原数据集/train/a_annotations.json'
    train_json_b = '/Users/videopls/Desktop/初赛数据集/初赛原数据集/train/b_annotations.json'
    test_json_b = '/Users/videopls/Desktop/初赛数据集/初赛原数据集/test/b_annotations.json'

    with open(train_json_a) as annos1:
        annotations1 = json.load(annos1)
    with open(train_json_b) as annos2:
        annotations2 = json.load(annos2)
    with open(test_json_b) as annos2:
        annotations3 = json.load(annos2)

    dict_label_1 = {}
    annotations1_1 = annotations1['annotations']
    for i in annotations1_1:
        if i['category_id'] not in dict_label_1:
            dict_label_1[i['category_id']] = 0
        dict_label_1[i['category_id']] += 1

    dict_label_2 = {}
    annotations2_1 = annotations2['annotations']
    for i in annotations2_1:
        if i['category_id'] not in dict_label_2:
            dict_label_2[i['category_id']] = 0
        dict_label_2[i['category_id']] += 1

    dict_label_3 = {}
    annotations3_1 = annotations3['annotations']
    for i in annotations3_1:
        if i['category_id'] not in dict_label_3:
            dict_label_3[i['category_id']] = 0
        dict_label_3[i['category_id']] += 1

    print(f"len_train_a = {len(dict_label_1)}, len_train_b = {len(dict_label_2)}, len_test_b = {len(dict_label_3)}")

    # dict_label_1 = sorted(dict_label_1.iteritems(), key=lambda x: x[0])
    # dict_label_2 = sorted(dict_label_2.iteritems(), key=lambda x: x[0])
    # dict_label_3 = sorted(dict_label_3.iteritems(), key=lambda x: x[0])

    miss_train_label_a = []
    for i in range(116):
        if i not in dict_label_1:
            miss_train_label_a.append(i)
    miss_train_label_b = []
    for i in range(116):
        if i not in dict_label_2:
            miss_train_label_b.append(i)
    miss_train_test_b = []
    for i in range(116):
        if i not in dict_label_3:
            miss_train_test_b.append(i)

    with open('/Users/videopls/Desktop/初赛数据集/初赛原数据集/dict_label.json', 'w') as xx:
        json.dump({'len_train_a': len(dict_label_1),
                   'miss_train_label_a': miss_train_label_a,
                    'train_json_a': dict_label_1,
                   'len_train_b': len(dict_label_2),
                   'miss_train_label_b': miss_train_label_b,
                   'train_json_b': dict_label_2,
                   'len_test_b': len(dict_label_3),
                   'miss_train_test_b': miss_train_test_b,
                   'test_json_b': dict_label_3}, xx, indent=4)





    print()
if __name__ == '__main__':
    # coco2cocoseg()
    # cocoaddcoco_idplus()
    coco_info()
