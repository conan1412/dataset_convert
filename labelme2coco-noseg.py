import os, sys
import json
import cv2
import base64
import numpy as np
import glob
import PIL
from PIL import Image, ImageDraw


def base64_cv2(base64_str):
    """
    base64转cv2
    """
    imgString = base64.b64decode(base64_str)
    nparr = np.fromstring(imgString, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class labelme2coco(object):
    def __init__(self, labelme_json=[], save_json_path="./tran.json"):
        """
        :param labelme_json: 所有labelme的json文件路径组成的列表
        :param save_json_path: json保存位置
        """
        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.images = []
        # self.categories = []
        self.categories = [{'supercategory': 'Cancer', 'id': 1, 'name': 'car'},
                            {'supercategory': 'Cancer', 'id': 2, 'name': 'bus'},
                            {'supercategory': 'Cancer', 'id': 3, 'name': 'truck'},
                            {'supercategory': 'Cancer', 'id': 4, 'name': 'person'},
                           {'supercategory': 'Cancer', 'id': 5, 'name': 'motorcycle'} ]
        self.annotations = []
        # self.data_coco = {}
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0

        self.save_json()

    def data_transfer(self):
        for num, json_file in enumerate(self.labelme_json):
            with open(json_file, "r") as fp:
                data = json.load(fp)  # 加载json文件
                self.images.append(self.image(data, num))
                for shapes in data["shapes"]:
                    label = shapes["label"]
                    # if label not in self.label:
                    #     self.categories.append(self.categorie(label))
                    #     self.label.append(label)



                    points = shapes["points"]  # 这里的point是用rectangle标注得到的，只有两个点，需要转成四个点
                    points.append([points[0][0],points[1][1]])
                    points.append([points[1][0],points[0][1]])
                    self.annotations.append(self.annotation(points, label, num))
                    self.annID += 1

    def image(self, data, num):
        image = {}
        img = base64_cv2(data["imageData"])  # 解析原图片数据
        filename = data["imagePath"].split('\\')[-1]
        # filename  根据labelme json的imagePath来获取
        image["file_name"] = filename
        cv2.imwrite(f"train/{filename}", img)

        height, width = img.shape[:2]
        image["height"] = height
        image["width"] = width
        image["id"] = num + 1

        self.height = height
        self.width = width

        return image

    # def categorie(self, label):
    #     categorie = {}
    #     categorie["supercategory"] = "Cancer"
    #     categorie["id"] = len(self.label) + 1  # 0 默认为背景
    #     categorie["name"] = label
    #     return categorie

    def annotation(self, points, label, num):
        annotation = {}
        annotation["segmentation"] = []
        annotation["iscrowd"] = 0
        annotation["image_id"] = num + 1
        annotation["bbox"] = list(map(float, self.getbbox(points)))
        annotation["area"] = annotation["bbox"][2] * annotation["bbox"][3]
        # annotation['category_id'] = self.getcatid(label)
        annotation["category_id"] = self.getcatid(label)  # 注意，源代码默认为1
        annotation["id"] = self.annID
        return annotation

    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie["name"]:
                return categorie["id"]
        return 1

    def getbbox(self, points):
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        # print()
        return self.mask2box(mask)

    def mask2box(self, mask):
        """从mask反算出其边框
        mask：[h,w]  0、1组成的图片
        1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
        """
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        # 解析左上角行列号
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)
        # print()
        return [
            left_top_c,
            left_top_r,
            right_bottom_c - left_top_c,
            right_bottom_r - left_top_r,
        ]  # [x1,y1,w,h] 对应COCO的bbox格式

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco = {}
        data_coco["images"] = self.images
        data_coco["categories"] = self.categories
        data_coco["annotations"] = self.annotations
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        # 保存json文件
        json.dump(
            self.data_coco, open(self.save_json_path, "w"), indent=4, cls=MyEncoder
        )  # indent=4 更加美观显示


if __name__ == '__main__':
    labelme_json = glob.glob("/Users/videopls/Desktop/own_dataset/orign_ann/train/*.json")
    # labelme_json = glob.glob("/Users/videopls/Desktop/own_dataset/image/test/*.json")
    labelme2coco(labelme_json, "/Users/videopls/Desktop/own_dataset/voc/annotations/train.json")
    # labelme2coco(labelme_json, "/Users/videopls/Desktop/own_dataset/test.json")
    print(f"*************** labelme2coco done ***************")