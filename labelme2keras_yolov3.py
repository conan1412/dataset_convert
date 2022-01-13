from os import getcwd
import os
import json
import glob
wd = getcwd()
"labelme标注的json 数据集转为keras 版yolov3的训练集"
# classes = ["car","motorcycle", "person", "bus", "truck"]
# classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'head']
# classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'box']
# classes = ['car']


# root_path = "/Users/videopls/Desktop/原始6类及抛洒物数据集/"
root_path = "/Users/videopls/Desktop/原始6类及抛洒物数据集/7类_211105(head摩托车包括人)/"
img_path = root_path + "own_images/"
# img_path = "/Users/videopls/Desktop/paosawu_all/ori_allimg/"

# image_ids = glob.glob(r"/Users/videopls/Desktop/测试/own_dataset/voc/images/*jpg")
# image_ids = glob.glob(r"/Users/videopls/Desktop/标注已完成/images/*jpg")
# image_ids = glob.glob(r"/Users/videopls/Desktop/ownimg_6coco/own_images/*jpg")
# image_ids = glob.glob(r"/Users/videopls/Desktop/paosawu_all/ori_allimg/*jpg")
image_ids = glob.glob(img_path+"*jpg")
print(image_ids)
json_ids = glob.glob(img_path+'*json')
# list_file = open('/Users/videopls/Desktop/ownimg_6coco/own_images_4-9.txt', 'w')
# list_file = open(root_path+ 'own_images_1018.txt', 'w')
list_file = open(root_path+ 'own_images7_1105.txt', 'w')

def convert_annotation(image_id, list_file):
     if os.path.exists('%s.json' % (image_id)):
          jsonfile=open('%s.json' % (image_id))
          in_file = json.load(jsonfile)

          # if os.path.basename(image_id) in ['296']:
          #      print()

          for i in range(0,len(in_file["shapes"])):
               object=in_file["shapes"][i]
               cls=object["label"]
               points=object["points"]
               xmin=int(points[0][0])
               ymin=int(points[0][1])
               xmax=int(points[1][0])
               ymax=int(points[1][1])
               # 标注人员可能标注的方向是从右下到左上，所以需要判断，哪个点是左上角
               xmin_tmp, ymin_tmp, xmax_tmp, ymax_tmp = \
                    min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax)
               xmin, ymin, xmax, ymax = xmin_tmp, ymin_tmp, xmax_tmp, ymax_tmp
               if cls not in classes:
                    print("cls not in classes")
                    continue
               cls_id = classes.index(cls)
               b = (xmin, ymin, xmax, ymax)
               list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
          jsonfile.close()
# for image_id in image_ids:
for json_id in json_ids:
     image_id = json_id.replace('json', 'jpg')
     list_file.write('%s%s' % (img_path, os.path.basename(image_id)))
     convert_annotation(image_id.split('.')[0], list_file)
     list_file.write('\n')
list_file.close()
