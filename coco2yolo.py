import json
import os
import shutil
import random
import datetime


class COCO2YOLO:
    def __init__(self, json_file, out_dir):
        self._check_file_and_dir(json_file, out_dir)
        self.labels = json.load(open(json_file, 'r', encoding='utf-8'))
        self.coco_id_name_map = self._categories()
        self.coco_name_list = list(self.coco_id_name_map.values())
        print("total images", len(self.labels['images']))
        print("total categories", len(self.labels['categories']))
        print("total labels", len(self.labels['annotations']))

    def _check_file_and_dir(self, file_path, dir_path):
        if not os.path.exists(file_path):
            raise ValueError("file not found")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def _categories(self):
        categories = {}
        for cls in self.labels['categories']:
            categories[cls['id']] = cls['name']
        return categories

    def _load_images_info(self):
        images_info = {}
        for image in self.labels['images']:
            id = image['id']
            file_name = image['file_name']
            if file_name.find('\\') > -1:
                file_name = file_name[file_name.index('\\')+1:]
            w = image['width']
            h = image['height']
            images_info[id] = (file_name, w, h)

        return images_info

    def _bbox_2_yolo(self, bbox, img_w, img_h):
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        centerx = bbox[0] + w / 2
        centery = bbox[1] + h / 2
        dw = 1 / img_w
        dh = 1 / img_h
        centerx *= dw
        w *= dw
        centery *= dh
        h *= dh
        return centerx, centery, w, h

    def _convert_anno(self, images_info):
        anno_dict = dict()
        for anno in self.labels['annotations']:
            bbox = anno['bbox']
            image_id = anno['image_id']
            category_id = anno['category_id']

            image_info = images_info.get(image_id)
            image_name = image_info[0]
            img_w = image_info[1]
            img_h = image_info[2]
            yolo_box = self._bbox_2_yolo(bbox, img_w, img_h)

            anno_info = (image_name, category_id, yolo_box)
            anno_infos = anno_dict.get(image_id)
            if not anno_infos:
                anno_dict[image_id] = [anno_info]
            else:
                anno_infos.append(anno_info)
                anno_dict[image_id] = anno_infos
        return anno_dict

    def save_classes(self):
        sorted_classes = list(map(lambda x: x['name'], sorted(self.labels['categories'], key=lambda x: x['id'])))
        print('coco names', sorted_classes)
        with open('coco.names', 'w', encoding='utf-8') as f:
            for cls in sorted_classes:
                f.write(cls + '\n')
        f.close()

    def coco2yolo(self, out_dir):
        print("loading image info...")
        images_info = self._load_images_info()
        print("loading done, total images", len(images_info))

        print("start converting...")
        anno_dict = self._convert_anno(images_info)
        print("converting done, total labels", len(anno_dict))

        print("saving txt file...")
        self._save_txt(anno_dict, out_dir)
        print("saving done")

    def _save_txt(self, anno_dict, out_dir):
        for k, v in anno_dict.items():
            file_name = os.path.splitext(v[0][0])[0] + ".txt"
            with open(os.path.join(out_dir, file_name), 'w', encoding='utf-8') as f:
                print(k, v)
                for obj in v:
                    cat_name = self.coco_id_name_map.get(obj[1])
                    category_id = self.coco_name_list.index(cat_name)
                    box = ['{:.6f}'.format(x) for x in obj[2]]
                    box = ' '.join(box)
                    line = str(category_id) + ' ' + box
                    f.write(line + '\n')

    def get_categories_size(self):
        return len(self.labels['categories'])

def get_date():
    t_delta = datetime.timedelta(hours=9)
    timezon_jst = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(timezon_jst)
    return now.strftime('%m_%d_%H_%M_%S')

def yolo2ultralytics(timestamp):
    output_yolo_dir = './datasets/'+ timestamp +'_dataset'

    if not os.path.exists(output_yolo_dir):
        os.makedirs(output_yolo_dir)

        os.makedirs(output_yolo_dir + "/images/train")
        os.makedirs(output_yolo_dir + "/images/val")
        os.makedirs(output_yolo_dir + "/labels/train")
        os.makedirs(output_yolo_dir + "/labels/val")

    files = os.listdir(output_dir)
    for file in files:
        label_file_path = os.path.join(output_dir, file)
        image_file_path = os.path.join('./out_image', file.replace(".txt",".jpg"))
        if random.uniform(0, 100) > 20:
            shutil.copy(label_file_path, output_yolo_dir + "/labels/train")
            shutil.copy(image_file_path, output_yolo_dir + "/images/train")
        else:
            shutil.copy(label_file_path, output_yolo_dir + "/labels/val")
            shutil.copy(image_file_path, output_yolo_dir + "/images/val")

def create_data_yaml(categories_size, timestamp):
    output_dir = "./out_yaml/" + timestamp +'_dataset'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        file = open(output_dir + "/data.yaml", 'w')

        file.write("path: " + timestamp +'_dataset' + " \n")
        file.write("train:  images/train \n")
        file.write("val: images/val \n")
        file.write("names: \n")

        for i in range(categories_size):
            file.write("    " + str(i) + ": \n")

    except Exception as e:
        print(e)
    finally:
        file.close()
    
def create_train_pyfile(timestamp):
    try:
        file = open("train_rtdetr_" + timestamp + ".py", 'w')

        file.write("from ultralytics import RTDETR \n")
        file.write("model = RTDETR('rtdetr-l.pt') \n")
        file.write("results = model.train(data='./out_yaml/" + timestamp + "_dataset/data.yaml', epochs=3, imgsz=640, project='./runs/') \n")

    except Exception as e:
        print(e)
    finally:
        file.close()

def clear_ws():
    shutil.rmtree("./out_image")
    shutil.rmtree("./out_yaml")


if __name__ == '__main__':
    create_time = get_date()
    source_dir = "./input_label"
    output_dir = './out_label'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = os.listdir(source_dir)

    cat_size = 0

    for file in files:
        file_path = os.path.join(source_dir, file)
        if ".folder_ml" in file_path:
            continue
        c2y = COCO2YOLO(file_path, output_dir)
        c2y.coco2yolo(output_dir)
        cat_size += c2y.get_categories_size()

    yolo2ultralytics(create_time)
    create_data_yaml(cat_size, create_time)
    create_train_pyfile(create_time)

    clear_ws()