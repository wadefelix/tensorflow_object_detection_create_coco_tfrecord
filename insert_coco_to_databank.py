# coding: utf-8

r"""Convert raw Microsoft COCO dataset to our databank.
Attention Please!!!

https://gist.github.com/chicham/6ed3842d0d2014987186 Convert MS COCO Annotation to Pascal VOC format

1)For easy use of this script, Your coco dataset directory struture should like this :
    +Your coco dataset root
        +train2014
        +val2014
        +annotations
            -instances_train2014.json
            -instances_val2014.json
2)To use this script, you should download python coco tools from "http://mscoco.org/dataset/#download" and make it.
After make, copy the pycocotools directory to the directory of this "create_coco_tf_record.py"
or add the pycocotools path to  PYTHONPATH of ~/.bashrc file.

Example usage:
    python ./insert_coco_to_databank.py --data_dir /home/merge/py-faster-rcnn/output/mscoco/ \
        --set train \
        --labelmap /home/merge/models/research/object_detection/data/mscoco_label_map.pbtxt
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pycocotools.coco import COCO
from object_detection.utils import label_map_util

import os, sys
import numpy as np

import logging

import dataset_util
import argparse

import MySQLdb

arguments = [
# flags, dest, type, default, help
['--data_dir', 'data_dir', str, None, 'Root directory to raw Microsoft COCO dataset.'],
['--set', 'set', str, 'train', 'Convert training set or validation set'],
['--labelmap', 'labelmap', str, None, 'label_map.pbtxt path'],
]


def parse_args():
    """
    Parse input arguments
    """
    global arguments
    parser = argparse.ArgumentParser(description='Convert raw Microsoft COCO dataset to GAPT')
    for item in arguments:
        parser.add_argument(item[0], dest=item[1], help=item[4], type=item[2], default=item[3])

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def convert_coco_dection_dataset(imgs_dir, annotations_filepath, labelmap):
    """Load data from dataset by pycocotools. This tools can be download from "http://mscoco.org/dataset/#download"
    Args:
        imgs_dir: directories of coco images
        annotations_filepath: file path of coco annotations file
    Return:
        None
    """
    coco = COCO(annotations_filepath)
    img_ids = coco.getImgIds() # totally 82783 images
    cat_ids = coco.getCatIds() # totally 90 catagories, however, the number of categories is not continuous, \
                               # [0,12,26,29,30,45,66,68,69,71,83] are missing, this is the problem of coco dataset.

    label_map = label_map_util.load_labelmap(labelmap)
    NUM_CLASSES = label_map.ListFields()[0][1][-1].id
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    conn= MySQLdb.connect(
            host='localhost',
            port = 3306,
            user='root',
            passwd='',
            db ='databank',
        read_default_file='/opt/lampp/etc/my.cnf')
    cur = conn.cursor()

    #插入一条数据
    sqli = 'update `databank_80` set `datatype`=%s where `filename`=%s'

    nb_imgs = len(img_ids)
    for index, img_id in enumerate(img_ids):
        if index % 100 == 0:
            print("Converting images: %d / %d" % (index, nb_imgs))
            conn.commit()
        img_info = {}
        bboxes = []
        labels = []

        img_detail = coco.loadImgs(img_id)[0]
        pic_height = img_detail['height']
        pic_width = img_detail['width']

        ann_ids = coco.getAnnIds(imgIds=img_id,catIds=cat_ids)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            bboxes_data = ann['bbox']
            bboxes_data = [bboxes_data[0]/float(pic_width), bboxes_data[1]/float(pic_height),\
                                  bboxes_data[2]/float(pic_width), bboxes_data[3]/float(pic_height)]
                         # the format of coco bounding boxs is [Xmin, Ymin, width, height]
            bboxes.append(bboxes_data)
            labels.append(category_index[ann['category_id']]['name'])

        #img_path = os.path.join(imgs_dir, img_detail['file_name'])
        img_info['height'] = pic_height
        img_info['width'] = pic_width
        img_info['bboxes'] = bboxes
        img_info['labels'] = labels

        example = dict_to_coco_example(img_info)
        cur.execute(sqli, (example, img_detail['file_name']))

    cur.close()
    conn.commit()
    conn.close()
    print("Converting is done for %d images." % (nb_imgs,))
    return None


def dict_to_coco_example(img_data):
    """Convert python dictionary formath data of one image to tf.Example proto.
    Args:
        img_data: infomation of one image, inclue bounding box, labels of bounding box,\
            height, width, encoded pixel data.
    Returns:
        example: The converted tf.Example
    """
    bboxes = img_data['bboxes']
    coords = []
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        xmin = int(bbox[0]*img_data['width'])
        xmax = int((bbox[0] + bbox[2])*img_data['width'])
        ymin = int(bbox[1]*img_data['height'])
        ymax = int((bbox[1] + bbox[3])*img_data['height'])
        category_id = img_data['labels'][i]
        coords.append('[{},{},{},{},{}]'.format(xmin,ymin,xmax,ymax,category_id))

    example = ';'.join(coords)
    return example


def main(FLAGS):
    if FLAGS.set == "train":
        imgs_dir = os.path.join(FLAGS.data_dir, 'train2014')
        annotations_filepath = os.path.join(FLAGS.data_dir,'annotations','instances_train2014.json')
        print("Convert coco train file to tf record")
    elif FLAGS.set == "val":
        imgs_dir = os.path.join(FLAGS.data_dir, 'val2014')
        annotations_filepath = os.path.join(FLAGS.data_dir,'annotations','instances_val2014.json')
        print("Convert coco val file to tf record")
    else:
        raise ValueError("you must either convert train data or val data")
    # converting total coco data
    convert_coco_dection_dataset(imgs_dir, annotations_filepath, FLAGS.labelmap)


if __name__ == '__main__':
    args = parse_args()
    main(args)

