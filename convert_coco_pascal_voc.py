# coding: utf-8

r"""Convert raw Microsoft COCO dataset to PASCAL VOC for object_detection.
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
    python ./convert_coco_pascal_voc.py --data_dir /home/merge/py-faster-rcnn/output/mscoco/ \
        --set train \
        --labelmap /home/merge/models/research/object_detection/data/mscoco_label_map.pbtxt \
        --output_filepath=/home/merge/py-faster-rcnn/output/mscoco/pascalvoc
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

import xml.dom.minidom

arguments = [
# flags, dest, type, default, help
['--data_dir', 'data_dir', str, None, 'Root directory to raw Microsoft COCO dataset.'],
['--output_filepath','output_filepath', str, None, 'Path to save PASCAL VOC Data.'],
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


def convert_coco_dection_dataset(imgs_dir, annotations_filepath, labelmap, out_path):
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

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    nb_imgs = len(img_ids)
    for index, img_id in enumerate(img_ids):
        if index % 100 == 0:
            print("Converting images: %d / %d" % (index, nb_imgs))
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

        img_detail['bboxes'] = bboxes
        img_detail['labels'] = labels
        img_detail['folder'] = imgs_dir

        dict_to_coco_example(img_detail, out_path)
        if index > 100: break

    print("Converting is done for %d images." % (nb_imgs,))
    return None


def createTextElement(dom, tagname, val=''):
    node = dom.createElement(tagname)
    nodeV = dom.createTextNode(val if isinstance(val,str) else str(val))
    node.appendChild(nodeV)
    return node


def dict_to_coco_example(img_data, out_path):
    """Convert python dictionary formath data of one image to tf.Example proto.
    Args:
        img_data: infomation of one image, inclue bounding box, labels of bounding box,\
            height, width, encoded pixel data.
    Returns:
        example: The converted tf.Example
    """
    dom = xml.dom.minidom.parse('pascal_voc_template.xml')
    root = dom.documentElement
    bboxes = img_data['bboxes']

    dom.getElementsByTagName('width')[0].childNodes[0].nodeValue = img_data['width']
    dom.getElementsByTagName('height')[0].childNodes[0].nodeValue = img_data['height']
    dom.getElementsByTagName('filename')[0].childNodes[0].nodeValue = img_data['file_name']
    dom.getElementsByTagName('folder')[0].childNodes[0].nodeValue = img_data['folder']
    dom.getElementsByTagName('id')[0].childNodes[0].nodeValue = img_data['id']
    dom.getElementsByTagName('flickr_url')[0].childNodes[0].nodeValue = img_data['flickr_url']
    dom.getElementsByTagName('coco_url')[0].childNodes[0].nodeValue = img_data['coco_url']
    dom.getElementsByTagName('date_captured')[0].childNodes[0].nodeValue = img_data['date_captured']

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        xmin = int(bbox[0]*img_data['width'])
        xmax = int((bbox[0] + bbox[2])*img_data['width'])
        ymin = int(bbox[1]*img_data['height'])
        ymax = int((bbox[1] + bbox[3])*img_data['height'])
        obj = dom.createElement('object')
        obj.appendChild(createTextElement(dom, 'name', img_data['labels'][i]))
        obj.appendChild(createTextElement(dom, 'pose', 'Unspecified'))
        obj.appendChild(createTextElement(dom, 'truncated', '1'))
        obj.appendChild(createTextElement(dom, 'difficult', '0'))
        bndbox = dom.createElement('bndbox')
        bndbox.appendChild(createTextElement(dom, 'xmin', xmin))
        bndbox.appendChild(createTextElement(dom, 'ymin', ymin))
        bndbox.appendChild(createTextElement(dom, 'xmax', xmax))
        bndbox.appendChild(createTextElement(dom, 'ymax', ymax))
        obj.appendChild(bndbox)
        root.appendChild(obj)

    with open(os.path.join(out_path, os.path.splitext(img_data['file_name'])[0]+'.xml'), 'w') as f:
        dom.writexml(f)
    return None


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
    convert_coco_dection_dataset(imgs_dir, annotations_filepath, FLAGS.labelmap, FLAGS.output_filepath)


if __name__ == '__main__':
    args = parse_args()
    main(args)

