#!/usr/bin/env python2

from PIL import Image
import argparse
import os
import csv

KEY_PEDESTRIAN = "Pedestrian"
KEY_CYCLIST = "Cyclist"
KEY_CAR = "Car"
KEY_VAN = "Van"
KEY_MISC = "Misc"
KEY_TRUCK = "Truck"
KEY_PERSON_SITTING = "Person_sitting"
KEY_TRAM = "Tram"
KEY_DONT_CARE = "DontCare"

CLAZZ_NUMBERS = {
        KEY_PEDESTRIAN : 0,
        KEY_CYCLIST : 1,
        KEY_CAR : 2
        }
SORTED_KEYS = [KEY_CAR, KEY_PEDESTRIAN, KEY_CYCLIST, KEY_VAN, KEY_TRUCK, KEY_TRAM, KEY_PERSON_SITTING, KEY_MISC, KEY_DONT_CARE]

def resolveClazzNumberOrNone(clazz):
    if clazz == KEY_CYCLIST:
        return CLAZZ_NUMBERS[KEY_CYCLIST]
    #if clazz in (KEY_PEDESTRIAN, KEY_PERSON_SITTING):
    if clazz == KEY_PEDESTRIAN:
        return CLAZZ_NUMBERS[KEY_PEDESTRIAN]
    #if clazz in (KEY_CAR, KEY_VAN):
    if clazz == KEY_CAR:
        return CLAZZ_NUMBERS[KEY_CAR]
    return None

def convertToYoloBBox(bbox, size):
    # Yolo uses bounding bbox coordinates and size relative to the image size.
    # This is taken from https://pjreddie.com/media/files/voc_label.py .
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (bbox[0] + bbox[1]) / 2.0
    y = (bbox[2] + bbox[3]) / 2.0
    w = bbox[1] - bbox[0]
    h = bbox[3] - bbox[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def readRealImageSize(img_path):
    # This loads the whole sample image and returns its size.
    return Image.open(img_path).size

def readFixedImageSize():
    # This is not exact for all images but most (and it should be faster).
    return (1242, 375)

def parseSample(lbl_path, img_dir, in_out_data):
    with open(lbl_path) as csv_file:
        reader = csv.DictReader(csv_file, fieldnames=["type", "truncated", "occluded", "alpha", "bbox2_left", "bbox2_top", "bbox2_right", "bbox2_bottom", "bbox3_height", "bbox3_width", "bbox3_length", "bbox3_x", "bbox3_y", "bbox3_z", "bbox3_yaw", "score"], delimiter=" ")
        sample_labels = []
        for row in reader:
            clazz_number = resolveClazzNumberOrNone(row["type"])
            if clazz_number is not None:
                lbl_file_name = os.path.basename(lbl_path)
                lbl_file_base = os.path.splitext(lbl_file_name)[0]
                img_path = os.path.join(img_dir, "image_2", "{}.png".format(lbl_file_base))
                size = readRealImageSize(img_path) # Use readFixedImageSize() for hard coded size.
                # Image coordinate is in the top left corner.
                bbox = (
                        float(row["bbox2_left"]),
                        float(row["bbox2_right"]),
                        float(row["bbox2_top"]),
                        float(row["bbox2_bottom"])
                       )
                yolo_bbox = convertToYoloBBox(bbox, size)
                # Yolo expects the labels in the form:
                # <object-class> <x> <y> <width> <height>.
                yolo_label = (clazz_number,) + yolo_bbox
                print(yolo_label)
                sample_labels.append(yolo_label)
        in_out_data[lbl_file_base] = sample_labels

def parseArguments():
    parser = argparse.ArgumentParser(description="Generates labels for training darknet on KITTI.")
    parser.add_argument("label_dir", help="data_object_label_2/training/label_2 directory; can be downloaded from KITTI")
    parser.add_argument("image_2_dir", help="data_object_image_2/training directory; can be downloaded from KITTI")
    args = parser.parse_args()

def main():
    parseArguments()

    print("Parsing data...")
    data = {}
    for dir_path, sub_dirs, files in os.walk(args.label_dir):
        for file_name in files:
            if file_name.endswith(".txt"):
                lbl_path = os.path.join(dir_path, file_name)
                parseSample(lbl_path, args.image_2_dir, data)

    print("Writing output files...")

if __name__ == "__main__":
    main()
