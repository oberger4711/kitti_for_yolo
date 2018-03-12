# KITTI for YOLO
This tool generates labels in YOLO format from the KITTI labels.
The generated files can be directly used to start a Training on the KITTI data for 2D object detection.

## Setup
Install python dependencies.
If you want to use a virtualenv for this:
```
virtualenv env
```
Then install the packages:
```
pip install -r requirements.txt
```

## Usage
Run help for how to use the script:
```
python kitt_label.py --help
```
After using you may want to exit the virtualenv:
```
deactivate
```
