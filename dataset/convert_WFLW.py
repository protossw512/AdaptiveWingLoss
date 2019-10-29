import sys
sys.path.insert(0, "../utils/")
import numpy as np
import os
import glob
import scipy.io as sio
import cv2
from skimage import io
from utils import cv_crop
import torch
from joblib import Parallel, delayed

def transform(point, center, scale, resolution, rotation=0, invert=False):
    _pt = np.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = np.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if rotation != 0:
        rotation = -rotation
        r = np.eye(3)
        ang = rotation * math.pi / 180.0
        s = math.sin(ang)
        c = math.cos(ang)
        r[0][0] = c
        r[0][1] = -s
        r[1][0] = s
        r[1][1] = c

        t_ = np.eye(3)
        t_[0][2] = -resolution / 2.0
        t_[1][2] = -resolution / 2.0
        t_inv = torch.eye(3)
        t_inv[0][2] = resolution / 2.0
        t_inv[1][2] = resolution / 2.0
        t = reduce(np.matmul, [t_inv, r, t_, t])

    if invert:
        t = np.linalg.inv(t)
    new_point = (np.matmul(t, _pt))[0:2]

    return new_point.astype(float)

def parse_pts(pts_file):
    pts = []
    with open(pts_file) as f:
        for line in f.readlines():
            line = line.strip()
            if line[0].isdigit() == False:
                continue
            else:
                idx = line.find(' ')
                x, y = float(line[:idx]), float(line[idx+1:])
                pts.append([x, y])
    if len(pts) != 68:
        print('Not enough points')
    else:
        return np.array(pts)

class WFLWInstance():
    def __init__(self, line, idx):
        self.idx = idx
        line = line.strip().split(' ')
        # convert landmarks
        landmarks_list = list(map(float, line[:196]))
        self.landmarks = []
        for i in range(0, 196, 2):
            self.landmarks.append([landmarks_list[i], landmarks_list[i+1]])
        self.landmarks = np.array(self.landmarks)

        # convert bboxes
        if len(line) == 207:
            self.bbox = list(map(float, line[196:200]))
        else:
            self.bbox = None

        # convert image name
        self.image_base_name = line[-1]
        self.image_first_point = line[0]

def load_meta_subset_data(meta_path):
    with open(meta_path) as f:
        lines = f.readlines()

    meta_data = []
    idx = 0
    for line in lines:
        line = line.strip().split(' ')
        meta_data.append(line[-1]+line[0])
    return meta_data

def load_meta_data(meta_path, meta_subset_data=None):
    with open(meta_path) as f:
        lines = f.readlines()

    meta_data = []
    idx = 0
    for line in lines:
        wflw_instance = WFLWInstance(line, idx)
        if meta_subset_data is not None and (wflw_instance.image_base_name+wflw_instance.image_first_point) in meta_subset_data:
            meta_data.append(wflw_instance)
            idx += 1
    return meta_data

def process_single(single, image_path, image_save_path, landmarks_save_path):
    # print('Processing: {}'.format(single.image_base_name))
    image_full_path = os.path.join(image_path, single.image_base_name)
    image = io.imread(image_full_path)
    if len(image.shape) == 2:
        image = np.stack((image, image, image), -1)

    pts = single.landmarks
    left, top, right, bottom = [int(x) for x in single.bbox]
    lr_pad = int(0.05 * (right - left) / 2)
    tb_pad = int(0.05 * (bottom - top) / 2)
    left = max(0, left - lr_pad)
    right = right + lr_pad
    top = max(0, top - tb_pad)
    bottom = bottom + tb_pad

    center = torch.FloatTensor(
        [right - (right - left) / 2.0, bottom -
            (bottom - top) / 2.0])
    scale_factor = 250.0
    scale = (right - left + bottom - top) / scale_factor
    new_image, new_landmarks = cv_crop(image, pts, center, scale, 450, 0)
    while np.min(new_landmarks) < 10 or np.max(new_landmarks) > 440:
        scale_factor -= 10
        scale = (right - left + bottom - top) / scale_factor
        new_image, new_landmarks = cv_crop(image, pts, center, scale, 450, 0)
        assert (scale_factor > 0), "Landmarks out of boundary!"
    if new_image != []:
        io.imsave(os.path.join(image_save_path, os.path.basename(image_full_path[:-4]+'_' + str(single.idx) + image_full_path[-4:])), new_image)
        np.save(os.path.join(landmarks_save_path, os.path.basename(image_full_path[:-4]+ '_' + str(single.idx) + '.pts')), new_landmarks)

if __name__ == '__main__':
    image_path = './WFLW_images/'
    meta_subset_path = './WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt'
    meta_path = './WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt'
    image_save_path = './WFLW_test/images/'
    landmarks_save_path = './WFLW_test/landmarks/'
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
    if not os.path.exists(landmarks_save_path):
        os.makedirs(landmarks_save_path)
    exts = ['*.png', '*.jpg']
    meta_subset_data = load_meta_subset_data(meta_subset_path)
    meta_data = load_meta_data(meta_path, meta_subset_data)
    assert (len(meta_data) == len(meta_subset_data)), "Some images are missing!"
    print("Total images: {0:d}".format(len(meta_data)))
    Parallel(n_jobs=10,
            backend='threading',
            verbose=10)(delayed(process_single)(single, image_path,
                                                image_save_path,
                                                landmarks_save_path) for single in meta_data)
