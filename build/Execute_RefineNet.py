#!/user/bin/python
# coding=utf-8
import os, sys
import sys
# Directories
REFINE_SOURCE_DIR = "D:\ProjectsD\ISMAR2019\code\src\SemanticSegmentation\RefineNet"
sys.path.insert(0, REFINE_SOURCE_DIR)
    
from models.resnet import rf_lw152
from utils.helpers import prepare_img
import cv2
import numpy as np
import torch
from PIL import Image


has_cuda = torch.cuda.is_available()
n_classes = 7
result = None

net = rf_lw152(n_classes, pretrained=True).eval().cuda()
with torch.no_grad():
    null_img = np.zeros((480, 640, 3), np.uint8)
    img_inp = torch.tensor(prepare_img(null_img).transpose(2, 0, 1)[None]).float().cuda()
    preds = net(img_inp)[0].data.cpu().numpy().transpose(1, 2, 0)
print(preds.shape)



def execute(rgb_image):
    global result
    with torch.no_grad():
        print('python taken!')
        cv2.imwrite("test.png", rgb_image)
        orig_size = rgb_image.shape[:2][::-1]
        img_inp = torch.tensor(prepare_img(rgb_image).transpose(2, 0, 1)[None]).float()
        img_inp = img_inp.cuda()
        segm = net(img_inp)[0].data.cpu().numpy().transpose(1, 2, 0)
        segm = cv2.resize(segm, orig_size, interpolation=cv2.INTER_CUBIC)
        segm = segm.argmax(axis=2).astype(np.uint8)
        res = np.zeros(orig_size[::-1], np.uint8)
        #res[segm == 1] = 1
        print(np.max(segm))
        result = segm
