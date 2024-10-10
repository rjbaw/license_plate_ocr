# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
test pretrained model.
Author: aiboy.wei@outlook.com .
'''

from PIL import Image, ImageDraw, ImageFont
from model.LPRNet import build_lprnet
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import cv2
import os
import imutils

import pdb


# set constants since arg parse wont work

#CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
#         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
#         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
#         '新',
#         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
#         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
#         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
#         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
#         ]

CHARS = [
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','-', ' '
         ]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

#test_img_dirs', default="./data/test", help='the test images path')
#test_batch_size', default=100, help='testing batch size.')

img_size = [94, 24]
dropout_rate = 0
lpr_max_len = 8 #license plate number max length
phase_train = False #train or test phase flag
# num_workers = 8 #Number of workers used in dataloading
cuda = True
#pretrained_model = '/home/dt18/tensorflow-onnx/license-plate/Final_LPRNet_model.pth'
pretrained_model = './test.pth'


# (width, height) = img.shape[:2]

# do i even need this??
#def collate_fn(batch):
#    imgs = []
#    labels = []
#    lengths = []
#    for _, sample in enumerate(batch):
#        img, label, length = sample
#        imgs.append(torch.from_numpy(img))
#        labels.extend(label)
#        lengths.append(length)
#    labels = np.asarray(labels).flatten().astype(np.float32)

#    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

def test(img ,device):

    lprnet = build_lprnet(lpr_max_len = lpr_max_len, phase = phase_train, class_num = len(CHARS), dropout_rate = dropout_rate)
    lprnet.to(device)
    print("Successful to build network!")

    # load pretrained model
    if pretrained_model:
        lprnet.load_state_dict(torch.load(pretrained_model))
        print("load pretrained model successful!")
    else:
        print("[Error] Can't found pretrained model, please check!")
        return False

    # img_size =

    height, width = img.shape[:2]
#    print(img.shape[:2])python detectcentroid.py --source 0 --weights yolov3-spp.weights

#    height, width = img.shape
    if height != img_size[1] or width != img_size[0]:
        imgsize = tuple(img_size)
#        img = cv2.resize(img, imgsize)
#        img = imutils.resize(img, width = 94)
#        img = imutils.resize(img, height = 24)
        img = cv2.resize(img, imgsize)
#        cv2.imshow('sd', img)

    # transform it
    img = img.astype('float32')
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))

    img = torch.from_numpy(img).to(device)

#    imgs = []
#    for i in range(4):
#        imgs.append(img)
#
#    img = torch.stack(imgs, 0)

    lb = Greedy_Decode_Eval(lprnet, img)

#    try:
#        Greedy_Decode_Eval(lprnet, img)
#    finally:
#        cv2.destroyAllWindows()
    return lb

def Greedy_Decode_Eval(Net, image):
    # TestNet = Net.eval() ????? for what

#    epoch_size = len(datasets) // test_batch_size
#    batch_iterator = iter(DataLoader(datasets, test_batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn))

# constants
#    Tp = 0
#    Tn_1 = 0
#    Tn_2 = 0

#    t1 = time.time()

#    for i in range(epoch_size):
        # load train data
#        images, labels, lengths = next(batch_iterator)
### torch.stack(imgs,0)
#        start = 0
#        targets = []

#        for length in lengths:
#            label = labels[start:start+length]
#            targets.append(label)
#            start += length
#        targets = np.array([el.numpy() for el in targets])

#        imgs = images.numpy().copy()

    img = image.cpu().numpy().copy()

#    if cuda:
#        images = Variable(images.cuda())
#    else:
#        images = Variable(images)

    if image.ndimension() == 3:
        image = image.unsqueeze(0)

        # forward
    prebs = Net(image)

        # greedy decode
    prebs = prebs.cpu().detach().numpy()
    preb_labels = list()

    for i in range(prebs.shape[0]):
        preb = prebs[i, :, :]
        preb_label = list()

    for j in range(preb.shape[1]):
        preb_label.append(np.argmax(preb[:, j], axis=0))

    no_repeat_blank_label = list()
    pre_c = preb_label[0]

    for c in preb_label: # dropout repeate label and blank label
        if (pre_c == c) or (c == len(CHARS) - 1):
            if c == len(CHARS) - 1:
                pre_c = c
            continue
        no_repeat_blank_label.append(c)
        pre_c = c
    preb_labels.append(no_repeat_blank_label)

    for i, label in enumerate(preb_labels):
        lb = check(img, label)

#        for i, label in enumerate(preb_labels):
            # show image and its predict label
#            if show:
#                show(imgs[i], label, targets[i])

#            if len(label) != len(targets[i]):
#                Tn_1 += 1
#                continue

#            if (np.asarray(targets[i]) == np.asarray(label)).all():
#                Tp += 1
#            else:
#                Tn_2 += 1
#    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
#    print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
#    t2 = time.time()
#    print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))
    return lb


def check(img, label):

    img = np.transpose(img, (1, 2, 0))
    img *= 128.
    img += 127.5
    img = img.astype(np.uint8)

    lb = ""
    for i in label:
        lb += CHARS[i]

    save_dir = './label/' + lb
    image_dir = save_dir + '.jpg'
    if os.path.exists(image_dir):
        image_dir = ''
        i = 1
        image_dir = save_dir + '_' + str(i) + '.jpg'
        while os.path.exists(image_dir):
            basename = os.path.basename(image_dir)
            imgname, suffix = os.path.splitext(basename)
            i = int(imgname.split('_')[1]) + 1
            holder_dir = save_dir + '_' + str(i)
            image_dir = holder_dir + '.jpg'
            holder_dir = ''

    cv2.imwrite(image_dir, img)

    for i, l in enumerate(lb):
        if i == 2 and l != '-':
            lb = None
            break
        elif i == 0 and l == '0':
            lb = None
            break
        elif i > 7:
            lb = None
            break
        elif i != 2 and l == '-':
            lb = None
            break
        else:
            continue

    if lb != None and len(lb) != 7:
        lb = None


#        if lb[0] == '0' or lb[:2] == '-' or lb[3:] == '-' or lb[2] != '-' or len(lb) != 7:
#        lb = None


#    tg = ""
#    for j in target.tolist():
#        tg += CHARS[int(j)]

#    flag = "F"
#    if lb == tg:
#        flag = "T"

    # img = cv2.putText(img, lb, (0,16), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 0, 255), 1)
###    img = cv2ImgAddText(img, lb, (0, 0))

#    cv2.imshow("test", img)
#    print("target: ", tg, " ### {} ### ".format(flag), "predict: ", lb)
#    cv2.waitKey()
#    cv2.destroyAllWindows()
    return lb

#def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=12):
#    if (isinstance(img, np.ndarray)):  # detect opencv format or not
#        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#    draw = ImageDraw.Draw(img)
#    fontText = ImageFont.truetype("data/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
#    draw.text(pos, text, textColor, font=fontText)
#
#    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


#if __name__ == "__main__":
#    test()
