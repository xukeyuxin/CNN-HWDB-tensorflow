#coding:utf-8
import tensorflow as tf
from train import OCR
import argparse
import os
import sys
import cv2
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument("-iw", "--input_weight", type=int, default=64)
parser.add_argument("-ih", "--input_height", type=int, default=64)

parser.add_argument("-e", "--epoch", type=int, default=100)
parser.add_argument("-b", "--batch_size", type=int, default=100)
parser.add_argument("-cn", "--class_num", type=int, default=3755)
parser.add_argument("-l", "--init_lr", type=float, default=1e-2)

parser.add_argument("-ac", "--action", type=str, default='release')
parser.add_argument("-lg", "--summary_dir", type=str, default='logs')
parser.add_argument("-mp", "--model_save_path", type=str, default='model')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


dir_names = ['logs','model','data']
for dir in dir_names:
    if(not os.path.exists(dir)):
        os.mkdir(dir)

def test(model):
    img_list = [ os.path.join('eval',path) for path in os.listdir('eval') ]
    img_content = [ np.expand_dims(cv2.imread(path,0), axis = -1) for path in img_list ]
    model(img_content)

if __name__ == '__main__':
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(0)

    # config.log_device_placement=False
    # config = tf.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.allow_growth = True

    with tf.Session(config = config) as sess:
        model = OCR(args,sess)
        test(model)
        if(args.action == 'train'):
            model.train()
        elif(args.action == 'eval'):
            model.eval()
        elif(args.action == 'test'):
            test(model)
        

