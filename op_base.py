import os
import tensorflow as tf
import random
import cv2

class op_base():
    def __init__(self,args):
        self.__dict__ = args.__dict__
    def shuffle(self):
        return random.shuffle(self.image_list)
    def init_train_data(self):
        dir_list = os.listdir('data/train')
        self.image_list = []
        for dir_item in dir_list:
            dir_path = os.path.join('data/train',dir_item)
            img_list = os.listdir(dir_path)
            item_list = [ (dir_item, os.path.join(dir_path,img_name_item)) for img_name_item in img_list ]
            image_list += item_list
        self.shuffle()

    def read_img(self,path,read_type = 'gray'):
        if(read_type == 'gray'):
            return cv2.imread(path,0) / 255.

    def load_train_data_generator(self,init_data = False):
        if(init_data):
            self.init_train_data()
        for img_label,img_path in self.image_list:
            label_index = int(img_label)
            label_matrix = tf.zeros([self.class_num],dtype = tf.int32)
            label_matrix[label_index] = 1
            label_matrix = tf.expand_dims(label_matrix,axis = 0)

            img_content = self.read_img(img_path)

            yield label_matrix, img_content


