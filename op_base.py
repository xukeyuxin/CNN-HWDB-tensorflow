import os
import tensorflow as tf
import random
import cv2
from tqdm import tqdm
import numpy as np

class op_base(object):
    def __init__(self,args):
        self.__dict__ = args.__dict__

    def shuffle(self):
        return random.shuffle(self.image_list)
        
    def init_train_data(self):
        dir_list = os.listdir('data/train')
        self.image_list = []
        for dir_item in tqdm(dir_list):
            dir_path = os.path.join('data/train',dir_item)
            img_list = os.listdir(dir_path)
            item_list = [ (dir_item, os.path.join(dir_path,img_name_item)) for img_name_item in img_list ]
            self.image_list += item_list
        self.shuffle()

    def init_test_data(self):
        dir_list = os.listdir('data/test')
        self.test_image_list = []
        for dir_item in tqdm(dir_list):
            dir_path = os.path.join('data/test',dir_item)
            img_list = os.listdir(dir_path)
            item_list = [ (dir_item, os.path.join(dir_path,img_name_item)) for img_name_item in img_list ]
            self.test_image_list += item_list

    ### gray img mix with white block
    def process(self,input):
        height,weight = input.shape[:2]
        ## threshold
        threshold_input = cv2.threshold(input, 220, 255, 0)[1]

        ## gray2float
        process_input = threshold_input / 255.

        ## add white
        add_size_0 = np.floor(np.abs(height - weight) / 2. ).astype(np.int32)
        add_size_1 = np.ceil(np.abs(height - weight) / 2. ).astype(np.int32)
        if(height > weight):
            add_block_0 = np.ones([height,add_size_0],dtype = np.int32)
            add_block_1 = np.ones([height,add_size_1],dtype = np.int32)
            new_input = np.concatenate([add_block_0, process_input, add_block_1],axis = 1)
        elif(weight >= height):
            add_block_0 = np.ones([add_size_0, weight],dtype = np.int32)
            add_block_1 = np.ones([add_size_1, weight],dtype = np.int32)
            new_input = np.concatenate([add_block_0, process_input, add_block_1],axis = 0)   

        ## add margin 
        current_size = new_input.shape[0]
        margin_size = current_size // 15  
        margin_height = np.ones([margin_size,current_size + 2 * margin_size])
        margin_weight = np.ones([current_size,margin_size])
        margin_input = np.concatenate([margin_weight, new_input, margin_weight],axis = 1)
        margin_input = np.concatenate([margin_height, margin_input, margin_height],axis = 0)

        ## resize 64
        resize_input = cv2.resize(margin_input,(64,64))

        ## expand_dim
        return np.expand_dims(resize_input, axis = -1)
        

 
    def read_img(self,path,read_type = 'gray'):
        if(read_type == 'gray'):
            img_content = cv2.imread(path,0)
            return self.process(img_content)

    def load_train_data_generator(self,load_path = 'train',init_data = False):
        one_batch_feed = []
        if(load_path == 'train'):
            if(init_data):
                self.init_train_data()
                print('finish init train data')
            _data_img_list = self.image_list
        elif(load_path == 'test'):
            self.init_test_data()
            _data_img_list = self.test_image_list

        for img_label,img_path in tqdm(_data_img_list):
            label_index = int(img_label)
            label_matrix = np.zeros((self.class_num),dtype = np.int32)
            label_matrix[label_index] = 1
            label_matrix = np.expand_dims(label_matrix,axis = 0)

            img_content = self.read_img(img_path)
            img_content = np.expand_dims(img_content,axis = 0)

            yield label_matrix, img_content



