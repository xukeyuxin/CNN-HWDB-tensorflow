import tensorflow as tf
import numpy as np
import layer as ly
from op_base import op_base
from util import *
import os

class OCR(op_base):
    def __init__(self,args,sess):
        op_base.__init__(self,args)
        self.sess = sess
        self.summaries = []
        self.init_base_graph()
        self.init_index2txt()

    def init_index2txt(self):
        self.index2text = {}
        with open('index2text.txt') as f:
            start_index = 0
            for item in f:
                index, text = item.strip().split()
                self.index2text[start_index] = text
                start_index += 1

    def init_base_graph(self):
        self.input_img = tf.placeholder(tf.float32,shape = [None,self.input_height,self.input_weight,1])
        self.input_label = tf.placeholder(tf.float32,shape = [None,self.class_num])
        self.lr = tf.get_variable('lr',shape = [],initializer = tf.constant_initializer(self.init_lr))
        self.summaries.append(tf.summary.scalar('lr',self.lr))

    def get_vars(self, name, scope=None):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    def get_single_var(self,name):
        choose_var = [ var for var in tf.global_variables() if var.op.name.startswith(name)]
        return choose_var
    
    def update_lr(self):
        new_lr = 0.5 * self.lr
        update_lr_op = tf.assign(self.lr,new_lr)
        return update_lr_op

    def classify(self,d_opt = None,name = 'classify',is_training = True): ### 64,64,1
        with tf.variable_scope(name,reuse = tf.AUTO_REUSE):
            x = tf.pad(self.input_img,[[0,0],[5,5],[5,5],[0,0]],"REFLECT")
            x = ly.conv2d(x,64,kernal_size=11,name = 'conv_0',padding='VALID',use_bias=True)
            x = ly.batch_normal(x,name = 'bn_0',is_training = is_training)
            x = ly.relu(x)
            
            x = ly.maxpooling2d(x) ## 32,32,64

            x = tf.pad(x,[[0,0],[3,3],[3,3],[0,0]],"REFLECT")
            x = ly.conv2d(x,128,kernal_size=7,name = 'conv_1',padding='VALID',use_bias=True)
            x = ly.batch_normal(x,name = 'bn_1',is_training = is_training)
            x = ly.relu(x)

            x = ly.maxpooling2d(x) ## 16,16,128

            x = tf.pad(x,[[0,0],[2,2],[2,2],[0,0]],"REFLECT")
            x = ly.conv2d(x,256,kernal_size=5,name = 'conv_2',padding='VALID',use_bias=True)
            x = ly.batch_normal(x,name = 'bn_2',is_training = is_training)
            x = ly.relu(x)

            x = ly.maxpooling2d(x) ## 8,8,256

            x = tf.pad(x,[[0,0],[1,1],[1,1],[0,0]],"REFLECT")
            x = ly.conv2d(x,512,kernal_size=3,name = 'conv_3',padding='VALID',use_bias=True)
            x = ly.batch_normal(x,name = 'bn_3',is_training = is_training)
            x = ly.relu(x)

            x = tf.pad(x,[[0,0],[1,1],[1,1],[0,0]],"REFLECT")
            x = ly.conv2d(x,512,kernal_size=3,name = 'conv_4',padding='VALID',use_bias=True)
            x = ly.batch_normal(x,name = 'bn_4',is_training = is_training)
            x = ly.relu(x)

            x = ly.maxpooling2d(x) ## 4,4,512

            x = ly.fc(x,1024,name = 'fc_0',use_bias=True)
            x = ly.batch_normal(x,name = 'bn_5',is_training = is_training)
            x = ly.relu(x)
            x = tf.nn.dropout(x,keep_prob = 0.5)

            x = ly.fc(x,self.class_num,name = 'fc_1',use_bias=True)
            self.pred_x_index = tf.argmax(tf.nn.softmax(x),axis = -1)
            self.pred_x_value = tf.reduce_max(tf.nn.softmax(x),axis = -1)

            if(is_training):
                cross_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.input_label,logits = x),axis = 0)
                l2_loss = 0.0005 *  tf.reduce_sum( [   tf.nn.l2_loss(var) for var in self.get_single_var('classify/fc') ] )
                loss = cross_loss + l2_loss 
                self.summaries.append(tf.summary.scalar('loss',loss))
                
                _grad = d_opt.compute_gradients(loss,var_list=self.get_vars('classify'))
                train_op = d_opt.apply_gradients(_grad)

                return train_op


    def accurity(self,label_batch_index,pred_x_index):
        acc = np.sum( [ int(index_real == index_pred) for index_real, index_pred in zip(*[label_batch_index,pred_x_index ]) ] ) / self.batch_size
        return acc
    
    def load_one_batch(self,data_generator,max_length = None):
        _item_batch = [ next(data_generator) for _ in range(max_length or self.batch_size) ]
        _zip = zip(*_item_batch)
        label_batch, img_batch = [ np.concatenate( item, axis = 0) for item in _zip ]
        return label_batch, img_batch
    
    def load_single_img(self,data_generator,max_length = None):
        _item_batch = np.array([ next(data_generator)  for _ in range(max_length or self.batch_size) ])
        print(_item_batch.shape)
        return _item_batch

    def train(self,is_training = True):
        
        d_opt = tf.train.MomentumOptimizer(self.lr,0.9)
        train_op = self.classify(d_opt,is_training = is_training)
        summary_op = tf.summary.merge(self.summaries)
        lr_update_op = self.update_lr()
        ## init
        self.sess.run(tf.global_variables_initializer())

        ## init summary
        self.summary_writer = tf.summary.FileWriter(self.summary_dir,self.sess.graph)
        ## saver
        self.saver = tf.train.Saver(max_to_keep=5)
        ## start train
        train_data_generator = self.load_train_data_generator(init_data = True)
        run_step = 0
        for epoch_time in range(1,self.epoch):
            if(epoch_time % 3 == 0):
                self.sess.run(lr_update_op)
            while True:
                try:
                    label_batch, img_batch = self.load_one_batch(train_data_generator)
                    run_step += 1 

                except StopIteration:
                    print('finish opech %s' % epoch_time)
                    self.shuffle()
                    train_data_generator = self.load_train_data_generator()
                    break
                
                _feed_dict = {self.input_img:img_batch, self.input_label:label_batch}
                _, pred_x_index, summary_str = self.sess.run([ train_op, self.pred_x_index, summary_op ],feed_dict = _feed_dict)
                if(run_step % 10 == 0):
                    self.summary_writer.add_summary(summary_str,run_step)

                if(run_step % 1000 == 0):
                    ### train acc
                    label_batch_index = np.argmax(label_batch, axis = -1)
                    self.accurity(label_batch_index,pred_x_index)
                    self.saver.save(self.sess,os.path.join(self.model_save_path,'classify_checkpoint_%s' % run_step))
        

    def eval(self):
        ## build graph
        self.classify(is_training = False)
        ## saver
        self.saver = tf.train.Saver(max_to_keep=5)
        ## init
        self.sess.run(tf.global_variables_initializer())
        ## restore
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_save_path))
        ## test data
        test_data_generator = self.load_train_data_generator(load_path='test')

        final_acc = []
        while True:
            try:
                label_batch, img_batch = self.load_one_batch(test_data_generator)
                _feed_dict = {self.input_img:img_batch, self.input_label:label_batch}
                pred_x_index = self.sess.run( self.pred_x_index,feed_dict = _feed_dict)
                label_batch_index = np.argmax(label_batch, axis = -1)
                final_acc.append(self.accurity(label_batch_index,pred_x_index))
            except StopIteration:
                print('finish eval %s' % np.mean(final_acc))
                return 
    
    def __call__(self,img_list):
        img_generator = self.load_hd_img_generator(img_list)
        max_length = len(img_list)
        self.batch_size = max_length

        ## build graph
        self.classify(is_training = False)
        ## saver
        self.saver = tf.train.Saver(max_to_keep=5)
        ## init
        self.sess.run(tf.global_variables_initializer())
        ## restore
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_save_path))

        result = []
        while True:
            try:
                img_batch = self.load_single_img(img_generator, max_length)
                _feed_dict = {self.input_img:img_batch}
                pred_x_index = self.sess.run( self.pred_x_index,feed_dict = _feed_dict)
                result = [ self.index2text[item_pred] for item_pred in pred_x_index] 
            except:
                return result
        
        return result
            


        

                


