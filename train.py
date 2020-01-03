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
        self.sess_arg = tf.Session()
        self.summaries = []

    def get_vars(self, name, scope=None):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    def get_single_var(self,name):
        choose_var = [ var for var in tf.global_variables() if var.op.name.startswith(name)]
        return choose_var
    
    def update_lr(self):
        new_lr = 0.5 * self.lr
        update_lr_op = tf.assign(self.lr,new_lr)
        return update_lr_op

    def classify(self,d_opt,name = 'classify',is_training = True): ### 64,64,1
        with tf.variable_scope(name,reuse = tf.AUTO_REUSE):
            x = tf.pad(self.input_img,[[0,0],[5,5],[5,5],[0,0]],"REFLECT")
            x = ly.conv2d(x,64,kernal_size=11,name = 'conv_0',padding='VALID',use_bias=True)
            x = ly.batch_normal(x,name = 'bn_0',is_training = is_training)
            x = ly.relu(x)
            
            x = ly.maxpooling2d(x) ## 32,32,64
            print(x.shape)

            x = tf.pad(x,[[0,0],[3,3],[3,3],[0,0]],"REFLECT")
            x = ly.conv2d(x,128,kernal_size=7,name = 'conv_1',padding='VALID',use_bias=True)
            x = ly.batch_normal(x,name = 'bn_1',is_training = is_training)
            x = ly.relu(x)

            x = ly.maxpooling2d(x) ## 16,16,128
            print(x.shape)

            x = tf.pad(x,[[0,0],[1,1],[1,1],[0,0]],"REFLECT")
            x = ly.conv2d(x,256,kernal_size=3,name = 'conv_2',padding='VALID',use_bias=True)
            x = ly.batch_normal(x,name = 'bn_2',is_training = is_training)
            x = ly.relu(x)

            x = ly.maxpooling2d(x) ## 8,8,256
            print(x.shape)

            x = ly.fc(x,1024,name = 'fc_0',use_bias=True)
            x = ly.batch_normal(x,name = 'bn_3',is_training = is_training)
            x = ly.relu(x)
            x = tf.nn.dropout(x,keep_prob = 0.5)

            x = ly.fc(x,self.class_num,name = 'fc_1',use_bias=True)

            cross_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.input_label,logits = x),axis = 0)
            l2_loss = 0.0005 *  tf.reduce_sum( [   tf.nn.l2_loss(var) for var in self.get_single_var('classify/fc') ] )

            loss = cross_loss + l2_loss 
            self.summaries.append(tf.summary.scalar('loss',loss))
            
            _grad = d_opt.compute_gradients(loss,var_list=self.get_vars('classify'))
            train_op = d_opt.apply_gradients(_grad)
            return train_op

    def train(self,is_training = True):
        self.input_img = tf.placeholder(tf.float32,shape = [self.batch_size,self.input_height,self.input_weight,1])
        self.input_label = tf.placeholder(tf.float32,shape = [self.batch_size,self.class_num])
        self.lr = tf.get_variable('lr',shape = [],initializer = tf.constant_initializer(self.init_lr))
        self.summaries.append(tf.summary.scalar('lr',self.lr))
        
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
                    _item_batch = [ next(train_data_generator) for _ in range(self.batch_size) ]
                    _zip = zip(*_item_batch)
                    label_batch, img_batch = [ np.concatenate( item, axis = 0) for item in _zip ]
                    run_step += 1 
                except StopIteration:
                    print('finish opech %s' % epoch_time)
                    self.shuffle()
                    train_data_generator = self.load_train_data_generator()
                    break
                
                _feed_dict = {self.input_img:img_batch, self.input_label:label_batch}
                _, summary_str = self.sess.run([ train_op, summary_op ],feed_dict = _feed_dict)
                if(run_step % 10 == 0):
                    self.summary_writer.add_summary(summary_str,run_step)
                if(run_step % 500 == 0):
                    self.saver.save(self.sess,os.path.join(self.model_save_path,'classify_checkpoint_%s' % run_step))
                


