#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import pandas as pd
import time
import random
import numpy as np
import sys
import argparse
import json

def get_batch_train(BATCH_SIZE, x_train, y_train, ystatus_train):
    while True:            
        j=0
        while (j+1)*BATCH_SIZE <= len(x_train):
            x_batch=x_train[int(j*BATCH_SIZE):int((j+1)*BATCH_SIZE),:]
            y_batch=y_train[int(j*BATCH_SIZE):int((j+1)*BATCH_SIZE)]
            ystatus_batch=ystatus_train[int(j*BATCH_SIZE):int((j+1)*BATCH_SIZE)]
                
            x_batch = np.array(x_batch)
            y_batch=np.array(y_batch).reshape((-1,1))
            ystatus_batch=np.array(ystatus_batch).reshape((-1,1))
            
            j+=1
            yield x_batch, y_batch, ystatus_batch

def get_batch_holdout(BATCH_SIZE, x_holdout, y_holdout, ystatus_holdout):
    while True:            
        j=0
        while (j+1)*BATCH_SIZE <= len(x_holdout):
            x_batch=x_holdout[int(j*BATCH_SIZE):int((j+1)*BATCH_SIZE),:]
            y_batch=y_holdout[int(j*BATCH_SIZE):int((j+1)*BATCH_SIZE)]
            ystatus_batch=ystatus_holdout[int(j*BATCH_SIZE):int((j+1)*BATCH_SIZE)]
                
            x_batch = np.array(x_batch)
            y_batch=np.array(y_batch).reshape((-1,1))
            ystatus_batch=np.array(ystatus_batch).reshape((-1,1))
            
            j+=1
            yield x_batch, y_batch, ystatus_batch
            
############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.json', help='configuration json file')

if __name__ == '__main__':
    
    
        args = parser.parse_args()
        with open(args.config) as f:
        	config = json.load(f)

        
        FEATURE_SIZE= config['feature_size']
        BATCH_SIZE=config['batch_size']
        LEARNING_RATE=config['lr']
        NUM_EPOCHES=config['num_epo']
        KEEP_PROB=config['keep_prob']
        REG_SCALE=config['reg_scale']
        
        
        model_path=config['model_path']

        x_train = np.loadtxt(fname=config['train_feature'],delimiter=",",skiprows=1)          
        y_train = np.loadtxt(fname=config['train_time'],delimiter=",",skiprows=1) 
        ystatus_train = np.loadtxt(fname=config['train_status'],delimiter=",",skiprows=1) 
        x_holdout = np.loadtxt(fname=config['val_feature'],delimiter=",",skiprows=1) 
        y_holdout = np.loadtxt(fname=config['val_time'],delimiter=",",skiprows=1)        
        ystatus_holdout = np.loadtxt(fname=config['val_status'],delimiter=",",skiprows=1) 

        NUM_TRAIN_STEPS=int(len(y_train)/BATCH_SIZE)
        EVA_STEP=2
        CHECKPOINT_FILE=model_path+'pretrain_4layer200_dropout'+str(KEEP_PROB)+'_reg'+str(REG_SCALE)+'_batch'+str(BATCH_SIZE)+'_epo'+str(NUM_EPOCHES)+'.ckpt'


        np.set_printoptions(threshold=np.inf)
        tf.reset_default_graph()        
        regularizer = tf.contrib.layers.l2_regularizer(scale=REG_SCALE)
        x = tf.placeholder(tf.float32,[None,FEATURE_SIZE], name='input_data')
        ystatus=tf.placeholder(tf.float32,[None,1],name='ystatus')
        R_matrix= tf.placeholder(tf.float32,[None,None],name='R_matrix')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        
        dense_layer1 = tf.layers.dense(inputs=x, units=6000, activation=tf.nn.relu,kernel_regularizer=regularizer)
        dense_drop1 = tf.nn.dropout(dense_layer1, keep_prob=keep_prob)
        dense_layer2 = tf.layers.dense(inputs=dense_drop1, units=2000, activation=tf.nn.relu,kernel_regularizer=regularizer)
        dense_drop2 = tf.nn.dropout(dense_layer2, keep_prob=keep_prob)
        dense_layer3 = tf.layers.dense(inputs=dense_drop2, units=200, activation=tf.nn.relu,kernel_regularizer=regularizer)
        dense_drop3 = tf.nn.dropout(dense_layer3, keep_prob=keep_prob)
        theta = tf.layers.dense(inputs=dense_drop3, units=1, activation=None,use_bias=False,kernel_regularizer=regularizer)
        theta=tf.reshape(theta,[-1])
        exp_theta=tf.exp(theta) 
                
        loss=-tf.reduce_mean(tf.multiply((theta - tf.log(tf.reduce_sum(tf.multiply(exp_theta , R_matrix),axis=1))), tf.reshape(ystatus,[-1]))) 
        l2_loss=tf.losses.get_regularization_loss()
        loss=loss+l2_loss
        
        optimizer=tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
        
        saver = tf.train.Saver()
        
        
        with tf.Session() as sess:
            print('Graph started...')
            print('NUM_TRAIN_STEPS',NUM_TRAIN_STEPS)
            print('NUM_EPOCHES',NUM_EPOCHES)
            sess.run(tf.global_variables_initializer()) # this could be very slow with large w and large NUM_CATES
            sess.run(tf.local_variables_initializer())
            for ep in range(NUM_EPOCHES):
        
                batch_gen_train=get_batch_train(BATCH_SIZE,x_train, y_train, ystatus_train) 
                batch_gen_holdout=get_batch_holdout(700,x_holdout, y_holdout, ystatus_holdout) 
                total_loss_train=0.0
                total_loss_holdout=0.0
        
                for step in range(NUM_TRAIN_STEPS):
                    batch_x_train, batch_y_train, batch_ystatus_train = next(batch_gen_train) 
                    batch_x_holdout, batch_y_holdout,batch_ystatus_holdout = next(batch_gen_holdout)
                    R_matrix_train = np.zeros([batch_y_train.shape[0], batch_y_train.shape[0]], dtype=int)
                    for i in range(batch_y_train.shape[0]):
                        for j in range(batch_y_train.shape[0]):
                            R_matrix_train[i,j] = batch_y_train[j] >= batch_y_train[i]
                    R_matrix_holdout = np.zeros([batch_y_holdout.shape[0], batch_y_holdout.shape[0]], dtype=int)
                    for i in range(batch_y_holdout.shape[0]):
                        for j in range(batch_y_holdout.shape[0]):
                            R_matrix_holdout[i,j] = batch_y_holdout[j] >= batch_y_holdout[i]  
                              
                    loss_batch_train,  _ = sess.run([loss, optimizer], feed_dict={x: batch_x_train, 
                                                                                  ystatus: batch_ystatus_train,
                                                                                  R_matrix: R_matrix_train,
                                                                                  keep_prob:KEEP_PROB})
                    loss_batch_holdout= sess.run(loss, feed_dict={x: batch_x_holdout, 
                                                                    ystatus: batch_ystatus_holdout,
                                                                    R_matrix: R_matrix_holdout,
                                                                     keep_prob:1})
        
                    total_loss_train += loss_batch_train
                    total_loss_holdout += loss_batch_holdout

        
                    
                    if (step+1) % EVA_STEP == 0: # print loss every EVA_STEP
                        print('Average train loss at Epoch %d and Step %d is: %f' %(ep, step, total_loss_train/EVA_STEP),';',
                        'Holdout loss is: %f' %(total_loss_holdout/EVA_STEP))
                        total_loss_train=0.0
                        total_loss_holdout=0.0
                        save_path = saver.save(sess, CHECKPOINT_FILE)
            print(("Model saved in file: %s" % save_path))
            
    
    
