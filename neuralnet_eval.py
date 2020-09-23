import os
import tensorflow as tf
import pandas as pd
import time
import random
import numpy as np
import sys
import argparse
import json

            
def CIndex(pred, ytime_test, ystatus_test):
    concord = 0.
    total = 0.
    N_test = ystatus_test.shape[0]
    ystatus_test = np.asarray(ystatus_test, dtype=bool)
    theta = pred
    for i in range(N_test):
        if ystatus_test[i] == 1:
            for j in range(N_test):
                if ytime_test[j] > ytime_test[i]:
                    total = total + 1
                    if theta[j] < theta[i]: concord = concord + 1
                    elif theta[j] == theta[i]: concord = concord + 0.5

    return(concord/total)               

def predict (FEATURE_SIZE,ROOT_PATH, CHECKPOINT_NAME, x_test,y_test, ystatus_test, x_train, y_train, ystatus_train):

    CHECKPOINT_FILE=ROOT_PATH+CHECKPOINT_NAME
    np.set_printoptions(threshold=np.inf)
    tf.reset_default_graph()
    
    #regularizer = tf.contrib.layers.l2_regularizer(scale=REG_SCALE)
    x = tf.placeholder(tf.float32,[None,FEATURE_SIZE], name='input_data')
    ystatus=tf.placeholder(tf.float32,[None,1],name='ystatus')
    R_matrix= tf.placeholder(tf.float32,[None,None],name='R_matrix')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    dense_layer1 = tf.layers.dense(inputs=x, units=6000, activation=tf.nn.relu)
    dense_drop1 = tf.nn.dropout(dense_layer1, keep_prob=keep_prob)
    dense_layer2 = tf.layers.dense(inputs=dense_drop1, units=2000, activation=tf.nn.relu)
    dense_drop2 = tf.nn.dropout(dense_layer2, keep_prob=keep_prob)
    dense_layer3 = tf.layers.dense(inputs=dense_drop2, units=200, activation=tf.nn.relu)
    dense_drop3 = tf.nn.dropout(dense_layer3, keep_prob=keep_prob)
    theta = tf.layers.dense(inputs=dense_drop3, units=1, activation=None,use_bias=False)
    theta=tf.reshape(theta,[-1])
    exp_theta=tf.exp(theta) 


    loss=-tf.reduce_mean(tf.multiply((theta - tf.log(tf.reduce_sum(tf.multiply(exp_theta , R_matrix),axis=1))), tf.reshape(ystatus,[-1]))) 
    
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
    
        sess.run(tf.global_variables_initializer()) 
        sess.run(tf.local_variables_initializer())
        #print(("Loading variables from '%s'." % CHECKPOINT_FILE))
        saver.restore(sess, CHECKPOINT_FILE)
        #print('restored')  
        
        R_matrix_test = np.zeros([y_test.shape[0], y_test.shape[0]], dtype=int)
        for i in range(y_test.shape[0]):
            for j in range(y_test.shape[0]):
                R_matrix_test[i,j] = y_test[j] >= y_test[i]  
                        
        loss_batch_test,pred_batch_test= sess.run([loss,theta], feed_dict={x: x_test, 
                                                    ystatus: ystatus_test,
                                                    R_matrix: R_matrix_test,
                                                    keep_prob:1})
        pred_train= sess.run(theta, feed_dict={x: x_train, keep_prob:1})
            
        cind=CIndex(pred_batch_test, y_test, np.asarray(ystatus_test))
    
        print(CHECKPOINT_NAME, loss_batch_test, cind)
        return cind, pred_train, pred_batch_test
        #print(pred_batch_test, y_test,np.asarray(ystatus_test) )

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.json', help='configuration json file')



if __name__ == '__main__':

        args = parser.parse_args()
        with open(args.config) as f:
        	config = json.load(f)

 
        FEATURE_SIZE=config['feature_size']
        RESTORE_SERIES=config['restore_series']
        SELECT_SIZE=config['select_size']
         

        model_path=config['model_path']
        
        x_tr1 = np.loadtxt(fname=config['train_feature_multi'],delimiter=",",skiprows=1)          
        y_tr1 = np.loadtxt(fname=config['train_time_multi'],delimiter=",",skiprows=1) 
        ystatus_tr1= np.loadtxt(fname=config['train_status_multi'],delimiter=",",skiprows=1) 
        x_tr2 = np.loadtxt(fname=config['train_feature_target'],delimiter=",",skiprows=1)          
        y_tr2 = np.loadtxt(fname=config['train_time_target'],delimiter=",",skiprows=1) 
        ystatus_tr2= np.loadtxt(fname=config['train_status_target'],delimiter=",",skiprows=1) 
        x_test = np.loadtxt(fname=config['test_feature'],delimiter=",",skiprows=1) 
        y_test = np.loadtxt(fname=config['test_time'],delimiter=",",skiprows=1)        
        ystatus_test = np.loadtxt(fname=config['test_status'],delimiter=",",skiprows=1) 
        
        x_test = np.array(x_test)
        y_test=np.array(y_test).reshape((-1,1))
        ystatus_test=np.array(ystatus_test).reshape((-1,1))        
         
                        
        CI_list=[]   
        score_train_list=[]
        score_test_list=[]
        START=1
        END= 2    

        for i in range(START, END):
            
            random.seed(i)
            smp_ind=random.sample(range(x_tr2.shape[0]),SELECT_SIZE)
            x_tr2_smp = x_tr2[smp_ind,]
            y_tr2_smp= y_tr2[smp_ind,]
            ystatus_tr2_smp = ystatus_tr2[smp_ind,]
            
            x_train=np.concatenate((x_tr1,x_tr2_smp),axis=0)
            y_train=np.concatenate((y_tr1,y_tr2_smp),axis=0)
            ystatus_train=np.concatenate((ystatus_tr1,ystatus_tr2_smp), axis=0)  
            print(x_train.shape)
            print(y_train.shape)
            
            x_train = np.array(x_train)
            y_train=np.array(y_train).reshape((-1,1))
            ystatus_train=np.array(ystatus_train).reshape((-1,1)) 

            CHECKPOINT_NAME=RESTORE_SERIES+'_dup'+str(i)+'.ckpt'
            CI,score_train,score_test=predict(FEATURE_SIZE,model_path, CHECKPOINT_NAME, x_test,y_test, ystatus_test, x_train, y_train, ystatus_train)
            print(CI)
            CI_list.append(CI)   
            score_train_list.append(score_train)
            score_test_list.append(score_test)
        print(CI_list)



