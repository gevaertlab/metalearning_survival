import sys
sys.path.append("/home/linayqiu/miniconda3/envs/tfgpu/lib/python3.6/site-packages")
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
import torch
import random 
from torch import nn
from torch.autograd import Variable
import pandas as pd
from operator import add
import time
import argparse
import json

class DAPLModel(nn.Module):
    
    def __init__(self):
        nn.Module.__init__(self)
        
        self.main = nn.Sequential(
            nn.Linear(17176, 6000), 
            nn.ReLU(),
            nn.Linear(6000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 200),
            nn.ReLU(),
            nn.Linear(200, 1, bias=False)
        )
        
    def forward(self, x):
        return self.main(x)

    
def do_final_learning(model, x_ftrain, ystatus_ftrain, y_ftrain, lr_inner, n_inner, shots_n, n_iters,reg_scale):
    new_model = DAPLModel()
    new_model.load_state_dict(model.state_dict())  
    inner_optimizer = torch.optim.SGD(new_model.parameters(), lr=lr_inner,weight_decay=reg_scale)
    

    for nn in range(n_iters):  
        start=time.time()       
        ind=random.sample(range(x_ftrain_smp.shape[0]), shots_n)
        x_batch=x_ftrain[ind,]
        ystatus_batch=ystatus_ftrain[ind,]
        y_batch=y_ftrain[ind,]
        R_matrix_batch = np.zeros([y_batch.shape[0], y_batch.shape[0]], dtype=int)
        for i in range(y_batch.shape[0]):
            for j in range(y_batch.shape[0]):
                R_matrix_batch[i,j] = y_batch[j] >= y_batch[i]            
                   
        for i in range(n_inner):
    
            x_batch=Variable(torch.FloatTensor(x_batch),requires_grad = True )
            R_matrix_batch=Variable(torch.FloatTensor(R_matrix_batch),requires_grad = True )
            ystatus_batch=Variable(torch.FloatTensor(ystatus_batch),requires_grad = True )
            
            theta=new_model(x_batch)               
            exp_theta=torch.reshape(torch.exp(theta),[x_batch.shape[0]])
            theta=torch.reshape(theta,[x_batch.shape[0]])

            loss=-torch.mean(torch.mul((theta - torch.log(torch.sum(torch.mul(exp_theta, R_matrix_batch),dim=1))), torch.reshape(ystatus_batch,[x_batch.shape[0]])))
 
                                
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()
                                
        end=time.time()
        print("1 iteration time:", end-start)
        print ('Iteration', nn)
        print ('AvgTrainML', loss.data[0])
        
        
    return new_model

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

def do_final_eval(trained_model,x_test,y_test,ystatus_test):
        x_batch=torch.FloatTensor(x_test)
        pred_batch_test=trained_model(x_batch)              
        cind=CIndex(pred_batch_test, y_test, np.asarray(ystatus_test))
        
        return cind,pred_batch_test


def output_pred(trained_model,x_test,y_test,ystatus_test):
        x_batch=torch.FloatTensor(x_test)        
        theta=trained_model(x_batch)               
    
        return theta
        
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.json', help='configuration json file')

             
if __name__ == '__main__':


        args = parser.parse_args()
        with open(args.config) as f:
        	config = json.load(f)

    
        FINAL_N_INNER=config['final_n_inner']
        FINAL_LR_INNER=config['final_lr_inner']
        FINAL_SHOTS_N=config['final_shots_n']
        FINAL_ITER=config['final_iter']
        FINAL_REG_SCALE=config['final_reg_scale']
        SELECT_SAMPLE=config['select_sample']
        RESTORE_SERIES=config['restore_series']
        SAVE_PARAMS=config['final_params_save']


        
        output_path=config['output_path']
        model_path=config['model_path']
        x_ftrain = np.loadtxt(fname=config['train_feature'],delimiter=",",skiprows=1)          
        y_ftrain = np.loadtxt(fname=config['train_time'],delimiter=",",skiprows=1) 
        ystatus_ftrain = np.loadtxt(fname=config['train_status'],delimiter=",",skiprows=1) 
        x_test = np.loadtxt(fname=config['test_feature'],delimiter=",",skiprows=1) 
        y_test = np.loadtxt(fname=config['test_time'],delimiter=",",skiprows=1)        
        ystatus_test = np.loadtxt(fname=config['test_status'],delimiter=",",skiprows=1)                 
        CI_list=[] 
        score_train_list=[]
        score_test_list=[]

        for i in range(1,11):
            random.seed(i)
            smp_ind=random.sample(range(x_ftrain.shape[0]),SELECT_SAMPLE)
            x_ftrain_smp = x_ftrain[smp_ind,]
            y_ftrain_smp = y_ftrain[smp_ind,]
            ystatus_ftrain_smp = ystatus_ftrain[smp_ind,]
            filepath=model_path+RESTORE_SERIES+'.pt'
            meta_model = DAPLModel() 
            meta_model.load_state_dict(torch.load(filepath)) 
            trained_model =do_final_learning(model=meta_model, x_ftrain=x_ftrain_smp, ystatus_ftrain=ystatus_ftrain_smp, y_ftrain=y_ftrain_smp, lr_inner=FINAL_LR_INNER, n_inner=FINAL_N_INNER, shots_n=FINAL_SHOTS_N, n_iters=FINAL_ITER,reg_scale=FINAL_REG_SCALE) 
            
            CI,score_test= do_final_eval(trained_model,x_test,y_test,ystatus_test) 

            print(CI)
            CI_list.append(CI)
            score_test_list.append(score_test.data.numpy().reshape(score_test.shape[0],))

        print(CI_list)
        np.savetxt(output_path+RESTORE_SERIES+"_"+SAVE_PARAMS+"_testCI.csv", np.asarray(CI_list), delimiter=",")
        
