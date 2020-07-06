# -*- coding: utf-8 -*-
"""
March 10, 2018
Author: Sindhu
"""

import numpy as np

def logistic(x): #input should be anumpy array
    return 1 / (1 + np.exp(-x))

class ann: #assumes single node at output layer
    def __init__(self,k1s,k2s,meds,weights,act_funcs):
        self.k1s = k1s
        self.k2s = k2s
        self.meds = meds
        self.weights = weights
        self.act_funcs = act_funcs
    def output(self, input_mat, input_normalised=True, denormalise_output=False):
        num_ip = len(input_mat[0])
        num_attr = len(input_mat)
        if not(input_normalised): #normalise input if required
            for i in range(num_attr):
                k1 = self.k1s[i]
                k2 = self.k2s[i]
                med = self.meds[i]
                for j in range(num_ip):
                    if input_mat[i][j]<=med:
                        input_mat[i][j] *= k1
                    else:
                        input_mat[i][j] *= k2
        output_mat = np.concatenate(([[1 for i in range(num_ip)]], input_mat)) #add the bias node
        num_layers = len(self.act_funcs)
        for i in range(num_layers):
            output_mat = self.act_funcs[i](np.dot(self.weights[i], output_mat))
            output_mat = np.concatenate([[1 for i in range(num_ip)]], output_mat)
        output_mat = output_mat[1:]
        if denormalise_output: #denormalise output if needed
            num_ip = len(input_mat[0])
            num_attr = len(input_mat)
            m = 0
            if self.act_funcs[-1]==logistic:
                m = 0.5
            for i in range(num_attr):
                k1 = self.k1s[i]
                k2 = self.k2s[i]
                for j in range(num_ip):
                    if output_mat[i][j]<=m:
                        output_mat[i][j] /= k1
                    else:
                        output_mat[i][j] /= k2
        return output_mat
            
def get_norm_params(train_data, op_act_func): #assumes min, median, max are all different for each variable
    k1s = []
    k2s = []
    meds = []
    num_cols = len(train_data[0])
    num_train = len(train_data)
    mid = int(num_train/2)
    for i in range(num_cols):
        col = sorted([line[i] for line in train_data])
        min_val = col[0]
        max_val = col[-1]
        med_val = col[mid]
        k1s.append(0.8/(med_val-min_val))
        k2s.append(0.8/(max_val-med_val))
        meds.append(med_val)
    if op_act_func==logistic:
        k1s[-1] = 0.4/(med_val-min_val)
        k2s[-1] = 0.4/(max_val-med_val)
    return (k1s, k2s, meds)
        
def normalise(data, k1s, k2s, meds):
    num_cols = len(data[0])
    num_data = len(data)
    for i in range(num_data):
        for j in range(num_cols):
            if data[i][j]<=meds[j]:
                data[i][j] *= k1s[j]
            else:
                data[i][j] *= k2s[j]
    return data

def nodes_per_hl(h,f,i,t): #assumes h>=1, f*t>=i+3
    if h==1:
        num_nodes = int((t*f-1)/(i+2))
    else:
        num_nodes = int((np.sqrt(t*f-1+(h+i+1)**2/(4*h-1)) - (h+i+1)/(2*np.sqrt(h+1)))/(h-1))
    if num_nodes>=1:
        return (h, num_nodes)
    else:
        return ((f*t-i-1)/2, 1)

def back_propagate(my_ann, train_data, val_data, mini_bat_size, lr, reg, mom, max_iters):
    #get data in required shape
    t_input_mat = []
    v_input_mat = []
    num_attr = len(train_data[0])-1
    for i in range(num_attr):
        t_input_mat.append([line[i] for line in train_data])
        v_input_mat.append([line[i] for line in val_data])
    t_output_mat = [[line[-1]] for line in train_data]
    v_output_mat = [[line[-1]] for line in val_data]
    #iterate
    
   
def make_ann(dataset, num_hl, f_unknowns, act_funcs, mini_bat_size, lr, reg, mom, max_iters, print_info=True): 
    #extract train, validation, test data
    all_data = [[float(num) for num in line.rstrip('\n').replace(',',' ').split()] for line in open(dataset)]
    num_data = len(all_data)
    test_ind = int(0.9*num_data)
    test_data = all_data[test_ind]
    train_data = []
    val_data = []
    for i in range(test_ind):
        if (i%9)==7 or (i%9)==8:
            val_data.append(all_data[i])
        else:
            train_data.append(all_data[i])
    #normalise all i/p and o/p variables
    result = get_norm_params(train_data, act_funcs[-1])
    k1s = result[0]
    k2s = result[1]
    meds = result[2]
    train_data = normalise(train_data, k1s, k2s, meds)
    val_data = normalise(val_data, k1s, k2s, meds)
    #initialise weights and ann
    num_attr = len(train_data[0])-1
    (h, n) = nodes_per_hl(num_hl, f_unknowns, num_attr, len(train_data))
    num_nodes = [num_attr] + [n for i in range(h)] + [1] #without biases
    num_layers = num_hl+2
    weights = []
    for i in range(1, num_layers):
        ncurr = num_nodes[i]
        nprev = num_nodes[i-1] + 1
        weights.append(np.random.rand(ncurr,nprev)/np.sqrt(nprev))
    my_ann = ann(k1s,k2s,meds,weights,act_funcs)
    #train the ann using back-propagation
    result = back_propagate(my_ann, train_data, val_data, mini_bat_size, lr, reg, mom, max_iters)


    


    
if __name__=='__main__':
    make_ann('ccpp.txt', 3, 0.5, [np.tanh,np.tanh,logistic], 64, 0.01, 0.1, 0.9,100)
            
        
    
    

