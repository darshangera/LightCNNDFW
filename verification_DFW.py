#Aum Sri Sai Ram

#Compute Face verification accuracy for DFW dataset using computed features

import numpy as np
import os
from eval_metrics import evaluate

def read_pairs(pairs_filename):
         pairs = []
         
         with open(pairs_filename, 'r') as f:
             for line in f.readlines():
         
                 pair = line.strip().split()
                 pairs.append(pair)
         #print(pairs[:10])
         return pairs[:]
        
def compute_similarity(pairs):
    cos_sim = []
    labels = []
    for i in range(len(pairs)):
        file1, file2, label = pairs[i][0], pairs[i][1], int(pairs[i][2])
        file1 = file1[:-3]+'feat'
        file2 = file2[:-3]+'feat'
        if label in [1]:  #+ve
            label = 1
        else:  #-ve
            if label == 3:
                label = 0
            
        f1 = open(file1,'rb')
        feat1 = np.fromfile(f1)
        
        f2 = open(file2,'rb')
        feat2 = np.fromfile(f2)
        
        sim = cosin_metric(feat1,feat2)
        cos_sim.append(sim)
        labels.append(label)
    return np.array(cos_sim),np.array(labels)
        

def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))        
        
        
        
   
if __name__=='__main__':
    #pairs  = read_pairs(pairs_filename ='../DisguisedFacesInTheWild/dfw_testdata_pairs_only123.txt')
    pairs  = read_pairs(pairs_filename ='../DisguisedFacesInTheWild/dfw_testdata_pairs.txt')
    similarity, labels = compute_similarity(pairs)
    acc,th = cal_accuracy( similarity, labels)
    accuracy = evaluate(1-similarity,labels)
    print('DFW face verification accuracy: ', acc, 'threshold: ', th, 'acc',np.mean(accuracy))
    #print(similarity, labels)
    
    
    

    
