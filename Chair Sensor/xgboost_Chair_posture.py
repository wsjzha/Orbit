# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:01:54 2017

@author:  Zhenhua
"""

# First XGBoost model for Pima Indians dataset
import numpy as np
import xgboost as xgb
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
import matplotlib.pyplot as plt
from xgboost import plot_tree

import pandas as pd
from tsfresh import extract_features, extract_relevant_features, select_features

import time 
import random

import pylab as pl

from decimal import *
getcontext().prec = 8

# for how to extract features, see tsfresh_Chair_user_identify_v1.py

# function for labeling the features
def label_features(dataframe, num_feature):
        dataframe_labeled = dataframe;
        dataframe_labeled.loc[-1] =  range(num_feature)  # adding a row
        dataframe_labeled.index = dataframe_labeled.index + 1
        dataframe_labeled = dataframe_labeled.sort_index()
        return dataframe_labeled

# some data about the system
# the tag 
num_tags = 16;
num_seconds = 1;

# load data
num_samples = num_tags*num_seconds+1;

num_class = 13;

dataset = np.loadtxt('C:/Users/Dest/OneDrive/Chair Sensor/Posture_raw_data - 01.15.2018 - all.csv',
                     delimiter=",", converters={num_samples: lambda x:int(x)-1 })

# consider the person's identity
Is_identity = 0;

if Is_identity == 1:
        X = dataset[:,0:num_samples]
else:
        X = dataset[:,0:num_samples-1]
Y = dataset[:,num_samples]

# setup parameters for xgboost
param = {}

# use softmax multi-class classification
param['objective'] = 'multi:softmax'

# scale weight of positive examples
param['eta'] = 0.05;
param['max_depth'] = 2;
param['silent'] = 1;
# param['nthread'] = 4    # number of parallel threads used to run xgboost, default to max
param['num_class'] = num_class;

param['gamma'] = 0;
param['subsample'] = 1;

# number of iterations
num_iter = 50;

# size for split and the random seed range
seed_max = num_iter*num_iter;
test_size = 32000;
num_round = 1800

# testing parameters
Is_save_file = 1;

# 
error_summary = np.zeros(num_iter);

# target list
target_names = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5', 
                'class 6', 'class 7', 'class 8', 'class 9', 'class 10', 'class 11', 
                'class 12', 'class 13']

# start of the iterations of cross validation
for i in range(num_iter):
        print('iteration ',i)
        
        # a single iteration, split the data into training group and testing group
        seed = random.randint(0, seed_max);
        train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=seed)
        
        xg_train = xgb.DMatrix(train_X, label=train_Y)
        xg_test = xgb.DMatrix(test_X, label=test_Y)
        
        watchlist = [ (xg_train,'train'), (xg_test, 'test') ]

        bst = xgb.train(param, xg_train, num_round, watchlist );
        # get prediction
        pred = bst.predict( xg_test );
        
        # feature importance
        # http://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.get_score
        feature_score_gain_temp = bst.get_score(importance_type='gain')
        feature_score_gain_temp = pd.DataFrame(feature_score_gain_temp, index =[i])
        
        feature_score_cover_temp = bst.get_score(importance_type='cover')
        feature_score_cover_temp = pd.DataFrame(feature_score_cover_temp, index =[i])
        
        feature_score_weight_temp = bst.get_score(importance_type='weight')
        feature_score_weight_temp = pd.DataFrame(feature_score_weight_temp, index =[i])
#        if i == 0:
#                feature_score_temp_0 = feature_score_temp
#        if i == 1:
#                feature_score_temp_1 = feature_score_temp
#        if i == 2:
#                feature_score_temp_2 = feature_score_temp
#        if i == 3:
#                feature_score_temp_3 = feature_score_temp
#        if i == 4:
#                feature_score_temp_4 = feature_score_temp                
#        feature_score_temp = feature_score_temp.fillna(0)
        if i == 0:
                # convert dictionary to dataframe
                # https://stackoverflow.com/questions/17839973/construct-pandas-dataframe-from-values-in-variables
                # https://stackoverflow.com/questions/38599912/index-must-be-called-with-a-collection-of-some-kind-assign-column-name-to-dataf
                feature_score_gain_summary = feature_score_gain_temp.copy()
                feature_score_cover_summary = feature_score_cover_temp.copy()
                feature_score_weight_summary = feature_score_weight_temp.copy()
        else:
                feature_score_gain_summary = feature_score_gain_summary.append(feature_score_gain_temp)     
                feature_score_cover_summary = feature_score_cover_summary.append(feature_score_cover_temp) 
                feature_score_weight_summary = feature_score_weight_summary.append(feature_score_weight_temp)
                
        # Confusion Matrix
        CM_temp = confusion_matrix(test_Y, pred)
        if Is_save_file == 1:
                f = open('Confusion_Matrix_for_Multi_Iteration.txt','ab')
                np.savetxt(f, CM_temp, fmt='%i', delimiter=' ')
                f.close()
        
        if i == 0:
                CM = CM_temp
        else:
                CM += CM_temp
                        
        # plot feature importance
        ax = xgb.plot_importance(bst)
        fig = ax.figure
        fig.set_size_inches(5, 10)
        fig.savefig('feature importance.png', format='png', dpi=300)
        
        # plot single tree
        plot_tree(bst, rankdir='LR')
        plt.show()
        
        error_summary[i] = sum( int(pred[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y));
        print ('multi:softmax predicting, classification error=%f' % error_summary[i])
        
        print(classification_report(test_Y, pred, target_names=target_names))
        class_report = classification_report(test_Y, pred, target_names=target_names)
        
        precision, recall, fscore, support = score(test_Y, pred)
        report_temp = np.concatenate((precision, recall, fscore, support), axis=0)
        report_temp = np.asmatrix(report_temp).T

        if i == 0:
                # by define, = means assigning reference
                report_summary = report_temp.copy()
                report = report_temp.copy()
        else:
                report += report_temp.copy()
                report_summary = np.concatenate((report_summary, report_temp), axis=1)
#        break

# sum the number of feature used for multiple cross validation
#https://stackoverflow.com/questions/25748683/pandas-sum-dataframe-rows-for-given-columns
feature_score_gain_sum = feature_score_gain_summary.sum(axis=0)
feature_score_cover_sum = feature_score_cover_summary.sum(axis=0)
feature_score_weight_sum = feature_score_weight_summary.sum(axis=0)

if Is_save_file == 1:
        f = open('Confusion_Matrix_for_Multi_Iteration_Final.txt','wb')
        np.savetxt(f, CM, fmt='%i', delimiter=' ')
        np.savetxt(f, 1/(i+1)*CM, fmt='%f', delimiter=' ')
        f.close()
        
        f = open('Classification_Report_for_Multi_Iteration_Final.txt','wb')       
        np.savetxt(f, 1/(i+1)*report, fmt='%f', delimiter=' ')
        f.close()
        
        f = open('Classification_Summary_for_Multi_Iteration.txt','wb')
        np.savetxt(f, report_summary, fmt='%f', delimiter=' ')
        f.close()
        
        f = open('Feature_score_weight_summary.txt','wb')
        np.savetxt(f, feature_score_weight_summary, fmt='%i', delimiter=' ')
        f.close()
        
        f = open('Feature_score_gain_summary.txt','wb')
        np.savetxt(f, feature_score_gain_summary, fmt='%i', delimiter=' ')
        f.close()
        
        f = open('Feature_score_cover_summary.txt','wb')
        np.savetxt(f, feature_score_cover_summary, fmt='%i', delimiter=' ')
        f.close()