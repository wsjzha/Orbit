# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:01:54 2017

@author:  Zhenhua
"""

# First XGBoost model for Pima Indians dataset
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
import matplotlib.pyplot as plt
from xgboost import plot_tree

import pandas as pd
import time 
import random

from sklearn.ensemble import RandomForestClassifier

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
file_path = 'C:/Users/Dest/OneDrive/Chair Sensor/Posture_raw_data - 08.05.2017.csv'
num_samples = num_tags*num_seconds+1;   # 1 is the identity label
dataset = np.loadtxt(file_path, delimiter=",", converters={num_samples: lambda x:int(x)-1 })

# consider the person's identity
Is_identity = 0;

if Is_identity == 1:
        X = dataset[:,0:num_samples]
else:
        X = dataset[:,0:num_samples-1]
        
# ignore some raw data(features)
#X = dataset[:,0:num_samples-1]

#X = feature_selected
Y = dataset[:,num_samples]

num_class = len(set(Y));

# setup parameters for xgboost
num_tree = 100;
tree_depth = 1;
min_split = 10;
min_leaf = 3;
num_threads = -1;
num_max_features = 0.33

# number of iterations
num_iter = 1;

# size for split and the random seed range
seed_max = num_iter*num_iter;
test_size = 0.4;

# testing parameters
Is_save_file = 0;

# record error
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
        
#        rf_train = np.DMatrix(train_X, label=train_Y)
#        rf_test = np.DMatrix(test_X, label=test_Y)

        rf = RandomForestClassifier(n_estimators=num_tree, min_samples_split=min_split, max_depth=tree_depth,
                       min_samples_leaf=min_leaf, max_features=num_max_features, n_jobs=num_threads, random_state=seed)
        rf.fit(train_X, train_Y)
        
        # get prediction
        pred = rf.predict( test_X );
        estimate_error = score(pred, test_Y, sample_weight=None)
        
        # feature importance
        # http://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.get_score
        feature_importance_temp = rf.feature_importances_   
        num_features = len(feature_importance_temp)
        feature_importance_temp = feature_importance_temp.reshape(num_features,1)
        feature_importance_temp = feature_importance_temp.transpose()
        

        if i == 0:
                # convert dictionary to dataframe
                feature_importance_summary = feature_importance_temp.copy()
        else:
                feature_importance_summary = np.append(feature_importance_summary, feature_importance_temp, axis=0)  
                
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
                        
        ## plot feature importance
        #ax = xgb.plot_importance(bst)
        #fig = ax.figure
        #fig.set_size_inches(20, 80)
        #fig.savefig('feature importance', format='png', dpi=300)
        #
        ## plot single tree
        #plot_tree(bst, rankdir='LR')
        #plt.show()
        
        error_summary[i] = sum( int(pred[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y));
        print ('rand_forest predicting, classification error=%f' % error_summary[i])
        
        print(classification_report(test_Y, pred, target_names=target_names))
        class_report = classification_report(test_Y, pred, target_names=target_names)
        
        precision, recall, fscore, support = score(test_Y, pred)
        report_temp = np.concatenate((precision, recall, fscore, support), axis=0)
        report_temp = np.asmatrix(report_temp).T

        if i == 0:
                # by define, “=” means assigning reference
                report_summary = report_temp.copy()
                report = report_temp.copy()
        else:
                report += report_temp.copy()
                report_summary = np.concatenate((report_summary, report_temp), axis=1)
#        break

# sum the number of feature used for multiple cross validation
#https://stackoverflow.com/questions/25748683/pandas-sum-dataframe-rows-for-given-columns
feature_importance_sum = feature_importance_summary.sum(axis=0)

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
        
        f = open('feature_importance_summary.txt','wb')
        np.savetxt(f, feature_importance_summary, fmt='%f', delimiter=' ')
        f.close()        
        
error_mean = np.mean(error_summary)
error_std = np.std(error_summary)

report_row_max = np.amax(report_summary, axis=1)
report_row_min = np.amin(report_summary, axis=1)
report_row_mean = report_summary.mean(1)
report_row_std = np.std(report_summary, axis=1)


# function for boxplot
def boxplot(report_summary, num_class):
        row, col = report_summary.shape
        precison = report_summary[0:num_class, :].copy()
        recall = report_summary[num_class:num_class, :].copy()
        f1_score = report_summary[2*num_class:num_class, :].copy()
        
        precison = np.transpose(precison)
        recall = np.transpose(recall)
        f1_score = np.transpose(f1_score)
        
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(6, 6), sharey=True)
        
        fs = 10;
        labels = list('ABCDEFGHIJKLM')
        axes[0, 1].boxplot(precison, labels=labels, showfliers=False)
        axes[0, 1].set_title('whis="range"', fontsize=fs)
        
        axes[0, 2].boxplot(recall, labels=labels, showfliers=False)
        axes[0, 2].set_title('whis="range"', fontsize=fs)
        
        axes[0, 3].boxplot(f1_score, labels=labels, showfliers=False)
        axes[0, 3].set_title('whis="range"', fontsize=fs)
        
        
boxplot(report_summary, num_class)        
        