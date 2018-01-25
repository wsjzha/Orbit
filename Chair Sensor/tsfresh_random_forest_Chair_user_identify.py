# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:01:54 2017

@author:  Zhenhua
"""

# First XGBoost model for Pima Indians dataset
import numpy as np
from sklearn import cross_validation
from sklearn.model_selection  import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
import matplotlib.pyplot as plt
from xgboost import plot_tree

import pandas as pd
import time 
from datetime import datetime
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

# function for balancing the samples
def balanced_subsample(x,y,subsample_size=1.0):
    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys


# some data about the system
# the tag 
num_tags = 16;
num_seconds = 11;

# load data
num_samples = num_tags*num_seconds;

dataset = np.loadtxt('C:/Users/Dest/OneDrive/Chair Sensor/Feature_Summary_xgboost.csv',
                     delimiter=",", converters={num_samples: lambda x:int(x)-1 })

## load tsfresh features
#df_unselected = pd.read_csv(r"1500304996.3159575_imputed_no_index.csv")
#df_selected = pd.read_csv(r"1500304996.3159575_imputed_feature_selected_no_index.csv")

## number of features
#num_feature_selected = len(df_selected.columns);
#num_feature_unselected = len(df_unselected.columns);
#
## convert data frame to float
#feature_selected = df_selected.values
#feature_unselected = df_unselected.values

# split data into X and y
X = dataset[:,0:num_samples]
#X = feature_selected
Y = dataset[:,num_samples]

num_class = len(set(Y));

## insert labels for features into dataframe
#df_selected_labeled = label_features(df_selected, num_feature_selected)
#df_unselected_labeled = label_features(df_unselected, num_feature_unselected)

# setup parameters for xgboost
num_tree = 200;
min_split = 10;
min_leaf = 3;
num_threads = -1;
num_max_features = 0.33

# number of iterations
num_iter = 100;

# size for split and the random seed range

seed_max = num_iter*num_iter;
test_Size = 0.4;

# testing parameters
Is_save_file = 0;

# record error
error_summary = np.zeros(num_iter);

# target list
target_names = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5', 
                'class 6', 'class 7', 'class 8', 'class 9', 'class 10', 'class 11', 
                'class 12', 'class 13', 'class 14', 'class 15', 'class 16', 
                'class 17', 'class 18', 'class 19', 'class 20', 'class 21',
                'class 22', 'class 23', 'class 24', 'class 25', 'class 26', 'class 27']

# primary user case
Is_primary_user = 1;
Is_sub_sample = 0;
num_primary_user = 5;


# start of the iterations of cross validation
for i in range(num_iter):
        print('iteration ',i)
        
        # a single iteration, split the data into training group and testing group
        seed = random.seed(datetime.now())
        if Is_primary_user == 1:
                primary_user_ID = random.sample(range(0,len(set(Y))-1), num_primary_user)
                temp_Y = Y.copy()
                for j in range(len(temp_Y)):
                        if not (temp_Y[j] in primary_user_ID):
                                temp_Y[j] = num_class
        else:
                temp_Y = Y
                
        if Is_sub_sample == 1:
                sub_X, temp_sub_Y = balanced_subsample(X, temp_Y, subsample_size=1.0)
        else:
                temp_sub_Y = temp_Y
                sub_X = X
        train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(sub_X, temp_sub_Y, test_size=test_Size, random_state=seed)
        
        
#        rf_train = np.DMatrix(train_X, label=train_Y)
#        rf_test = np.DMatrix(test_X, label=test_Y)

        rf = RandomForestClassifier(n_estimators=num_tree, min_samples_split=min_split, 
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
        print ('multi:softmax predicting, classification error=%f' % error_summary[i])
        
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
        f = open('Confusion_Matrix_for_Multi_Iteration_Final.txt','ab')
        np.savetxt(f, CM, fmt='%i', delimiter=' ')
        np.savetxt(f, 1/(i+1)*CM, fmt='%f', delimiter=' ')
        f.close()
        
        f = open('Classification_Report_for_Multi_Iteration_Final.txt','ab')       
        np.savetxt(f, 1/(i+1)*report, fmt='%f', delimiter=' ')
        f.close()
        
        f = open('Classification_Summary_for_Multi_Iteration.txt','ab')
        np.savetxt(f, report_summary, fmt='%f', delimiter=' ')
        f.close()
        
np.mean(error_summary)
np.std(error_summary)

print(set(test_Y))