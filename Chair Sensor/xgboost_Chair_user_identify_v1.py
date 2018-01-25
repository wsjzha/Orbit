# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:01:54 2017

@author:  Zhenhua
"""

# First XGBoost model for Pima Indians dataset
import numpy
import xgboost as xgb
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from xgboost import plot_tree


#def perf_measure(y_actual, y_hat):
#     TP = 0
#     FP = 0
#     TN = 0
#     FN = 0
#
#     for i in range(len(y_hat)): 
#         if y_actual[i]==y_hat[i]:
#            TP += 1
#     for i in range(len(y_hat)): 
#         if y_hat[i]==1 and y_actual!=y_hat[i]:
#            FP += 1
#     for i in range(len(y_hat)): 
#         if y_actual[i]==y_hat[i]==0:
#            TN += 1
#     for i in range(len(y_hat)): 
#         if y_hat[i]==0 and y_actual!=y_hat[i]:
#            FN += 1
#     return(TP, FP, TN, FN)


# load data
feature_num = 80;
num_class = 27;
#dataset = numpy.loadtxt('C:/Users/Dest/OneDrive/Chair Sensor/Feature_Summary_xgboost.csv', 
#                        delimiter=",", converters={176: lambda x:int(x)-1 })
dataset = numpy.loadtxt('C:/Users/Dest/OneDrive/Chair Sensor/Feature_Summary_xgboost.csv', 
                        delimiter=",", converters={feature_num: lambda x:int(x)-1 })
# split data into X and y
X = dataset[:,0:feature_num]
Y = dataset[:,feature_num]


# setup parameters for xgboost
param = {}

# use softmax multi-class classification
param['objective'] = 'multi:softmax'

# scale weight of positive examples
param['eta'] = 0.1;
param['max_depth'] = 10;
param['silent'] = 1;
param['nthread'] = 4;
param['num_class'] = num_class;

# split data into train and test sets
seed = 1;
test_size = 0.4;
train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(	X, Y, test_size=test_size, random_state=seed)

xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)

watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
num_round = 1
bst = xgb.train(param, xg_train, num_round, watchlist );
# get prediction
pred = bst.predict( xg_test );

# feature importance
xgb.plot_importance(bst)

# plot single tree
plot_tree(bst, rankdir='LR')
plt.show()

target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8', 'class 9', 'class 10', 'class 11', 'class 12', 'class 13', 'class 14']
# TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
print ('multi:softmax predicting, classification error=%f' % (sum( int(pred[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))

print(classification_report(test_Y, pred, target_names=target_names))

#TP, FP, TN, FN = perf_measure(test_Y, pred)
#print('True Positive: %f' %TP)
# do the same thing again, but output probabilities
#param['objective'] = 'multi:softprob'
#bst = xgb.train(param, xg_train, num_round, watchlist );
## Note: this convention has been changed since xgboost-unity
## get prediction, this is in 1D array, need reshape to (ndata, nclass)
#yprob = bst.predict( xg_test ).reshape( test_Y.shape[0], num_class )
#ylabel = numpy.argmax(yprob, axis=1)
#
#print ('multi:softprob predicting, classification error=%f' % (sum( int(ylabel[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))
#
#print(classification_report(test_Y, ylabel, target_names=target_names))

#print ('running cross validation')
# do cross validation, this will print result out as
# [iteration]  metric_name:mean_value+std_value
# std_value is standard deviation of the metric
#xgboost.cv(	param, dtrain, num_round, 
#		nfold=5, metrics={'error'}, seed = 0,
#		callbacks=[xgboost.callback.print_evaluation(show_stdv=True)])



# fit model no training data
# booster=gbtree
#model = xgboost.XGBClassifier(	max_depth=10, learning_rate=0.1, n_estimators=100, objective='multi:softmax', silent=False)
