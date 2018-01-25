# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:01:54 2017

@author:  Zhenhua
"""

# First XGBoost model for Pima Indians dataset
import numpy as np

import pandas as pd
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute

import time 
#import multiprocessing
# tsfresh uses multiprocessing techniques, which require special support for windows
# https://docs.python.org/2/library/multiprocessing.html
if __name__ == '__main__':
        # some data about the system
        # the tag 
        num_tags = 16;
        num_seconds = 11;
        
        # load data
        num_samples = 176;
        feature_num = 176;
        num_class = 22;
        
        dataset = np.loadtxt('C:/Users/Dest/OneDrive/Chair Sensor/Feature_Summary_xgboost.csv', 
                                delimiter=",", converters={num_samples: lambda x:int(x)-1 })
        
        
        df = pd.DataFrame([])
        # format the raw data for tsfresh feature extraction
        # https://tsfresh.readthedocs.io/en/latest/text/data_formats.html
        pd.set_option('display.float_format', lambda x: '%.6f' % x)
        for x in dataset:
                temp_x = x[:num_samples]
                
                print(temp_x)
                temp_df_initial = pd.DataFrame({ 'id': 1,
                                     'time': np.arange(num_seconds),
                                     'tag_2006': temp_x[0:num_seconds],
                                     'tag_2008': temp_x[num_seconds:2*num_seconds],
                                     'tag_2026': temp_x[2*num_seconds:3*num_seconds],
                                     'tag_2036': temp_x[3*num_seconds:4*num_seconds],
                                     'tag_2107': temp_x[4*num_seconds:5*num_seconds],
                                     'tag_2110': temp_x[5*num_seconds:6*num_seconds],
                                     'tag_2113': temp_x[6*num_seconds:7*num_seconds],
                                     'tag_2117': temp_x[7*num_seconds:8*num_seconds],
                                     'tag_2124': temp_x[8*num_seconds:9*num_seconds],
                                     'tag_2171': temp_x[9*num_seconds:10*num_seconds],
                                     'tag_2210': temp_x[10*num_seconds:11*num_seconds],
                                     'tag_2223': temp_x[11*num_seconds:12*num_seconds],
                                     'tag_2331': temp_x[12*num_seconds:13*num_seconds],
                                     'tag_2332': temp_x[13*num_seconds:14*num_seconds],
                                     'tag_2410': temp_x[14*num_seconds:15*num_seconds],
                                     'tag_2964': temp_x[15*num_seconds:16*num_seconds]})
                temp_df = temp_df_initial[['id', 'time', 'tag_2006', 'tag_2008', 
                                           'tag_2026', 'tag_2036', 'tag_2107', 
                                           'tag_2110', 'tag_2113', 'tag_2117', 
                                           'tag_2124', 'tag_2171', 'tag_2210',
                                           'tag_2223', 'tag_2331', 'tag_2332', 
                                           'tag_2410', 'tag_2964' ]]

                temp_extracted_features = extract_features(temp_df, column_id="id",
                                                           column_sort="time",
                                                           column_kind=None, 
                                                           column_value=None) 

                df = df.append(temp_extracted_features, ignore_index = True)
#                break;
        

        # save the features into a csv file
        current_time = time.time()
        df.to_csv(str(current_time) + '_no_index.csv', index=False)
        
        # list the class label in Y
        Y = dataset[:,feature_num]
        y = pd.Series(Y)
        
#        # for test purpose
#        df = pd.read_csv(r"C:\WinPython-64bit-3.6.1.0Qt5\notebooks\Chair Sensor\user_identifcaiton_imputed_no_index.csv")
#        copy_df = df
        
        # inmpute features, so inf and nan are gone
        impute(df)
        df.to_csv(str(current_time) + '_imputed_no_index.csv', index=False)
        
        # remove useless features  - to be check whether they are truely useless
        features_filtered = select_features(df, y)
        features_filtered.to_csv(str(current_time) + '_imputed_feature_selected_no_index.csv', index=False)
        